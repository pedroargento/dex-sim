import numpy as np
from numba import njit

@njit
def _run_simulation_loop_numba_columnar(
    log_returns: np.ndarray,
    pct_returns: np.ndarray,
    amihud_le: np.ndarray,
    sigmas: np.ndarray,
    initial_price: float,
    sigma_daily: float,
    initial_notional_target: float,
    margin_mult_grid: np.ndarray,
    breaker_state_grid: np.ndarray,
    # Config Params (Flattened)
    trader_arrival_enabled: bool,
    trader_arrival_pairs: int,
    trader_arrival_eq_val: float,
    trader_arrival_lev_min: float,
    trader_arrival_lev_max: float,
    # Model Params
    im_factor: float,
    im_is_es: bool,
    slippage_factor: float,
    do_partial_liquidation: bool,
    # Pre-allocated Trader Pools
    pool_arrival_tick: np.ndarray,
    pool_notional: np.ndarray,
    pool_direction: np.ndarray,
    pool_equity: np.ndarray,
):
    """
    Time-Major Columnar Simulation Engine.
    Processes all paths simultaneously at each tick (SIMD-friendly).
    """
    P, T = log_returns.shape
    
    # Outputs
    df_required = np.zeros(P, dtype=np.float64)
    defaults = np.zeros(P, dtype=np.int64)
    
    price_paths = np.zeros((P, T), dtype=np.float64)
    lev_long_path = np.full((P, T), np.nan, dtype=np.float64)
    lev_short_path = np.full((P, T), np.nan, dtype=np.float64)
    
    liquidation_fraction = np.zeros((P, T), dtype=np.float64)
    notional_path = np.zeros((P, T), dtype=np.float64)
    equity_long_path = np.zeros((P, T), dtype=np.float64)
    equity_short_path = np.zeros((P, T), dtype=np.float64)
    df_path = np.zeros((P, T), dtype=np.float64)
    slippage_cost = np.zeros((P, T), dtype=np.float64)

    intent_accepted_normal = np.zeros((P, T), dtype=np.float64)
    intent_accepted_reduce = np.zeros((P, T), dtype=np.float64)
    intent_rejected = np.zeros((P, T), dtype=np.float64)

    # Constants
    GAMMA = 0.8
    MAX_TRADERS = len(pool_arrival_tick) # Pool Size per Path
    # We allocate a dense matrix of (P, MAX_TRADERS)
    # But we only use up to active_count[p]
    # This might be huge. 
    # If P=5000, MAX_TRADERS=10000 -> 50M elements * 8 bytes = 400MB per array.
    # 5 arrays = 2GB. Feasible.
    
    # Columnar Arrays (P, MAX_TRADERS)
    positions = np.zeros((P, MAX_TRADERS), dtype=np.float64)
    equities = np.zeros((P, MAX_TRADERS), dtype=np.float64)
    im_locked = np.zeros((P, MAX_TRADERS), dtype=np.float64)
    mm_required = np.zeros((P, MAX_TRADERS), dtype=np.float64)
    start_step = np.full((P, MAX_TRADERS), -1, dtype=np.int64)
    
    active_counts = np.zeros(P, dtype=np.int64)
    
    # Initial State
    prices = np.full(P, initial_price, dtype=np.float64)
    
    # Pool Pointers per Path
    pool_ptrs = np.zeros(P, dtype=np.int64)
    
    # Lifetimes & Snapshots
    # For lifetimes, we can return flat array and reconstruct.
    # Snapshots: Store for Path 0 only to save memory.
    
    MAX_LIFETIMES = MAX_TRADERS * P
    trader_lifetimes = np.zeros(MAX_LIFETIMES, dtype=np.int64) # Placeholder logic
    lifetime_count = 0
    
    snap_pos = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_eq = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_lev = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_mm_usage = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_count = 0

    # --- Initialize T=0 ---
    if initial_notional_target > 0:
        # Vectorized Init for Path 0..P
        # We assume all paths start identical for this part
        # Longs
        positions[:, 0] = initial_notional_target / initial_price
        equities[:, 0] = initial_notional_target * 0.2
        start_step[:, 0] = 0
        
        # Shorts
        positions[:, 1] = -initial_notional_target / initial_price
        equities[:, 1] = initial_notional_target * 0.2
        start_step[:, 1] = 0
        
        active_counts[:] = 2
        
        # IM
        if im_is_es:
            base_rate = sigma_daily * im_factor
        else:
            base_rate = im_factor
            
        im_base = initial_notional_target * base_rate
        # margin_mult at t=0 is margin_mult_grid[:, 0]
        im_inits = im_base * margin_mult_grid[:, 0]
        mm_inits = im_inits * GAMMA
        
        im_locked[:, 0] = im_inits
        im_locked[:, 1] = im_inits
        mm_required[:, 0] = mm_inits
        mm_required[:, 1] = mm_inits

    # --- Main Time Loop ---
    for t in range(T):
        # 1. Update Prices
        # prices shape (P,)
        prev_prices = prices.copy()
        prices *= np.exp(log_returns[:, t])
        price_paths[:, t] = prices
        
        delta_prices = prices - prev_prices
        
        # 2. Context
        # breaker_state_grid shape (P, T) -> slice (P,)
        b_states = breaker_state_grid[:, t]
        m_mults = margin_mult_grid[:, t]
        
        # 3. PnL Update (Vectorized over active range)
        # Ideally we vectorize over (P, N).
        # But active_counts differ.
        # Strategy: Iterate max(active_counts) or just mask.
        # Masking: 
        # mask = (col_idx < active_counts[:, None])
        # equities += positions * delta_prices[:, None] * mask
        
        max_active = np.max(active_counts)
        
        # We only need to process up to max_active columns
        if max_active > 0:
            # Create mask for active traders
            # broadcasting: (P, 1) vs (1, max_active) -> (P, max_active)
            col_indices = np.arange(max_active)
            active_mask = col_indices < active_counts.reshape(-1, 1)
            
            # PnL
            # positions slice (P, max_active)
            # delta_prices (P,) -> (P, 1)
            pnl = positions[:, :max_active] * delta_prices.reshape(-1, 1)
            # Only apply to active
            # In Numba, advanced indexing/masking might be slower than loops if not careful.
            # But let's try explicit loop over P if needed, or flattened operations.
            # Flattened elementwise ops are best.
            
            # Update equities
            # Numba supports: equities[:, :max_active] += pnl * active_mask
            # But 'active_mask' is boolean.
            # We can just iterate P and vector-update row P?
            # "for p in range(P): equities[p, :ac[p]] += ..."
            # This is "semi-vectorized". 
            # True vectorization:
            # pnl_matrix = positions[:, :max_active] * delta_prices.reshape(-1, 1)
            # equities[:, :max_active] += pnl_matrix  (we add to dead traders too? No harm if 0 pos)
            # Since we swap-and-pop, dead traders are at end. 
            # Positions of dead traders should be 0.
            # So `positions * delta_p` is 0 for dead traders. 
            # So we can update the entire block `[:, :max_active]` safely!
            # EXCEPT: We must ensure `positions` for inactive slots are 0. (They are initialized 0).
            # And swap-and-pop must clear the old slot? Or just ensure count is decremented.
            # If we swap, the old slot becomes duplicate of last. We must clear the *new* empty slot?
            # Actually, swap-and-pop: `pos[i] = pos[last]`. We don't clear `pos[last]`.
            # So `pos[last]` remains non-zero garbage.
            # RISK: We update equity of garbage slot.
            # Does it matter? It's outside active_count.
            # Yes, unless we re-activate that slot later without clearing equity.
            # Activation overwrites equity. So it seems safe.
            
            equities[:, :max_active] += positions[:, :max_active] * delta_prices.reshape(-1, 1)

        # 4. Trader Arrival (Vectorized-ish)
        if trader_arrival_enabled:
            # We can't easily vectorize "while pool_ptr[p] < ...".
            # We have to loop P here.
            # But the inner logic is simple copy.
            for p in range(P):
                # Process arrivals for path p
                # Inline the pool spawn logic
                # Reuse pool_ptrs[p]
                
                while pool_ptrs[p] < MAX_TRADERS and pool_arrival_tick[pool_ptrs[p]] == t:
                    pid = pool_ptrs[p]
                    pool_ptrs[p] += 1
                    
                    # Breaker Check
                    trade_notional = pool_notional[pid]
                    direction = pool_direction[pid]
                    eq_val = pool_equity[pid]
                    price_p = prices[p]
                    q = (trade_notional / price_p) * direction
                    vol_usd = trade_notional
                    
                    if b_states[p] == 2: # HARD
                        intent_rejected[p, t] += vol_usd
                        continue
                    
                    # Margin Check
                    if im_is_es:
                        rate = sigma_daily * im_factor
                    else:
                        rate = im_factor
                    
                    im_req = trade_notional * rate * m_mults[p]
                    
                    if eq_val >= im_req:
                        # Spawn
                        tid = active_counts[p]
                        active_counts[p] += 1
                        
                        equities[p, tid] = eq_val
                        positions[p, tid] = q
                        im_locked[p, tid] = im_req
                        mm_required[p, tid] = im_req * GAMMA
                        start_step[p, tid] = t
                        
                        intent_accepted_normal[p, t] += vol_usd
                    else:
                        intent_rejected[p, t] += vol_usd

        # 5. Liquidation & Compaction (Semi-Vectorized)
        # Hard to fully vectorize conditional logic + swap-and-pop.
        # Loop over P, then inner loop over active_count.
        # This is the main compute kernel.
        
        step_liqs = np.zeros(P, dtype=np.float64)
        
        for p in range(P):
            ac = active_counts[p]
            price_p = prices[p]
            
            # Scan active traders
            # Use while loop for swap-and-pop
            i = 0
            while i < ac:
                # Load
                eq = equities[p, i]
                mm = mm_required[p, i]
                pos = positions[p, i]
                
                if eq < mm:
                    shortfall = mm - eq
                    notional_t = abs(pos * price_p)
                    
                    if notional_t < 1e-9:
                        # Dust - Kill
                        ac -= 1
                        if i != ac:
                            # Swap
                            positions[p, i] = positions[p, ac]
                            equities[p, i] = equities[p, ac]
                            im_locked[p, i] = im_locked[p, ac]
                            mm_required[p, i] = mm_required[p, ac]
                            start_step[p, i] = start_step[p, ac]
                        continue
                    
                    k = 1.0
                    if do_partial_liquidation:
                        k = shortfall / notional_t
                        if k > 1.0: k = 1.0
                    
                    qty_close = pos * k
                    cost = abs(qty_close * price_p) * slippage_factor
                    
                    # Update
                    eq -= cost
                    equities[p, i] = eq
                    slippage_cost[p, t] += cost
                    
                    pos -= qty_close
                    positions[p, i] = pos
                    
                    im_locked[p, i] *= (1.0 - k)
                    mm = mm * (1.0 - k)
                    mm_required[p, i] = mm
                    
                    if k > step_liqs[p]:
                        step_liqs[p] = k
                        
                    # Default Check
                    if eq < 0 or k == 1.0:
                        if eq < 0:
                            loss = -eq
                            df_path[p, t] += loss
                            df_required[p] += loss
                            defaults[p] = 1
                        
                        # Record Lifetime logic here (omitted for brevity in first pass)
                        
                        # Kill
                        ac -= 1
                        if i != ac:
                            positions[p, i] = positions[p, ac]
                            equities[p, i] = equities[p, ac]
                            im_locked[p, i] = im_locked[p, ac]
                            mm_required[p, i] = mm_required[p, ac]
                            start_step[p, i] = start_step[p, ac]
                        continue
                
                i += 1
            
            active_counts[p] = ac
            liquidation_fraction[p, t] = step_liqs[p]

        # 6. Aggregation (Vectorized)
        # We need to sum positions[:, :max_active] where column < active_count
        # Masking approach again.
        
        max_active = np.max(active_counts)
        if max_active > 0:
            col_indices = np.arange(max_active)
            # Mask: (P, max_active)
            mask = col_indices < active_counts.reshape(-1, 1)
            
            # Positions slice
            P_slice = positions[:, :max_active]
            E_slice = equities[:, :max_active]
            
            # Longs
            # Mask for positive positions AND active
            long_mask = (P_slice > 0) & mask
            short_mask = (P_slice < 0) & mask
            
            # Sum
            # np.sum(..., axis=1) -> (P,)
            pos_L = np.sum(P_slice * long_mask, axis=1)
            pos_S = np.sum(-P_slice * short_mask, axis=1) # Magnitude
            eq_L = np.sum(E_slice * long_mask, axis=1)
            eq_S = np.sum(E_slice * short_mask, axis=1)
            
            notional_path[:, t] = (pos_L + pos_S) * prices * 0.5
            equity_long_path[:, t] = eq_L
            equity_short_path[:, t] = eq_S
            
            # Leverage
            # Avoid div by zero
            # Numba vectorization for division?
            # Standard loop for safety or np.where
            
            for p in range(P):
                if eq_L[p] > 0 and pos_L[p] > 0:
                    lev_long_path[p, t] = (pos_L[p] * prices[p]) / eq_L[p]
                if eq_S[p] > 0 and pos_S[p] > 0:
                    lev_short_path[p, t] = (pos_S[p] * prices[p]) / eq_S[p]

    # End of Time Loop
    
    # TODO: Snapshot Path 0
    
    return (
        df_required, defaults, price_paths, lev_long_path, lev_short_path,
        liquidation_fraction, notional_path, equity_long_path, equity_short_path, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        trader_lifetimes[:lifetime_count], snap_pos[:snap_count], snap_eq[:snap_count], snap_lev[:snap_count], snap_mm_usage[:snap_count]
    )
