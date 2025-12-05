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
    # initial_notional_target removed
    margin_mult_grid: np.ndarray,  # (P, T)
    breaker_state_grid: np.ndarray,  # (P, T)
    # IM config
    im_factor: float,
    im_is_es: bool,
    # Liquidation config
    slippage_factor: float,
    do_partial_liquidation: bool,
    gamma: float,
    # Trader pool template (1D, length N)
    pool_arrival_tick: np.ndarray,
    pool_notional: np.ndarray,
    pool_direction: np.ndarray,
    pool_equity: np.ndarray,
    pool_behavior_id: np.ndarray,
    # Behavior Config
    expand_rate: float,
    reduce_rate: float,
):
    """
    Columnar simulation engine with 2 Behaviors:
    0: Expanders (increase OI)
    1: Reducers (decrease OI)

    Symmetry: Net expansion is balanced to keep delta ~ 0.
    Breakers:
      NORMAL: Expand + Reduce allowed
      SOFT: Expand (higher IM) + Reduce allowed
      HARD: Reduce Only (Expand rejected)
    """
    P, T = log_returns.shape
    N = pool_notional.shape[0]

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------
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
    
    # ECP Arrays
    ecp_position = np.zeros(P, dtype=np.float64)
    ecp_position_path = np.zeros((P, T), dtype=np.float64)
    ecp_slippage_cost = np.zeros(P, dtype=np.float64) # Tracks slippage absorbed by system if any (though mostly trader pays)

    intent_accepted_normal = np.zeros((P, T), dtype=np.float64)
    intent_accepted_reduce = np.zeros((P, T), dtype=np.float64)
    intent_rejected = np.zeros((P, T), dtype=np.float64)

    # Placeholders for snapshots / lifetimes (API compatibility)
    trader_lifetime_counts = np.zeros(1, dtype=np.int64)
    trader_lifetime_pos = np.zeros(1, dtype=np.float64)
    trader_lifetime_eq = np.zeros(1, dtype=np.float64)
    trader_lifetime_im = np.zeros(1, dtype=np.float64)
    trader_lifetime_mm = np.zeros(1, dtype=np.float64)

    # --------------------------------------------------------
    # Internal state (P, N)
    # --------------------------------------------------------
    

    positions = np.zeros((P, N), dtype=np.float64)
    equities = np.zeros((P, N), dtype=np.float64)
    im_locked = np.zeros((P, N), dtype=np.float64)
    mm_required = np.zeros((P, N), dtype=np.float64)
    is_active = np.ones((P, N), dtype=np.int8)
    
    # Scratch space for desired deltas (reused per step)
    desired_delta_q = np.zeros((P, N), dtype=np.float64)

    prices = np.full(P, initial_price, dtype=np.float64)
    prev_prices = np.full(P, initial_price, dtype=np.float64)

    # --------------------------------------------------------
    # Initial state: symmetric OI from pool template
    # --------------------------------------------------------
    for p in range(P):
        for i in range(N):
            equities[p, i] = pool_equity[i]
            # positions start at 0 as pool_notional is now 0
            positions[p, i] = pool_direction[i] * pool_notional[i] / initial_price

            # IM base rate (approx)
            if im_is_es:
                base_rate = sigmas[p, 0] * im_factor
            else:
                base_rate = im_factor

            im_req = pool_notional[i] * base_rate * margin_mult_grid[p, 0]
            im_locked[p, i] = im_req
            mm_required[p, i] = im_req * gamma

    # Record t = 0
    price_paths[:, 0] = initial_price
    ecp_position_path[:, 0] = ecp_position
    
    # Initial aggregation
    for p in range(P):
        pos_L = 0.0; pos_S = 0.0; eq_L = 0.0; eq_S = 0.0
        for i in range(N):
            q = positions[p, i]
            e = equities[p, i]
            if q > 0.0:
                pos_L += q; eq_L += e
            elif q < 0.0:
                pos_S -= q; eq_S += e
        notional_path[p, 0] = (pos_L + pos_S) * prices[p] * 0.5
        equity_long_path[p, 0] = eq_L
        equity_short_path[p, 0] = eq_S
        if pos_L > 1e-9 and eq_L > 0.0:
            lev_long_path[p, 0] = (pos_L * prices[p]) / eq_L
        if pos_S > 1e-9 and eq_S > 0.0:
            lev_short_path[p, 0] = (pos_S * prices[p]) / eq_S

    step_liqs = np.zeros(P, dtype=np.float64)

    # --------------------------------------------------------
    # Main time loop
    # --------------------------------------------------------
    for t in range(1, T):
        # 1. Price & PnL Update
        for p in range(P):
            prev_prices[p] = prices[p]
            prices[p] = prev_prices[p] * np.exp(log_returns[p, t])
            price_paths[p, t] = prices[p]
            
            dP = prices[p] - prev_prices[p]
            # Optimization: PnL update can be skipped if positions are 0
            # But standard loop handles it
            for i in range(N):
                if is_active[p, i] == 0:
                    continue
                equities[p, i] += positions[p, i] * dP

        # 2. Behavior Logic (Expand/Reduce) with Symmetric Mirroring
        # Primary Traders: 0 to N/2 - 1
        # Mirror Traders: N/2 to N - 1
        half_N = N // 2
        
        for p in range(P):
            price_p = prices[p]
            ret_p = pct_returns[p, t] # previous return? log_returns is current step return.
            # "sign(previous_return)" - wait, behavior depends on return?
            # Prompt says: "Expanders ... delta_usd = expand_rate * equity * sign(previous_return)"
            # Wait, expanders follow trend? "push position away from zero". 
            # If flat, they need a direction.
            # "If flat, split evenly on initialization: half long, half short"
            # In new logic, they are flat at start.
            # So Expander A (i=0) might go Long, Mirror A (i=N/2) goes Short.
            # If Expander logic is trend-following:
            # If ret > 0 -> Buy. If ret < 0 -> Sell.
            # But let's stick to the prompt: "push position away from zero".
            # If pos > 0 -> Buy. If pos < 0 -> Sell.
            # If pos == 0 -> use `sign(previous_return)` or random?
            # Prompt: "delta_usd = expand_rate * equity * sign(previous_return)"
            # This implies trend following for entry.
            
            breaker = breaker_state_grid[p, t]
            m_mult = margin_mult_grid[p, t]
            
            if im_is_es:
                im_rate = sigmas[p, t] * im_factor
            else:
                im_rate = im_factor
            im_rate *= m_mult

            # We only iterate primary traders
            for i in range(half_N):
                mirror_idx = i + half_N
                
                # If either primary or mirror is dead, stop pair?
                # "Liquidation ... close only trader i ... do NOT adjust trader j"
                # But for NEW trades, if one is dead, symmetry breaks if we trade for the other.
                # "For any trade delta_q ... positions[i] += delta_q; positions[mirror] -= delta_q"
                # Implies if we can't apply to both, we shouldn't apply to either?
                # Or we accept asymmetry if one is dead?
                # "Net position = 0 at all times". This implies we MUST trade both or neither.
                if is_active[p, i] == 0 or is_active[p, mirror_idx] == 0:
                    continue
                
                eq_i = equities[p, i]
                pos_i = positions[p, i]
                bid = pool_behavior_id[i] # Mirror has same ID
                
                delta_usd = 0.0
                
                # Behavior Logic
                if bid == 0: # EXPANDER
                    if breaker < 2: # Not HARD
                        # Direction
                        direction = 0.0
                        if pos_i > 1e-9:
                            direction = 1.0
                        elif pos_i < -1e-9:
                            direction = -1.0
                        else:
                            # Flat: follow return sign
                            # If return is 0 (rare), use pool_direction/index parity?
                            if ret_p > 0:
                                direction = 1.0
                            elif ret_p < 0:
                                direction = -1.0
                            else:
                                # Fallback to alternating based on index
                                direction = 1.0 if (i % 2 == 0) else -1.0
                        
                        delta_usd = expand_rate * eq_i * direction
                
                elif bid == 1: # REDUCER
                    # Reduce towards zero
                    if abs(pos_i) > 1e-9:
                        reduce_usd = reduce_rate * eq_i
                        # Cap at full close
                        current_notional = abs(pos_i * price_p)
                        if reduce_usd > current_notional:
                            reduce_usd = current_notional
                        
                        if pos_i > 0:
                            delta_usd = -reduce_usd
                        else:
                            delta_usd = reduce_usd
                
                if abs(delta_usd) < 1e-9:
                    continue
                    
                delta_q = delta_usd / price_p
                
                # Gate: Breaker HARD (handled inside Expander check)
                
                # Gate: Margin Check
                # Check primary
                new_pos_i = pos_i + delta_q
                new_notional_i = abs(new_pos_i * price_p)
                new_im_i = new_notional_i * im_rate
                
                if eq_i < new_im_i:
                    intent_rejected[p, t] += abs(delta_usd)
                    continue
                    
                # Check mirror (symmetric opposite)
                pos_mirror = positions[p, mirror_idx]
                eq_mirror = equities[p, mirror_idx]
                
                # Mirror does opposite delta_q
                new_pos_mirror = pos_mirror - delta_q
                new_notional_mirror = abs(new_pos_mirror * price_p)
                new_im_mirror = new_notional_mirror * im_rate
                
                if eq_mirror < new_im_mirror:
                    intent_rejected[p, t] += abs(delta_usd)
                    continue
                
                # Accepted
                positions[p, i] = new_pos_i
                im_locked[p, i] = new_im_i
                mm_required[p, i] = new_im_i * gamma
                
                positions[p, mirror_idx] = new_pos_mirror
                im_locked[p, mirror_idx] = new_im_mirror
                mm_required[p, mirror_idx] = new_im_mirror * gamma
                
                val = abs(delta_usd)
                if bid == 0:
                    intent_accepted_normal[p, t] += val
                else:
                    intent_accepted_reduce[p, t] += val


        # 3. Liquidations (Iterate ALL traders)
        for p in range(P):
            price_p = prices[p]
            step_liqs[p] = 0.0

            if im_is_es:
                im_rate = sigmas[p, t] * im_factor
            else:
                im_rate = im_factor
            im_rate *= margin_mult_grid[p, t]

            for i in range(N):
                if is_active[p, i] == 0:
                    continue

                eq_i = equities[p, i]
                pos_i = positions[p, i]
                notional_i = abs(pos_i * price_p)
                
                # A. Auto-Flatten (Zombie Trader)
                # If equity is negligible, force close immediately
                if eq_i < 1e-8:
                    # Slippage cost on full close
                    cost = notional_i * slippage_factor
                    
                    # ECP Absorbs
                    ecp_position[p] -= pos_i
                    
                    # DF takes the hit (equity is basically 0, so loss = cost)
                    # Technically loss = cost - equity, but eq ~ 0.
                    loss = cost - eq_i
                    if loss > 0:
                        df_required[p] += loss
                        df_path[p, t] += loss
                        defaults[p] = 1 # Count as default? Yes, insolvent exit.
                    
                    slippage_cost[p, t] += cost
                    
                    # Clear State
                    equities[p, i] = 0.0
                    positions[p, i] = 0.0
                    im_locked[p, i] = 0.0
                    mm_required[p, i] = 0.0
                    is_active[p, i] = 0
                    continue

                # B. Margin Check
                im_i = notional_i * im_rate
                mm_i = im_i * gamma
                
                im_locked[p, i] = im_i
                mm_required[p, i] = mm_i

                if eq_i < mm_i:
                    if notional_i < 1e-9:
                        is_active[p, i] = 0
                        continue

                    shortfall = mm_i - eq_i
                    k = 1.0
                    if do_partial_liquidation:
                        k = shortfall / notional_i
                        if k > 1.0:
                            k = 1.0

                    qty_close = pos_i * k
                    cost = abs(qty_close * price_p) * slippage_factor

                    # Update Trader
                    equities[p, i] = eq_i - cost
                    positions[p, i] = pos_i - qty_close
                    
                    # ECP Absorbs the closed position
                    ecp_position[p] -= qty_close
                    
                    # Recalc margins after close
                    # New IM/MM proportional to remaining pos
                    im_locked[p, i] = im_i * (1.0 - k)
                    mm_required[p, i] = mm_i * (1.0 - k)

                    slippage_cost[p, t] += cost

                    if k > step_liqs[p]:
                        step_liqs[p] = k

                    # C. Insolvency Check (Gap Risk)
                    # If equity is negative after partial close (or full close), default.
                    if equities[p, i] < 0.0:
                        loss = -equities[p, i]
                        df_required[p] += loss
                        df_path[p, t] += loss
                        defaults[p] = 1

                        # Full remaining close out absorbed by ECP
                        remaining_pos = positions[p, i]
                        
                        equities[p, i] = 0.0
                        positions[p, i] = 0.0
                        
                        if abs(remaining_pos) > 1e-9:
                             ecp_position[p] -= remaining_pos
                        
                        im_locked[p, i] = 0.0
                        mm_required[p, i] = 0.0
                        is_active[p, i] = 0

            liquidation_fraction[p, t] = step_liqs[p]

        # 4. Aggregation
        for p in range(P):
            pos_L = 0.0; pos_S = 0.0; eq_L = 0.0; eq_S = 0.0

            for i in range(N):
                if is_active[p, i] == 0:
                    continue
                q = positions[p, i]
                e = equities[p, i]
                if q > 0.0:
                    pos_L += q; eq_L += e
                elif q < 0.0:
                    pos_S -= q; eq_S += e

            notional_path[p, t] = (pos_L + pos_S) * prices[p] * 0.5
            equity_long_path[p, t] = eq_L
            equity_short_path[p, t] = eq_S
            if pos_L > 1e-9 and eq_L > 0.0:
                lev_long_path[p, t] = (pos_L * prices[p]) / eq_L
            if pos_S > 1e-9 and eq_S > 0.0:
                lev_short_path[p, t] = (pos_S * prices[p]) / eq_S
            
            # Log ECP Position at end of step
            ecp_position_path[p, t] = ecp_position[p]

    return (
        df_required,
        defaults,
        price_paths,
        lev_long_path,
        lev_short_path,
        liquidation_fraction,
        notional_path,
        equity_long_path,
        equity_short_path,
        df_path,
        slippage_cost,
        intent_accepted_normal,
        intent_accepted_reduce,
        intent_rejected,
        trader_lifetime_counts,
        trader_lifetime_pos,
        trader_lifetime_eq,
        trader_lifetime_im,
        trader_lifetime_mm,
        ecp_position_path,
        ecp_slippage_cost,
    )