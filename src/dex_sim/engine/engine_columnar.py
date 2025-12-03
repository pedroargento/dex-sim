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
    # Heterogeneity Params (New)
    pool_behavior_id: np.ndarray,
    behavior_params_array: np.ndarray, # Flattened array of params [momentum_str, mean_rev_str, random_scale, whale_scale...]
    rng_matrix_trades: np.ndarray,     # (P, T, MAX_TRADERS) or just pre-sampled deltas?
    rng_matrix_behavior: np.ndarray,   # (P, T, MAX_TRADERS)
):
    """
    Time-Major Columnar Simulation Engine with Heterogeneous Trader Behavior.
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
    MAX_TRADERS = len(pool_arrival_tick) 
    
    num_traders = MAX_TRADERS 
    
    # Arrays (P, N)
    positions = np.zeros((P, num_traders), dtype=np.float64)
    equities = np.zeros((P, num_traders), dtype=np.float64)
    im_locked = np.zeros((P, num_traders), dtype=np.float64)
    mm_required = np.zeros((P, num_traders), dtype=np.float64)
    start_step = np.full((P, num_traders), 0, dtype=np.int64)
    
    # Flags: 1=Active, 0=Dead/Defaulted. 
    is_active = np.ones((P, num_traders), dtype=np.int8)
    
    # Initial State
    prices = np.full(P, initial_price, dtype=np.float64)
    
    # --- Initialization (All Traders Active at T=0) ---
    for p in range(P):
        equities[p, :] = pool_equity
        # Initial positions
        positions[p, :] = (pool_notional / initial_price) * pool_direction
        
        # Initial Margin
        if im_is_es:
            base_rate = sigma_daily * im_factor
        else:
            base_rate = im_factor
        
        im_reqs = pool_notional * base_rate * margin_mult_grid[p, 0]
        im_locked[p, :] = im_reqs
        mm_required[p, :] = im_reqs * GAMMA
        
    price_paths[:, 0] = initial_price

    # Behavior Params Unpack
    # [mom_str, mr_str, rand_scale, whale_scale, hedger_scale, lp_var]
    mom_str = behavior_params_array[0]
    mr_str = behavior_params_array[1]
    rand_scale = behavior_params_array[2]
    whale_scale = behavior_params_array[3]
    hedger_scale = behavior_params_array[4]
    lp_var = behavior_params_array[5]

    # --- Main Time Loop ---
    for t in range(1, T):
        # 1. Price Update
        prev_prices = prices.copy()
        prices *= np.exp(log_returns[:, t])
        price_paths[:, t] = prices
        delta_prices = prices - prev_prices
        
        # Returns for behavior logic
        path_returns = pct_returns[:, t] # (P,)
        
        # 2. Context
        b_states = breaker_state_grid[:, t]
        m_mults = margin_mult_grid[:, t]
        
        # 3. PnL Update
        equities += positions * delta_prices.reshape(-1, 1)
        
        # 4. Behavioral Trading (The new core)
        for p in range(P):
            price_p = prices[p]
            ret_p = path_returns[p]
            b_state = b_states[p]
            m_mult = m_mults[p]
            
            if im_is_es:
                im_rate = sigma_daily * im_factor
            else:
                im_rate = im_factor
            im_rate *= m_mult
            
            # Pre-fetch row pointers
            row_pos = positions[p]
            row_eq = equities[p]
            row_im = im_locked[p]
            row_mm = mm_required[p]
            row_active = is_active[p]
            
            for i in range(num_traders):
                if row_active[i] == 0:
                    continue
                
                bid = pool_behavior_id[i]
                curr_pos = row_pos[i]
                curr_eq = row_eq[i]
                
                # Generate Delta Q (USD value)
                delta_usd = 0.0
                
                if bid == 0: # Momentum
                    if ret_p > 0:
                        delta_usd = curr_eq * mom_str
                    elif ret_p < 0:
                        delta_usd = -curr_eq * mom_str
                        
                elif bid == 1: # Mean Revert
                    if ret_p > 0:
                        delta_usd = -curr_eq * mr_str
                    elif ret_p < 0:
                        delta_usd = curr_eq * mr_str
                        
                elif bid == 2: # Random
                    delta_usd = 0.0 # Placeholder
                    
                # --- Execution Logic ---
                if abs(delta_usd) < 1e-9:
                    continue
                    
                delta_q = delta_usd / price_p
                
                # Breaker Gate (Hard)
                if b_state == 2:
                    if (curr_pos > 0 and delta_q > 0) or (curr_pos < 0 and delta_q < 0):
                        intent_rejected[p, t] += abs(delta_usd)
                        continue
                    if abs(curr_pos) < 1e-9:
                        intent_rejected[p, t] += abs(delta_usd)
                        continue
                
                # Margin Check
                new_pos = curr_pos + delta_q
                new_notional = abs(new_pos * price_p)
                new_im = new_notional * im_rate
                new_mm = new_im * GAMMA
                
                if curr_eq >= new_im:
                    # Execute
                    row_pos[i] = new_pos
                    row_im[i] = new_im
                    row_mm[i] = new_mm
                    
                    if b_state == 2:
                        intent_accepted_reduce[p, t] += abs(delta_usd)
                    else:
                        intent_accepted_normal[p, t] += abs(delta_usd)
                else:
                    intent_rejected[p, t] += abs(delta_usd)

        # 5. Liquidation (Vectorized or Loop)
        step_liqs = np.zeros(P, dtype=np.float64)
        
        for p in range(P):
            price_p = prices[p]
            row_eq = equities[p]
            row_mm = mm_required[p]
            row_pos = positions[p]
            row_im = im_locked[p]
            row_active = is_active[p]
            
            for i in range(num_traders):
                if row_active[i] == 0:
                    continue
                    
                if row_eq[i] < row_mm[i]:
                    # Liquidation logic (Partial/Full)
                    shortfall = row_mm[i] - row_eq[i]
                    notional_t = abs(row_pos[i] * price_p)
                    
                    if notional_t < 1e-9:
                        row_active[i] = 0 # Dead
                        continue
                    
                    k = 1.0
                    if do_partial_liquidation:
                        k = shortfall / notional_t
                        if k > 1.0: k = 1.0
                        
                    qty_close = row_pos[i] * k
                    cost = abs(qty_close * price_p) * slippage_factor
                    
                    # Update
                    row_eq[i] -= cost
                    slippage_cost[p, t] += cost
                    row_pos[i] -= qty_close
                    row_im[i] *= (1.0 - k)
                    row_mm[i] *= (1.0 - k)
                    mm_required[p, i] = row_mm[i] # Write back if scalar
                    
                    if k > step_liqs[p]:
                        step_liqs[p] = k
                    
                    # Default Check
                    if row_eq[i] < 0:
                        loss = -row_eq[i]
                        df_path[p, t] += loss
                        df_required[p] += loss
                        defaults[p] = 1
                        
                        # Kill
                        row_active[i] = 0
                        row_eq[i] = 0.0
                        row_pos[i] = 0.0
                        
            liquidation_fraction[p, t] = step_liqs[p]

        # 6. Aggregation (Vectorized)
        for p in range(P):
            p_pos = 0.0; p_neg = 0.0
            e_pos = 0.0; e_neg = 0.0
            row_pos = positions[p]
            row_eq = equities[p]
            row_act = is_active[p]
            
            for i in range(num_traders):
                if row_act[i]:
                    q = row_pos[i]
                    e = row_eq[i]
                    if q > 0:
                        p_pos += q
                        e_pos += e
                    elif q < 0:
                        p_neg -= q # Mag
                        e_neg += e
            
            notional_path[p, t] = (p_pos + p_neg) * prices[p] * 0.5
            equity_long_path[p, t] = e_pos
            equity_short_path[p, t] = e_neg
            
            if p_pos > 1e-9 and e_pos > 0:
                lev_long_path[p, t] = (p_pos * prices[p]) / e_pos
            if p_neg > 1e-9 and e_neg > 0:
                lev_short_path[p, t] = (p_neg * prices[p]) / e_neg

    # Return results...
    return (
        df_required, defaults, price_paths, lev_long_path, lev_short_path,
        liquidation_fraction, notional_path, equity_long_path, equity_short_path, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        # Placeholders for lifetimes/snapshots
        np.zeros(1, dtype=np.int64), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    )