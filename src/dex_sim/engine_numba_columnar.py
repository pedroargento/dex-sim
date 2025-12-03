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
    margin_mult_grid: np.ndarray,  # (P, T)
    breaker_state_grid: np.ndarray,  # (P, T)
    # IM config
    im_factor: float,
    im_is_es: bool,
    # Liquidation config
    slippage_factor: float,
    do_partial_liquidation: bool,
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
    GAMMA = 0.8

    positions = np.zeros((P, N), dtype=np.float64)
    equities = np.zeros((P, N), dtype=np.float64)
    im_locked = np.zeros((P, N), dtype=np.float64)
    mm_required = np.zeros((P, N), dtype=np.float64)
    is_active = np.ones((P, N), dtype=np.int8)
    
    # Scratch space for desired deltas (reused per step)
    # But since we parallelize over P implicitly via loop, we allocate (P, N)
    # Optimization: could be done inside loop but allocating outside is safer for numba
    desired_delta_q = np.zeros((P, N), dtype=np.float64)

    prices = np.full(P, initial_price, dtype=np.float64)
    prev_prices = np.full(P, initial_price, dtype=np.float64)

    # --------------------------------------------------------
    # Initial state: symmetric OI from pool template
    # --------------------------------------------------------
    for p in range(P):
        for i in range(N):
            equities[p, i] = pool_equity[i]
            positions[p, i] = pool_direction[i] * pool_notional[i] / initial_price

            # IM base rate (approx)
            if im_is_es:
                # Use the true ES factor calculated in Python
                # IM = sigma_t * es_factor * notional
                base_rate = sigmas[p, 0] * im_factor
            else:
                base_rate = im_factor

            im_req = pool_notional[i] * base_rate * margin_mult_grid[p, 0]
            im_locked[p, i] = im_req
            mm_required[p, i] = im_req * GAMMA

    # Record t = 0
    price_paths[:, 0] = initial_price
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
            for i in range(N):
                if is_active[p, i] == 0:
                    continue
                equities[p, i] += positions[p, i] * dP

        # 2. Behavior Logic (Expand/Reduce)
        # We process path by path to allow complex logic
        for p in range(P):
            price_p = prices[p]
            breaker = breaker_state_grid[p, t] # 0=Normal, 1=Soft, 2=Hard
            m_mult = margin_mult_grid[p, t]
            
            if im_is_es:
                # Use the true ES factor calculated in Python
                im_rate = sigmas[p, t] * im_factor
            else:
                im_rate = im_factor
            im_rate *= m_mult

            # A. Calculate Desired Deltas
            expand_long_notional = 0.0
            expand_short_notional = 0.0
            
            for i in range(N):
                if is_active[p, i] == 0:
                    desired_delta_q[p, i] = 0.0
                    continue
                
                eq = equities[p, i]
                pos = positions[p, i]
                bid = pool_behavior_id[i]
                
                d_q = 0.0
                
                # Behavior 0: EXPANDERS
                if bid == 0:
                    # Only expand if NOT Hard breaker
                    if breaker < 2:
                        desired_notional = expand_rate * eq
                        # Direction: same as current pos. If flat (shouldn't happen often), use pool_direction
                        if pos > 1e-9:
                            d_q = desired_notional / price_p
                        elif pos < -1e-9:
                            d_q = -desired_notional / price_p
                        else:
                            # Fallback to pool direction if flat
                            d_q = (desired_notional / price_p) * pool_direction[i]
                        
                        if d_q > 0:
                            expand_long_notional += d_q * price_p
                        else:
                            expand_short_notional += abs(d_q * price_p)

                # Behavior 1: REDUCERS
                elif bid == 1:
                    # Always allowed
                    desired_notional = reduce_rate * eq
                    # Direction: Opposite to current pos (reduce towards 0)
                    if pos > 1e-9:
                        d_q = -desired_notional / price_p
                        # Don't flip sign
                        if (pos + d_q) < 0:
                            d_q = -pos
                    elif pos < -1e-9:
                        d_q = desired_notional / price_p # Positive d_q to reduce neg pos
                        if (pos + d_q) > 0:
                            d_q = -pos
                    else:
                        d_q = 0.0

                desired_delta_q[p, i] = d_q

            # B. Symmetry Check (for Expanders)
            # Reducers reduce existing pos, so if system is balanced, reduction keeps it approx balanced.
            # We balance expansion to ensure no drift.
            imbalance = expand_long_notional - expand_short_notional
            scale_long = 1.0
            scale_short = 1.0
            
            if imbalance > 1e-9 and expand_long_notional > 1e-9:
                # Longs expanding too much, scale them down
                # Target = expand_short_notional
                scale_long = expand_short_notional / expand_long_notional
            elif imbalance < -1e-9 and expand_short_notional > 1e-9:
                # Shorts expanding too much
                scale_short = expand_long_notional / expand_short_notional

            # C. Execution Gate
            for i in range(N):
                d_q = desired_delta_q[p, i]
                if abs(d_q) < 1e-12:
                    continue
                
                bid = pool_behavior_id[i]
                
                # Apply symmetry scaling to expanders
                if bid == 0:
                    if d_q > 0:
                        d_q *= scale_long
                    else:
                        d_q *= scale_short
                
                if abs(d_q) < 1e-12:
                    continue

                curr_pos = positions[p, i]
                new_pos = curr_pos + d_q
                new_notional = abs(new_pos * price_p)
                new_im = new_notional * im_rate
                # Check Leverage Cap
                curr_eq = equities[p, i]
                if curr_eq <= 0:
                     continue
                
                # Margin Check: Equity >= IM_new? 
                # Only constraint is having enough equity to cover IM.
                if curr_eq >= new_im:
                    # Execute
                    positions[p, i] = new_pos
                    im_locked[p, i] = new_im
                    mm_required[p, i] = new_im * GAMMA
                    
                    usd_val = abs(d_q * price_p)
                    if bid == 1:
                         intent_accepted_reduce[p, t] += usd_val
                    else:
                         intent_accepted_normal[p, t] += usd_val
                else:
                    intent_rejected[p, t] += abs(d_q * price_p)


        # 3. Liquidations
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
                
                # Refresh MM based on current pos/price
                # (Position might have changed in step 2)
                pos_i = positions[p, i]
                notional_i = abs(pos_i * price_p)
                im_i = notional_i * im_rate
                mm_i = im_i * GAMMA
                
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

                    equities[p, i] = eq_i - cost
                    positions[p, i] = pos_i - qty_close
                    # Update im/mm for next step/metrics
                    im_locked[p, i] = im_i * (1.0 - k)
                    mm_required[p, i] = mm_i * (1.0 - k)

                    slippage_cost[p, t] += cost

                    if k > step_liqs[p]:
                        step_liqs[p] = k

                    if equities[p, i] < 0.0:
                        loss = -equities[p, i]
                        df_required[p] += loss
                        df_path[p, t] += loss
                        defaults[p] = 1

                        equities[p, i] = 0.0
                        positions[p, i] = 0.0
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
    )
