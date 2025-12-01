from typing import List, Dict, Tuple, Any
import numpy as np
import math
from numba import njit

from .mc_generator import MCReturnsGenerator
from .data_structures import MultiModelResults, SingleModelResults
from .models import RiskModel, FullCloseOut, PartialCloseOut
from .models.components import Trader

# ============================================================
#  Python Simulation Core (Multi-Trader)
# ============================================================

def _spawn_traders(
    traders: List[Trader],
    config, 
    current_price: float,
    step: int
) -> List[Tuple[int, float]]:
    """
    Spawns new trader pairs and generates initial trade intents.
    Returns a list of (trader_index, delta_notional) intents.
    """
    if not config.enabled:
        return []

    new_intents = []
    
    # Params
    pairs = config.pairs_per_tick
    eq_val = config.equity_dist_params.get("value", 10000.0)
    notional_sigma = config.notional_dist_params.get("sigma", 1.0)
    
    for _ in range(pairs):
        # Create two traders
        t1 = Trader(equity=eq_val, position=0.0, start_step=step)
        t2 = Trader(equity=eq_val, position=0.0, start_step=step)
        
        idx1 = len(traders)
        traders.append(t1)
        idx2 = len(traders)
        traders.append(t2)
        
        # Generate Notional (in USD)
        # Simple lognormal around a base or just raw lognormal? 
        # Assuming standard lognormal output is the multiplier or value?
        # Prompt says "sample notional size from configured distribution"
        # Let's assume lognormal(0, sigma) * 100,000 as a scaler? 
        # Or just lognormal(10, sigma)?
        # Let's aim for something reasonable relative to equity -> leverage 2x-10x
        # Base it on equity * random leverage
        lev_min, lev_max = config.leverage_range
        leverage = np.random.uniform(lev_min, lev_max)
        trade_notional = eq_val * leverage
        
        # Convert to units of asset (q)
        q = trade_notional / current_price
        
        # Intent: T1 Long, T2 Short
        new_intents.append((idx1, q))
        new_intents.append((idx2, -q))
        
    return new_intents

# ============================================================
#  Numba Simulation Core (Array-Based)
# ============================================================

@njit
def _run_simulation_loop_numba(
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
    im_factor: float, # 1/lev or es_factor
    im_is_es: bool,   # True if ES, False if Fixed (affects sigma scaling)
    slippage_factor: float,
    do_partial_liquidation: bool,
):
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

    # Constants & Preallocations
    # Max traders = Initial (2) + T * pairs * 2
    # Let's be safe and allocate somewhat generously. 
    # Numba works best with fixed arrays.
    MAX_TRADERS = 2 + T * trader_arrival_pairs * 2 + 100
    
    # Gamma
    GAMMA = 0.8

    # Lifetimes: We can't easily return variable length list from Numba.
    # We will fill a large array and return count or valid slice.
    MAX_LIFETIMES = MAX_TRADERS
    trader_lifetimes = np.zeros(MAX_LIFETIMES * P, dtype=np.int64) # Flattened
    lifetime_count = 0
    
    # Snapshots: Hard to do dicts in Numba. We will return structured arrays or separate arrays for Path 0.
    # Snapshots for Path 0 at T-1
    # We will return them as arrays: snap_pos, snap_eq, snap_lev, snap_mm
    snap_pos = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_eq = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_lev = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_mm_usage = np.zeros(MAX_TRADERS, dtype=np.float64)
    snap_count = 0

    for p in range(P):
        # --- Trader Arrays (Per Path) ---
        positions = np.zeros(MAX_TRADERS, dtype=np.float64)
        equities = np.zeros(MAX_TRADERS, dtype=np.float64)
        im_locked = np.zeros(MAX_TRADERS, dtype=np.float64)
        mm_required = np.zeros(MAX_TRADERS, dtype=np.float64)
        alive = np.zeros(MAX_TRADERS, dtype=np.int8)
        start_step = np.zeros(MAX_TRADERS, dtype=np.int64)
        
        next_trader_id = 0
        price = initial_price
        
        # Initialize ONE pair
        if initial_notional_target > 0:
            # Long
            tid = next_trader_id
            next_trader_id += 1
            alive[tid] = 1
            equities[tid] = initial_notional_target * 0.2
            positions[tid] = initial_notional_target / price
            start_step[tid] = 0
            
            # Short
            tid2 = next_trader_id
            next_trader_id += 1
            alive[tid2] = 1
            equities[tid2] = initial_notional_target * 0.2
            positions[tid2] = -initial_notional_target / price
            start_step[tid2] = 0
            
            # Initial Margin
            # IM = Notional * Factor (if Fixed) OR Notional * Sigma * Factor (if ES)
            # im_factor passed in should encapsulate this logic or we do it here.
            # If ES: im = notional * sigma * es_factor
            # If Fixed: im = notional * (1/lev)
            # Let's assume im_factor passed is correct multiplier for 'notional' or 'notional*sigma'
            # To simplify: calculate base_rate
            
            if im_is_es:
                base_im_rate = sigma_daily * im_factor
            else:
                base_im_rate = im_factor
            
            im_base = initial_notional_target * base_im_rate
            mult0 = margin_mult_grid[p, 0]
            im_init = im_base * mult0
            mm_init = im_init * GAMMA
            
            im_locked[tid] = im_init
            mm_required[tid] = mm_init
            im_locked[tid2] = im_init
            mm_required[tid2] = mm_init

        price_paths[p, 0] = price
        
        # Aggregate T=0
        pos_L = 0.0
        pos_S = 0.0
        eq_L = 0.0
        eq_S = 0.0
        
        for i in range(next_trader_id):
            if alive[i]:
                q = positions[i]
                e = equities[i]
                if q > 0:
                    pos_L += q
                    eq_L += e
                elif q < 0:
                    pos_S += -q # Store positive magnitude for OI
                    eq_S += e
        
        notional_path[p, 0] = (pos_L + pos_S) * price * 0.5
        equity_long_path[p, 0] = eq_L
        equity_short_path[p, 0] = eq_S
        
        if pos_L > 0 and eq_L > 0:
            lev_long_path[p, 0] = (pos_L * price) / eq_L
        if pos_S > 0 and eq_S > 0:
            lev_short_path[p, 0] = (pos_S * price) / eq_S

        dead_path = False

        for t in range(1, T):
            if dead_path and not trader_arrival_enabled:
                price_paths[p, t] = price
                continue

            # 1. Update Price
            ret_log = log_returns[p, t]
            price *= np.exp(ret_log)
            price_paths[p, t] = price
            
            price_prev = price / np.exp(ret_log)
            
            # 2. Context
            b_state = breaker_state_grid[p, t]
            m_mult = margin_mult_grid[p, t]
            
            # 3. PnL Updates
            # PnL = q * (price - price_prev)
            delta_p = price - price_prev
            
            for i in range(next_trader_id):
                if alive[i]:
                    pnl = positions[i] * delta_p
                    equities[i] += pnl
            
            # 4. Trader Arrival & New Trades
            if trader_arrival_enabled:
                # Generating intents inline
                for _ in range(trader_arrival_pairs):
                    # Spawn 2 traders
                    # We need 2 slots
                    if next_trader_id + 2 >= MAX_TRADERS:
                        break # Safety break
                        
                    t1 = next_trader_id
                    t2 = next_trader_id + 1
                    next_trader_id += 2
                    
                    alive[t1] = 1
                    alive[t2] = 1
                    start_step[t1] = t
                    start_step[t2] = t
                    equities[t1] = trader_arrival_eq_val
                    equities[t2] = trader_arrival_eq_val
                    positions[t1] = 0.0
                    positions[t2] = 0.0
                    im_locked[t1] = 0.0
                    im_locked[t2] = 0.0
                    mm_required[t1] = 0.0
                    mm_required[t2] = 0.0
                    
                    # Generate Intent
                    # lev = uniform(min, max)
                    lev = np.random.uniform(trader_arrival_lev_min, trader_arrival_lev_max)
                    trade_notional = trader_arrival_eq_val * lev
                    q = trade_notional / price
                    
                    # T1 Long (+q), T2 Short (-q)
                    # Process T1
                    
                    # Gate Check
                    accepted = True
                    if b_state == 2: # HARD
                        # Reduces exposure?
                        # If pos=0, any trade increases exposure -> Rejected
                        if positions[t1] == 0.0:
                            accepted = False
                        elif (positions[t1] > 0 and q < 0) or (positions[t1] < 0 and q > 0):
                            accepted = True
                        else:
                            accepted = False
                    
                    if not accepted:
                        intent_rejected[p, t] += abs(q * price)
                    else:
                        # Margin Check
                        # IM Rate
                        if im_is_es:
                            rate = sigma_daily * im_factor
                        else:
                            rate = im_factor
                        
                        im_req = abs(q * price) * rate * m_mult
                        mm_req = im_req * GAMMA
                        
                        if equities[t1] - im_locked[t1] >= im_req:
                            positions[t1] += q
                            im_locked[t1] += im_req
                            mm_required[t1] += mm_req
                            
                            if b_state == 2:
                                intent_accepted_reduce[p, t] += abs(q * price)
                            else:
                                intent_accepted_normal[p, t] += abs(q * price)
                        else:
                            intent_rejected[p, t] += abs(q * price)

                    # Process T2 (-q)
                    accepted = True
                    q2 = -q
                    if b_state == 2:
                        if positions[t2] == 0.0:
                            accepted = False
                        elif (positions[t2] > 0 and q2 < 0) or (positions[t2] < 0 and q2 > 0):
                            accepted = True
                        else:
                            accepted = False
                            
                    if not accepted:
                        intent_rejected[p, t] += abs(q2 * price)
                    else:
                        if im_is_es:
                            rate = sigma_daily * im_factor
                        else:
                            rate = im_factor
                        im_req = abs(q2 * price) * rate * m_mult
                        mm_req = im_req * GAMMA
                        
                        if equities[t2] - im_locked[t2] >= im_req:
                            positions[t2] += q2
                            im_locked[t2] += im_req
                            mm_required[t2] += mm_req
                            if b_state == 2:
                                intent_accepted_reduce[p, t] += abs(q2 * price)
                            else:
                                intent_accepted_normal[p, t] += abs(q2 * price)
                        else:
                            intent_rejected[p, t] += abs(q2 * price)

            # 5. Liquidation
            step_liq_fraction = 0.0
            
            for i in range(next_trader_id):
                if alive[i]:
                    if equities[i] < mm_required[i]:
                        shortfall = mm_required[i] - equities[i]
                        notional_t = abs(positions[i] * price)
                        
                        if notional_t < 1e-9:
                            continue
                        
                        k = 1.0
                        if do_partial_liquidation:
                            k = shortfall / notional_t
                            if k > 1.0: k = 1.0
                        
                        qty_close = positions[i] * k
                        cost = abs(qty_close * price) * slippage_factor
                        
                        equities[i] -= cost
                        slippage_cost[p, t] += cost
                        positions[i] -= qty_close
                        im_locked[i] *= (1.0 - k)
                        mm_required[i] *= (1.0 - k)
                        
                        if k > step_liq_fraction:
                            step_liq_fraction = k
                        
                        if equities[i] < 0:
                            loss = -equities[i]
                            df_path[p, t] += loss
                            df_required[p] += loss
                            defaults[p] = 1
                            alive[i] = 0 # Kill
                            equities[i] = 0.0
                            positions[i] = 0.0
                            
                            # Record Lifetime
                            if start_step[i] >= 0:
                                dur = t - start_step[i]
                                if lifetime_count < len(trader_lifetimes):
                                    trader_lifetimes[lifetime_count] = dur
                                    lifetime_count += 1
                                start_step[i] = -1 # Marked
            
            liquidation_fraction[p, t] = step_liq_fraction

            # 6. Aggregate
            pos_L = 0.0
            pos_S = 0.0
            eq_L = 0.0
            eq_S = 0.0
            
            for i in range(next_trader_id):
                if alive[i]:
                    q = positions[i]
                    e = equities[i]
                    if q > 0:
                        pos_L += q
                        eq_L += e
                    elif q < 0:
                        pos_S += -q
                        eq_S += e
            
            notional_path[p, t] = (pos_L + pos_S) * price * 0.5
            equity_long_path[p, t] = eq_L
            equity_short_path[p, t] = eq_S
            
            if pos_L > 1e-9:
                lev_long_path[p, t] = (pos_L * price) / eq_L if eq_L > 0 else np.nan
            else:
                lev_long_path[p, t] = np.nan
                
            if pos_S > 1e-9:
                lev_short_path[p, t] = (pos_S * price) / eq_S if eq_S > 0 else np.nan
            else:
                lev_short_path[p, t] = np.nan

        # End Path
        # If Path 0, take snapshots
        if p == 0:
            for i in range(next_trader_id):
                if alive[i] and abs(positions[i]) > 1e-9:
                    if snap_count < MAX_TRADERS:
                        snap_pos[snap_count] = positions[i] * price
                        snap_eq[snap_count] = equities[i]
                        # Lev = Notional / Equity
                        if equities[i] > 0:
                            snap_lev[snap_count] = abs(positions[i] * price) / equities[i]
                            snap_mm_usage[snap_count] = mm_required[i] / equities[i]
                        else:
                            snap_lev[snap_count] = 0.0
                            snap_mm_usage[snap_count] = 0.0
                        snap_count += 1
        
        # Remaining lifetimes
        for i in range(next_trader_id):
            if alive[i] and start_step[i] >= 0:
                dur = T - start_step[i]
                if lifetime_count < len(trader_lifetimes):
                    trader_lifetimes[lifetime_count] = dur
                    lifetime_count += 1

    # Trim lifetimes and snapshots
    final_lifetimes = trader_lifetimes[:lifetime_count]
    
    # Reconstruct snapshots list of dicts for compatibility
    # We can't do this in Numba easily. 
    # We will return the arrays and reconstruction happens in Python wrapper.
    
    return (
        df_required, defaults, price_paths, lev_long_path, lev_short_path,
        liquidation_fraction, notional_path, equity_long_path, equity_short_path, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        final_lifetimes, snap_pos[:snap_count], snap_eq[:snap_count], snap_lev[:snap_count], snap_mm_usage[:snap_count]
    )

def _run_simulation_loop(
    model: RiskModel,
    log_returns: np.ndarray,
    pct_returns: np.ndarray,
    amihud_le: np.ndarray,
    sigmas: np.ndarray,
    initial_price: float,
    sigma_daily: float,
    initial_notional_target: float,
    margin_mult_grid: np.ndarray,
    breaker_state_grid: np.ndarray,
):
    # Check backend selector
    # Since RiskModel object structure changes are limited, we can check an attribute or default to Python.
    # Prompt: "if model.backend == 'numba': use _run_simulation_loop_numba else: use _run_simulation_loop"
    # But _run_simulation_loop IS the python function.
    # We need to either rename the python one or put the switch in `run_models`.
    # The prompt says "Add: if model.backend == 'numba': ... in run_models()".
    # So `_run_simulation_loop` remains the Python implementation.
    
    P, T = log_returns.shape
    
    # Initialize Result Arrays
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

    # Trade Intent Logs
    intent_accepted_normal = np.zeros((P, T), dtype=np.float64)
    intent_accepted_reduce = np.zeros((P, T), dtype=np.float64)
    intent_rejected = np.zeros((P, T), dtype=np.float64)

    # Granular Data Containers (for worst case path or sample)
    # We collect lifetimes for ALL traders across ALL paths to get statistics
    # Using a simple list might grow large (5000 * 100 = 500k items), which is fine.
    all_trader_lifetimes = [] 
    
    # We collect snapshots ONLY for the path with max df_required (computed post-hoc? No, we don't know yet)
    # Or we can collect snapshots for Path 0 as a representative sample?
    # "Worst path" is only known after simulation.
    # Let's store snapshots for Path 0 always, and maybe update if we find a worse path?
    # Updating is complex. Let's just store Path 0 for distribution visualization. 
    # Usually Path 0 is random.
    # Better: Collect for Path 0. If the user wants worst case, we'd need to re-run or store all.
    # Given memory constraints, storing Path 0 snapshots is a safe default.
    representative_snapshots = [] 

    # Helper to compute MM from IM (gamma factor)
    GAMMA = 0.8

    print(f"Simulating {model.name} over {P} paths...")
    
    for p in range(P):
        # Per-path state
        traders: List[Trader] = []
        price = initial_price
        
        # Initialize ONE pair to match the requested initial_notional (backward compatibility)
        if initial_notional_target > 0:
            t_long = Trader(equity=initial_notional_target*0.2, position=initial_notional_target/price, start_step=0) # 5x lev
            t_short = Trader(equity=initial_notional_target*0.2, position=-initial_notional_target/price, start_step=0)
            
            # Initialize their margin
            im_base = model.initial_margin(initial_notional_target, sigma_daily)
            mult0 = margin_mult_grid[p, 0]
            im_init = im_base * mult0
            mm_init = im_init * GAMMA
            
            t_long.im_locked = im_init
            t_long.mm_required = mm_init
            t_short.im_locked = im_init
            t_short.mm_required = mm_init
            
            traders.append(t_long)
            traders.append(t_short)

        # Record t=0
        price_paths[p, 0] = price
        
        # Aggregate t=0
        pos_L = sum(max(t.position, 0) for t in traders)
        pos_S = sum(max(-t.position, 0) for t in traders)
        notional_path[p, 0] = (pos_L + pos_S) / 2.0 * price 
        equity_long_path[p, 0] = sum(t.equity for t in traders if t.position > 0)
        equity_short_path[p, 0] = sum(t.equity for t in traders if t.position < 0)
        
        if pos_L > 0 and equity_long_path[p, 0] > 0:
            lev_long_path[p, 0] = (pos_L * price) / equity_long_path[p, 0]
        if pos_S > 0 and equity_short_path[p, 0] > 0:
            lev_short_path[p, 0] = (pos_S * price) / equity_short_path[p, 0]

        dead_path = False

        for t in range(1, T):
            if dead_path and not model.trader_arrival.enabled:
                price_paths[p, t] = price
                continue

            # 1. Update Price
            ret_log = log_returns[p, t]
            ret_pct = pct_returns[p, t]
            price *= np.exp(ret_log)
            price_paths[p, t] = price
            
            # 2. Context
            b_state = breaker_state_grid[p, t]
            m_mult = margin_mult_grid[p, t]
            
            # 3. PnL Updates
            price_prev = price / np.exp(ret_log)
            for trader in traders:
                pnl = trader.position * (price - price_prev)
                trader.equity += pnl
                trader.unrealized_pnl += pnl
            
            # 4. Trader Arrival & New Trades
            if model.trader_arrival.enabled:
                new_intents = _spawn_traders(traders, model.trader_arrival, price, step=t)
                
                for t_idx, delta_q in new_intents:
                    trader = traders[t_idx]
                    
                    # Logging Volumes
                    vol_usd = abs(delta_q * price)
                    
                    # a. Breaker Gate (HARD = Reduce Only)
                    if b_state == 2: # HARD
                        if not trader.reduces_exposure(delta_q):
                            intent_rejected[p, t] += vol_usd
                            continue # Reject
                        else:
                            # Accepted Reduce
                            # Check margin next
                            pass
                    else:
                        # Normal/Soft -> Accepted Normal potential
                        pass
                    
                    # b. Margin Check for Opening
                    trade_notional = abs(delta_q * price)
                    im_base = model.initial_margin(trade_notional, sigma_daily)
                    im_req = im_base * m_mult
                    mm_req = im_req * GAMMA 
                    
                    if trader.equity - trader.im_locked >= im_req:
                        # Accept
                        trader.position += delta_q
                        trader.im_locked += im_req
                        trader.mm_required += mm_req
                        
                        # Log Acceptance
                        if b_state == 2:
                            intent_accepted_reduce[p, t] += vol_usd
                        else:
                            intent_accepted_normal[p, t] += vol_usd
                    else:
                        # Rejected due to Margin
                        intent_rejected[p, t] += vol_usd
            
            # 5. Liquidation Check
            step_liq_fraction = 0.0
            
            # Iterate copy or handle removal?
            # We won't remove traders from list to keep indices stable if needed, 
            # or we assume they just sit with 0 pos.
            # If equity < 0, they are dead.
            
            for trader in traders:
                if trader.equity < trader.mm_required:
                    shortfall = trader.mm_required - trader.equity
                    notional_t = abs(trader.position * price)
                    
                    if notional_t < 1e-9:
                        continue
                        
                    is_partial = isinstance(model.liquidation, PartialCloseOut)
                    k = 1.0
                    if is_partial:
                        k = shortfall / notional_t
                        if k > 1.0: k = 1.0
                    
                    qty_close = trader.position * k
                    cost = abs(qty_close * price) * model.liquidation.slippage_factor
                    
                    trader.equity -= cost
                    slippage_cost[p, t] += cost
                    trader.position -= qty_close
                    trader.im_locked *= (1.0 - k)
                    trader.mm_required *= (1.0 - k)
                    
                    step_liq_fraction = max(step_liq_fraction, k)
                    
                    if trader.equity < 0:
                        loss = -trader.equity
                        df_path[p, t] += loss
                        df_required[p] += loss
                        defaults[p] = 1
                        trader.equity = 0.0
                        trader.position = 0.0 
                        trader.im_locked = 0.0
                        trader.mm_required = 0.0
                        
                        # Record Lifetime
                        if trader.start_step >= 0:
                            duration = t - trader.start_step
                            all_trader_lifetimes.append(duration)
                            trader.start_step = -1 # Mark as recorded
            
            liquidation_fraction[p, t] = step_liq_fraction

            # 6. Aggregate & Store Stats
            pos_L = sum(max(t.position, 0) for t in traders)
            pos_S = sum(max(-t.position, 0) for t in traders)
            eq_L = sum(t.equity for t in traders if t.position > 0)
            eq_S = sum(t.equity for t in traders if t.position < 0)
            
            notional_path[p, t] = (pos_L * price + pos_S * price) / 2.0
            equity_long_path[p, t] = eq_L
            equity_short_path[p, t] = eq_S
            
            if pos_L > 1e-9:
                lev_long_path[p, t] = (pos_L * price) / eq_L if eq_L > 0 else np.nan
            if pos_S > 1e-9:
                lev_short_path[p, t] = (pos_S * price) / eq_S if eq_S > 0 else np.nan

        # End of Path: Collect snapshots if this is Path 0
        if p == 0:
            # Snapshot all active traders at final step
            for tr in traders:
                if abs(tr.position) > 1e-9: # Only active
                    snapshot = {
                        "position": tr.position * price, # USD value
                        "equity": tr.equity,
                        "leverage": (abs(tr.position * price) / tr.equity) if tr.equity > 0 else 0.0,
                        "mm_usage": (tr.mm_required / tr.equity) if tr.equity > 0 else 0.0
                    }
                    representative_snapshots.append(snapshot)
        
        # Record lifetimes for survivors
        for tr in traders:
            if tr.start_step >= 0 and tr.equity > 0:
                all_trader_lifetimes.append(T - tr.start_step)

    return (
        df_required, defaults, price_paths, lev_long_path, lev_short_path,
        liquidation_fraction, notional_path, equity_long_path, equity_short_path, df_path, slippage_cost,
        intent_accepted_normal, intent_accepted_reduce, intent_rejected,
        np.array(all_trader_lifetimes), representative_snapshots
    )


# ============================================================
#  Breaker / R_t computation (Reused)
# ============================================================


def _build_rt_and_mult(
    log_returns: np.ndarray,
    amihud_le: np.ndarray,
    sigmas: np.ndarray,
    breaker_soft: float,
    breaker_hard: float,
    breaker_multipliers: tuple[float, float, float],
):
    """
    Build systemic stress index R_t, breaker_state, and margin multiplier.
    (Same as before)
    """
    P, T = log_returns.shape

    # Tunable weights
    w_vol = 0.5
    w_le = 0.4
    w_sig = 0.1

    # sigmas is [P, T] already
    vol_term = w_sig * sigmas

    rt = w_vol * np.abs(log_returns) + w_le * amihud_le + vol_term

    breaker_state = np.zeros_like(rt, dtype=np.int8)
    breaker_state[(rt >= breaker_soft) & (rt < breaker_hard)] = 1
    breaker_state[rt >= breaker_hard] = 2

    mult0, mult1, mult2 = breaker_multipliers
    margin_mult = np.empty_like(rt, dtype=np.float64)
    margin_mult[breaker_state == 0] = mult0
    margin_mult[breaker_state == 1] = mult1
    margin_mult[breaker_state == 2] = mult2

    return rt, breaker_state, margin_mult


# ============================================================
#  Main engine
# ============================================================


def run_models(
    models: List[RiskModel],
    num_paths: int = 5_000,
    initial_price: float = 4000.0,
    notional: float = 400_000_000.0,
    stress_factor: float = 1.0,
    garch_params_file: str = "garch_params.json",
) -> MultiModelResults:
    """
    Multi-trader Monte-Carlo engine.
    """

    mc = MCReturnsGenerator(
        garch_params_file=garch_params_file,
        num_paths=num_paths,
        stress_factor=stress_factor,
    )
    log_returns, amihud_le, sigmas = mc.generate_paths(num_paths)
    horizon = mc.horizon
    sigma_daily = mc.last_sigma_t

    # Precompute pct returns for updates
    pct_returns = np.exp(log_returns) - 1.0

    model_results: Dict[str, SingleModelResults] = {}

    for model in models:
        # Pre-compute Breaker States
        soft = model.breaker.soft
        hard = model.breaker.hard
        mults = model.breaker.multipliers

        rt, breaker_state, margin_mult = _build_rt_and_mult(
            log_returns=log_returns,
            amihud_le=amihud_le,
            sigmas=sigmas,
            breaker_soft=soft,
            breaker_hard=hard,
            breaker_multipliers=mults,
        )

        # Backend Switching
        # Check if 'backend' attribute exists, defaults to 'python'
        backend = getattr(model, 'backend', 'python')
        
        if backend == 'numba':
            # Prepare Numba Args
            # Extract config values
            ta = model.trader_arrival
            ta_enabled = ta.enabled
            ta_pairs = ta.pairs_per_tick
            ta_eq_val = float(ta.equity_dist_params.get("value", 10000.0))
            ta_lev_min = float(ta.leverage_range[0])
            ta_lev_max = float(ta.leverage_range[1])
            
            # IM logic
            # Check IM type
            # Assuming standard IM types: ES_IM or FixedLeverageIM
            # We need to distill it down to `im_factor` and `im_is_es`
            # Use introspection on model.im
            im_is_es = hasattr(model.im, 'conf') # Heuristic or check class name
            if im_is_es:
                # ES: compute factor once. 
                # ES_IM.compute uses: sigma * es_factor.
                # We need es_factor.
                # Call internal logic? Or replicate.
                # model.im is an ES_IM instance. 
                # We can calculate ES factor from scipy stats here (in Python)
                # code: t_inv = t.ppf(conf, df); factor = ...
                # Or just call model.im.compute(1.0, 1.0) to get 'es_factor * 1 * 1'?
                # model.im.compute(1.0, 1.0) returns: 1.0 * es_factor * 1.0.
                # So im_factor = model.im.compute(1.0, 1.0) / 1.0 = es_factor.
                # Wait, compute(notional, sigma) -> expected_shortfall * notional
                # expected_shortfall = sigma * es_factor
                # So compute(1, 1) = 1 * es_factor * 1 = es_factor.
                im_factor = model.im.compute(1.0, 1.0)
            else:
                # Fixed: compute(notional, sigma) -> notional / leverage.
                # Factor = 1 / leverage.
                im_factor = 1.0 / model.im.leverage
            
            slippage = model.liquidation.slippage_factor
            do_partial = isinstance(model.liquidation, PartialCloseOut)

            (
                df_required,
                defaults_i,
                price_paths,
                lev_long,
                lev_short,
                liquidation_fraction,
                notional_path,
                equity_long,
                equity_short,
                df_path,
                slippage_cost,
                intent_accepted_normal,
                intent_accepted_reduce,
                intent_rejected,
                trader_lifetimes_flat,
                # Snapshots come as arrays
                snap_pos, snap_eq, snap_lev, snap_mm_usage
            ) = _run_simulation_loop_numba(
                log_returns=log_returns,
                pct_returns=pct_returns,
                amihud_le=amihud_le,
                sigmas=sigmas,
                initial_price=initial_price,
                sigma_daily=sigma_daily,
                initial_notional_target=notional,
                margin_mult_grid=margin_mult,
                breaker_state_grid=breaker_state,
                trader_arrival_enabled=ta_enabled,
                trader_arrival_pairs=ta_pairs,
                trader_arrival_eq_val=ta_eq_val,
                trader_arrival_lev_min=ta_lev_min,
                trader_arrival_lev_max=ta_lev_max,
                im_factor=im_factor,
                im_is_es=im_is_es,
                slippage_factor=slippage,
                do_partial_liquidation=do_partial
            )
            
            # Reconstruct snapshots dict list
            trader_snapshots = []
            for i in range(len(snap_pos)):
                trader_snapshots.append({
                    "position": snap_pos[i],
                    "equity": snap_eq[i],
                    "leverage": snap_lev[i],
                    "mm_usage": snap_mm_usage[i]
                })
            
            trader_lifetimes = trader_lifetimes_flat

        else:
            # Python Backend
            (
                df_required,
                defaults_i,
                price_paths,
                lev_long,
                lev_short,
                liquidation_fraction,
                notional_path,
                equity_long,
                equity_short,
                df_path,
                slippage_cost,
                intent_accepted_normal,
                intent_accepted_reduce,
                intent_rejected,
                trader_lifetimes,
                trader_snapshots
            ) = _run_simulation_loop(
                model=model,
                log_returns=log_returns,
                pct_returns=pct_returns,
                amihud_le=amihud_le,
                sigmas=sigmas,
                initial_price=initial_price,
                sigma_daily=sigma_daily,
                initial_notional_target=notional,
                margin_mult_grid=margin_mult,
                breaker_state_grid=breaker_state,
            )

        model_results[model.name] = SingleModelResults(
            name=model.name,
            df_required=df_required,
            defaults=defaults_i.astype(bool),
            price_paths=price_paths,
            lev_long=lev_long,
            lev_short=lev_short,
            rt=rt,
            breaker_state=breaker_state,
            margin_multiplier=margin_mult,
            # New Fields
            liquidation_fraction=liquidation_fraction,
            notional_paths=notional_path,
            equity_long=equity_long,
            equity_short=equity_short,
            df_path=df_path,
            slippage_cost=slippage_cost,
            # Pipeline
            intent_accepted_normal=intent_accepted_normal,
            intent_accepted_reduce=intent_accepted_reduce,
            intent_rejected=intent_rejected,
            trader_lifetimes=trader_lifetimes,
            trader_snapshots=trader_snapshots
        )

    return MultiModelResults(
        models=model_results,
        num_paths=num_paths,
        horizon=horizon,
        initial_price=initial_price,
        notional=notional,
        log_returns=log_returns,
        amihud_le=amihud_le,
        sigma_path=sigmas,
        metadata={"stress_factor": stress_factor, "sigma_daily": sigma_daily},
    )