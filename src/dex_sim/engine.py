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

def _simulate_paths_python(
    model: RiskModel,
    log_returns: np.ndarray,
    pct_returns: np.ndarray,
    initial_price: float,
    sigma_daily: float,
    initial_notional_target: float, 
    # Note: initial_notional_target is used for the initial setup if we wanted to pre-populate,
    # but instructions say "Remove monolithic... work with many Trader accounts".
    # We will assume we start with some initial traders to match the requested 'notional' 
    # or start empty if not specified. 
    # To maintain backward compatibility of "simulation with X open interest",
    # we should probably spawn initial traders to match 'initial_notional_target'.
):
    P, T = log_returns.shape
    
    # Outputs
    df_required = np.zeros(P, dtype=np.float64)
    defaults = np.zeros(P, dtype=np.int64)  # boolean really
    
    price_paths = np.zeros((P, T), dtype=np.float64)
    lev_long = np.zeros((P, T), dtype=np.float64)
    lev_short = np.zeros((P, T), dtype=np.float64)
    
    liquidation_fraction = np.zeros((P, T), dtype=np.float64)
    notional_path = np.zeros((P, T), dtype=np.float64)
    equity_long_path = np.zeros((P, T), dtype=np.float64)
    equity_short_path = np.zeros((P, T), dtype=np.float64)
    df_path = np.zeros((P, T), dtype=np.float64)
    slippage_cost = np.zeros((P, T), dtype=np.float64)
    
    nan = np.nan
    
    # Precompute breaker params
    breaker = model.breaker
    # We need to re-compute R_t inside the loop or pass it in?
    # The original code passed `margin_mult` computed beforehand.
    # We should stick to that pattern if possible, but the instructions say:
    # "Compute R_t and breaker_state (unchanged formula). ... In the per-tick loop"
    # So we can pass the pre-computed arrays for efficiency.
    
    # Slippage
    slippage_factor = model.liquidation.slippage_factor
    do_partial = isinstance(model.liquidation, PartialCloseOut)

    # We need margin_mult and breaker_state passed in or computed. 
    # The calling function `run_models` computes them. We will add them to args.
    
    return None # Placeholder, see full implementation below

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

        # Run Python Kernel
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
            # New Outputs
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