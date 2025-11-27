# Proposal: Building a Cartesi-Based DEX (CCP) — Real-Time Risk Management Inside a Rollup

*Abstract*

This proposal introduces a Cartesi-based Central Counterparty (CCP) architecture for decentralized derivatives. It embeds continuous, deterministic risk management within a rollup environment, replacing traditional liquidation-based safety with adaptive margining and circuit-breaker logic. The design combines exchange-grade execution latency with on-chain verifiability, aiming to establish a self-contained, solvable clearing layer for DeFi.

## TL;DR

We propose an **in-rollup CCP (Central Counterparty) architecture** for decentralized derivatives.

The goal: replace static, reactive liquidation models in DeFi with **deterministic, continuous risk management** — running inside the Cartesi Rollup, with a **centralized sequencer** providing sub-second pre-confirmations for realistic market UX.

This design transforms DeFi exchanges into self-contained clearinghouses capable of enforcing risk dynamically, in real time, and provably on-chain.

---

## The Problem

Most DeFi perpetual DEXs (GMX, dYdX, Hyperliquid, etc.) share structural flaws:

- Fixed leverage tiers (5×, 10×, 20×).
- Margin calls triggered only when price moves hit preset levels.
- Liquidations executed by external keepers, often too late.
- No portfolio netting, no dynamic margin, no system-wide awareness.

When volatility spikes, these systems **cascade**: liquidations trigger more liquidations, insurance funds deplete, and volatility amplifies.

We want to fix that — not by building a new DEX, but by **rebuilding how DeFi clears risk**.

## Strategic Context

> Cartesi CCP Innovates on Three Fronts
> 
1. **In-Rollup Enforcement**
    
    All trading, margin, and liquidation logic runs *inside* the Cartesi Rollup.
    
    This makes risk management deterministic, tamper-proof, and instantly enforceable — without relying on off-chain keepers or oracles.
    
    Every margin call, breaker trigger, and auction occurs within the same verifiable execution layer.
    
2. **Incremental Risk Framework**
    
    The system continuously measures volatility, liquidity, and exposure using compact, recursive updates (EWMA, VaR, liquidity elasticity).
    
    This enables real-time margin recalibration without replaying history — a true *stream processor for market risk* operating in deterministic space.
    
3. **Dynamic Margining & Auctions**
    
    Leverage, margin, and liquidation rules are adaptive rather than fixed.
    
    When risk metrics tighten, leverage contracts; when volatility eases, it expands again.
    
    In stress events, the circuit-breaker FSM automatically transitions the system through **NORMAL → SOFT → HARD → RECOVERY**, coordinating auctions and breaker logic seamlessly inside the VM.
    

---

### Institutional Relevance & Regulatory Angle

- The **Cartesi** **CCP model** mirrors real-world clearinghouse standards (CME, Eurex, LCH), making it inherently more credible to institutional participants.
- The **in-rollup design** provides real enforcement — not advisory or probabilistic like oracle-based risk systems.
- The **sequencer layer** ensures sub-second UX, giving parity with centralized exchanges while retaining verifiability.
- There is room for **creative regulatory structuring**: by anchoring core risk logic on-chain but keeping interface layers modular, the system can pursue **regulatory arbitrage** or hybrid licensing — e.g., fully compliant on-shore wrappers accessing globally liquidity.

---

### Why It Matters

Traditional DeFi systems react to volatility; the Cartesi CCP *anticipates it*.

By embedding continuous risk computation and deterministic enforcement within a verifiable rollup, the design turns chaotic liquidation markets into orderly, provable clearing systems.

---

| Aspect | Existing Perp DEXs | Cartesi CCP |
| --- | --- | --- |
| Margining | Static | Dynamic (continuous IM/MM recalibration) |
| Liquidation | Reactive | Deterministic, auction-based |
| Enforcement | Keeper/Oracle | In-rollup logic |
| Data Scope | Per-position | Systemic (global Δ, Γ, VaR) |
| Speed | Off-chain + Oracle lag | Sub-second (sequencer pre-confirms) |

---

## Why a CCP Model

A **Central Counterparty (CCP)** is what keeps traditional markets solvent.

Every trade is reissued as `A ↔ CCP ↔ B`, so the CCP becomes the buyer to every seller and the seller to every buyer.

**Advantages of this model:**

- **Novation:** No bilateral counterparty exposure.
- **Central margining:** Unified risk and collateral management.
- **Dynamic margining:** Margin scales with volatility and exposure.
- **Default waterfall:** Structured loss absorption (margin → insurance → default fund).
- **Systemic awareness:** Risk measured globally, not per position.

In traditional finance, CCPs like CME Clearing or Eurex PRISMA run real-time risk models based on portfolio sensitivities (“Greeks”).

Cartesi allows us to implement something similar *on-chain*, deterministically.

---

## Understanding Leverage in DeFi

**Leverage = Position size ÷ Margin.**

It amplifies both gains and losses.

In current DeFi:

- Leverage is **static** (e.g., 10× always available).
- Margins are fixed percentages of notional value.
- Liquidations happen abruptly once equity < maintenance margin.

This creates “knife-edge” behavior — either fully solvent or instantly liquidated.

In a CCP model:

- Leverage becomes **dynamic**, adjusting continuously with market conditions.
- When volatility rises or one side of the market gets crowded, leverage tightens automatically.
- In calm markets, leverage relaxes again.

**Traders still get leverage,** but within a system that *prevents liquidation cascades* rather than cleaning up after them.

---

## What Matters to Traders

| Concern | Traditional DeFi | Cartesi CCP |
| --- | --- | --- |
| Execution speed | Fast but fragile | Fast via sequencer + deterministic in VM |
| Margin | Fixed % | Adaptive per volatility + system delta |
| Liquidation risk | Sudden, chaotic | Preemptive, gradual breaker logic |
| Transparency | Hidden risk formulas | Public risk metrics (VaR, Δ, ES) |
| Solvency | Local, isolated | Systemic and provable |
| Funding stability | Volatile | Stabilized via dynamic leverage asymmetry |

In short, **traders trade the same way**, but the environment stops being suicidal during stress.

---

## Role of the Centralized Sequencer

Without a sequencer, Cartesi’s determinism comes at the cost of latency.

With a sequencer:

- Traders get *soft pre-confirmations.*
- The system retains verifiability while achieving real exchange speed.

Think of it as a **two-phase commit**:

1. Soft commit (sequencer pre-confirm).
2. Hard commit (Cartesi confirmation + L1 anchoring).

## Why It Must Run Inside the Rollup

Rollups introduce a **dispute window** — typically a week — before L1 finalization.

That’s too slow for enforcing margin calls or breaker states on L1.

Only by running the trading logic **inside the Cartesi Machine** do we get:

- **Immediate enforcement** of margin and circuit breakers.
- **Consistent risk view** between matching, clearing, and liquidation.
- **Deterministic replayability** for audit and dispute defense.

---

## Incremental Risk Computation Framework

### Purpose

The Cartesi CCP must assess portfolio- and system-level risk continuously inside the Rollup VM, where computational cost matters.

Recomputing historical data each batch is infeasible, so the risk engine maintains **incremental state variables** that update recursively with every new block or trade batch.

This design mirrors how real clearinghouses (e.g., CME PRISMA, ICE SPAN) achieve real-time exposure tracking: by using *online estimators* instead of full time-series recomputation.

---

### Concept

At each new batch (tick):

1. The Cartesi adapter ingests minimal external data
    
    • ΔPrice, Volume, and Liquidity Depth (from Uniswap/Aave)
    
    • Block timestamp
    
    • Funding and volatility indices (EWMA σ²)
    
2. The risk engine updates its rolling statistics deterministically.
3. Updated metrics feed directly into:
    
    • Initial Margin (IM)
    
    • Maintenance Margin (MM)
    
    • Circuit-breaker state evaluation
    
    • System-wide exposure metrics (Δ_sys, Γ_sys, VaR_sys)
    

---

### Incremental Formulas

| Metric | Recursive Update |
| --- | --- |
| **Volatility (EWMA)** | σ²ₜ = λ σ²ₜ₋₁ + (1 – λ)(ΔPₜ)² |
| **Covariance Matrix** | Σₜ = (1 – λ) Σₜ₋₁ + λ (Δrₜ Δrₜᵀ) |
| **Liquidity Elasticity (LE)** | LEₜ = β₀ + β₁ · log(Vₜ / Depthₜ) |
| **VaR approximation** | VaRₜ = μₜ – zₐ σₜ |
| **Systemic Δ and Γ** | Δ_sys = Σ Δᵢ  Γ_sys = Σ Γᵢ |

Only the latest observation (ΔPₜ, Vₜ, Depthₜ) is needed each step — no historical replay required.

> ⚠️ Note on VaR Limitations
> 
> 
> Value-at-Risk (VaR) is used here only as a *baseline solvency metric*, **not a complete measure of risk**.
> 
> Traditional VaR models underestimate tail and liquidity risk, creating a false sense of security during crises.
> 
> The **Cartesi CCP** integrates VaR into a **composite risk index (Rₜ)** that also accounts for volatility, liquidity elasticity, and systemic Δ / Γ exposure.
> 
> This ensures that tail stress, liquidity shocks, and correlated drawdowns automatically trigger **margin and breaker adjustments**, not post-factum liquidations.
> 

---

### Performance Characteristics

| Aspect | Incremental Approach | Full Historical Approach |
| --- | --- | --- |
| Computation | O(N) per batch | O(T × N) |
| Data Access | Last block only | Entire history |
| VM Storage | Compact sufficient stats | Full time-series |
| Determinism | Fully reproducible | Complex replays |
| Cost Growth | Constant | Unbounded |

The CCP thus operates as a **stream processor for market risk** — continuous, low-latency, and auditable.

---

### Circuit-Breaker Transition Logic

$$
R_t = w_\Delta |\Delta_{sys}| + w_\Gamma \Gamma_{sys} + w_V VaR_{sys} + w_L LE_t
$$

Incremental metrics feed a composite **risk index Rₜ**, which governs the CCP’s operating mode:

### State Machine

| State | Meaning | Trigger (Rₜ thresholds) | CCP Actions |
| --- | --- | --- | --- |
| **NORMAL** | Market balanced | Rₜ < θ₁ | Standard margin and leverage |
| **SOFT** | Elevated risk or imbalance | θ₁ ≤ Rₜ < θ₂ | Raise IM multipliers; cap new OI; throttle orders |
| **HARD** | Systemic stress imminent | Rₜ ≥ θ₂ | Reduce-only mode; batch liquidations; activate default waterfall |
| **RECOVERY** | Normalization phase | Rₜ < θ₁ for n ticks | Gradually relax margins; resume normal trading |
- Thresholds (θ₁, θ₂) adapt with recent realized volatility and liquidity depth, ensuring the system tightens under stress and loosens in calm markets.
- Each transition is **logged deterministically** in VM state for audit and replay.
- The breaker FSM runs on every batch tick (e.g., 1–5 s).

---

### Research Directions

- Tune λ (memory decay) for stability vs reactivity.
- Benchmark EWMA vs GARCH(1,1) responses on crypto data.
- Calibrate θ₁, θ₂ to historical stress events (2021–2022 crashes).
- Test breaker transition latency and false-trigger rates under synthetic loads.

## Margin & Settlement Mechanics

### Overview

The Cartesi CCP performs **financial settlement only** — all positions are cash-settled in USDC (or a single stable collateral), not physically delivered.

Every trader maintains a **margin account** inside the Rollup VM that tracks:

- `Free Margin`: USDC not locked as collateral, available for new positions or withdrawal.
- `Initial Margin (IM)`: funds reserved to open and maintain active positions.
- `Variation Margin (VM)`: running unrealized PnL, marked-to-market each batch.
- `Equity`: `Free + IM + VM – fees`.

All PnL, funding, and liquidation flows occur inside the VM as deterministic balance updates.

---

### Mark-to-Market and PnL Realization

At every batch tick:

1. The composite mark price (P_mark) updates using Section 10’s MtM mechanism.
2. Each position’s unrealized PnL is computed as

$$
\text{PnL}_t = q_i \times (P_{\text{mark},t} - P_{\text{entry},i})

$$

1. PnL is added to (or subtracted from) the trader’s variation margin.
2. If equity drops below maintenance margin, the position enters the liquidation queue.
3. All adjustments are logged deterministically for audit and dispute proofs.

No asset delivery occurs — only USDC balances move between accounts.

---

### Initial Margin (IM)

IM is the up-front collateral required to open a position.

It covers potential losses during a short period of adverse price movement.

$$
IM_i = k_{IM} \times \sigma_{\text{asset}} \times \sqrt{T_{\text{horizon}}} \times |q_i| \times P_{\text{mark}}
$$

where `k_IM` is a risk multiplier tuned by the risk engine.

- When volatility (σ) or liquidity stress rises, IM increases incrementally.
- In calm regimes, IM relaxes automatically through the breaker feedback loop.

---

### Maintenance Margin (MM)

MM defines the minimum collateral required to keep a position open.

If `Equity < MM`, the account is flagged for margin call or liquidation.

Typical ratio:  `MM ≈ (0.5–0.8) × IM`.

MM acts as a safety buffer against intraday volatility and delayed liquidations.

When the breaker enters **SOFT** or **HARD** states, the ratio tightens dynamically.

---

### Settlement Flow

| Event | Action | Ledger Effect |
| --- | --- | --- |
| Trade open | Lock IM | Free → IM |
| Mark-to-market update | Adjust VM | PnL realized in USDC |
| PnL positive | Release VM to Free Margin | Increase available funds |
| PnL negative | Deduct VM from IM | Trigger margin check |
| Liquidation | Transfer loss → auction or insurance fund | Update CCP balance |
| Finalization | Anchor state to L1 | All balances provable on-chain |

All settlements occur per-batch.

### Leverage Definition

Leverage is implicit — not chosen by the user directly but computed from current margin conditions:

$$
Li = \frac{|\text{Position Notional}|}{\text{IM}_i}
$$

- **Dynamic Leverage:** automatically adjusts with volatility and systemic risk (index Rₜ from Section 12.4).
- **Asymmetry:** if the market is crowded long, allowed long leverage reduces; short leverage increases symmetrically.
- **Cap Enforcement:** when Rₜ ≥ θ₁ (SOFT mode), leverage caps tighten; when Rₜ ≥ θ₂ (HARD mode), new leverage = 0 (reduce-only).

This ensures systemic solvency without manual intervention.

### Outcome

> This mechanism completes the financial loop of the CCP:
> 
> 
> positions open, evolve, and settle entirely in-rollup, with deterministic cash flows and predictable leverage.
> 
> Margin and PnL logic become transparent and provable to all participants.
> 

## Collateral Architecture: aaveUSD Integration

In traditional exchanges, margin collateral just sits idle.

Here, we can improve capital efficiency by holding traders’ margin in a **yield-bearing stablecoin** — e.g. **aaveUSD**, a token that automatically earns Aave interest in the background.

This way, traders **still earn yield** on their posted margin while it remains locked inside the Cartesi CCP.

---

### How It Works

Each trader’s margin account holds a balance of **aaveUSD tokens**.

The CCP keeps all balances denominated in **USDC-equivalent value**, but the actual token stored is **aaveUSD**.

Let:

- $I_t$: yield index from Aave (`liquidityIndex`)
- $C_{i,t}$: number of aaveUSD tokens held by trader *i* at time *t*
- $Δr_t$: the yield accrued between batch *t–1* and *t*

Yield is computed incrementally as:

$Δr_t=\frac{I_t}{I_{t-1}} - 1$

Then each trader’s collateral updates as:

$C_{i,t}=C_{i,t−1}×(1+Δr_t)$

Their collateral **grows deterministically inside the rollup** every batch, even if they do nothing.

The updated collateral is then valued in USDC terms:

$V_{i,t} = P_{aaveUSD,t} × C_{i,t} ≈ C_{i,t}$

Since 1 aaveUSD ≈ 1 USDC, the accounting remains simple and consistent.

---

### What Happens During Trading

All trading PnL (profit and loss) is still measured in **USDC-equivalent terms**.

When a position gains or loses value, the system just updates the internal ledger — no token transfers yet.

**Example:**

- Trader A (long 10 ETH @ $4 000)
- Trader B (short 10 ETH @ $4 000)
- New mark price = $3 800

Then:

| Trader | PnL | Effect |
| --- | --- | --- |
| Long | − $2 000 | Margin decreases by $2 000 |
| Short | + $2 000 | Margin increases by $2 000 |

So the CCP updates:

```
VM_long  -= 2000
VM_short += 2000

```

No real tokens move yet — it’s just a **bookkeeping entry** inside the rollup’s deterministic state.

---

### What Happens When a Trader Withdraws

When the short trader (the winner) withdraws funds:

1. The CCP reduces their **Free Margin** by $2 000 (USDC-equivalent).
2. Converts that value into **aaveUSD tokens** at the current yield index:

$ΔC_{withdraw}=\frac{2000}{P_{aaveUSD,t}}≈2000$

1. Transfers ~ 2 000 aaveUSD from the CCP custody contract to the trader’s wallet.

Since aaveUSD continuously earns yield, the trader can later redeem it on Aave for slightly **more than 2 000 USDC**, depending on accrued interest.

---

### Haircuts and Safety

Because aaveUSD depends on Aave’s smart contracts, the CCP applies a **haircut** $h_{yield}$ to manage that extra risk:

$EligibleIM_{i,t}=(1−h_{yield})×V_{i,t}$

This simply means not all of the aaveUSD value counts toward margin — usually 98-99%.

If Aave is ever stressed or depegs slightly, the haircut protects solvency.

---

### Why This Matters

| Benefit | Explanation |
| --- | --- |
| **Earn Yield on Margin** | Traders keep earning interest on idle funds instead of losing opportunity cost. |
| **Same Accounting Simplicity** | Everything stays in USDC-equivalent units inside the CCP ledger. |
| **Automatic Updates** | Yield accrual (Δrₜ) is deterministic and computed each batch. |
| **Full Transparency** | All balances and interest updates are visible and verifiable inside the rollup. |
| **Safe Withdrawals** | Traders receive aaveUSD tokens directly; no hidden off-chain conversions. |

---

### What Could Go Wrong (and How We Mitigate It)

| Risk | Description | Mitigation |
| --- | --- | --- |
| **Smart-contract risk** | Aave or aaveUSD contract failure | Haircut on collateral value; whitelist only audited protocols |
| **De-peg risk** | aaveUSD ≠ 1 USDC | Automatic haircut increase under stress |
| **Protocol downtime** | Yield index update stalls | System pauses new deposits until resync |

---

### Summary

Even though traders’ collateral is held in **aaveUSD**, all accounting and risk management inside the CCP remain denominated in **USDC-equivalent value.**

Profit and loss are tracked in stable units, while yield accrues deterministically through Aave’s liquidity index.

When traders withdraw, they receive the equivalent amount of **aaveUSD**, representing their current margin plus accrued interest.

This design lets traders earn yield on idle margin while preserving the deterministic, on-chain nature of the CCP’s financial logic.

## Liquidation Auctions

Liquidation auctions are the **mechanism of last resort** in a risk-managed exchange.

They determine what happens when a trader’s position loses so much value that their collateral is no longer enough to cover potential losses.

In traditional markets, this happens instantly: the clearinghouse automatically offsets the position.

In DeFi, we need an **on-chain process** that lets solvent traders take over insolvent ones *without freezing or crashing the market.*

---

### The Simple Explanation

Think of a liquidation auction as a **fire sale for bad trades.**

When a trader can’t meet their margin:

1. The CCP seizes that position.
2. It announces: “We have 100 ETH of exposure to sell.”
3. Other traders bid to take over that exposure (at a price).
4. The CCP transfers the position to the winners, using the failed trader’s collateral to pay for the loss.

If bids cover the entire exposure, the system stays balanced.

If they don’t, the leftover loss goes down the **default waterfall**:

`Trader margin → Insurance fund → Default fund → DAO`.

In a healthy market, the auction clears almost instantly and no systemic loss occurs.

### Why Auctions Are Critical for a CCP

A Central Counterparty (CCP) guarantees that the market always remains solvent.

To do that, it must have a **deterministic way to close** losing positions without chaos or front-running.

In the Cartesi CCP:

- The **Circuit Breaker** switches to HARD mode during stress.
- All undercollateralized accounts move to a **Liquidation Queue** inside the rollup.
- The CCP batches those positions into a **net exposure** and opens an **internal auction** to reassign risk.

### Types of Liquidation Auctions

| Type | Description | Pros | Cons |
| --- | --- | --- | --- |
| **Dutch (descending price)** | Price drops until someone buys exposure | Fast | Can overshoot, slippage risk |
| **English (ascending)** | Bidders compete to pay more for collateral | Maximizes recovery | Slower |
| **Batch Clearing** | Collect bids, clear at equilibrium price | Fair, deterministic | More complex |
| **Keeper Race (legacy DeFi)** | External bots compete on gas | Simple | Chaotic, non-deterministic |
| **Internal CCP Auction** | CCP batches and clears exposure inside rollup | Frictionless, fast, provable | Needs active bidders |

### How It Works Step-by-Step

**Example:**

A long position worth 100 ETH becomes undercollateralized.

```markdown
1.  Risk engine flags position → margin call failed.
2.  CCP seizes the position → moves it into Liquidation Pool.
3.  Auction opens: “Selling 100 ETH exposure.”
4.  3-second bidding window inside the rollup.
5.  Solvent traders submit sealed bids.
6.  CCP finds clearing price where bids cover 100 ETH.
7.  Positions transfer to winners; CCP uses collateral to cover losses.
8.  Market returns to NORMAL state.

```

If bids only covered 95 ETH, the remaining 5 ETH shortfall goes to the insurance fund.

### Integration with Circuit Breakers

| Breaker State | Auction Behavior |
| --- | --- |
| **NORMAL** | Regular trading, no liquidations |
| **SOFT** | Margin ramps up; new open interest capped |
| **HARD** | Liquidation batch triggered, reduce-only mode |
| **RECOVERY** | Normal trading resumes post-auction |

Liquidation auctions are the **enforcement mechanism** behind the HARD breaker state.

### Activity and Liquidity Requirements

Liquidation auctions **require market activity** — someone must be there to buy when others are forced to sell.

If no one bids, the auction can’t clear, and losses propagate down the waterfall.

To ensure this loop works even during stress:

1. **Seed internal market makers:** deterministic bots quoting inside the rollup.
2. **Simulate liquidity:** replay crash scenarios before launch to test recovery.
3. **Reward participation:** incentives for quoting and auction bidding, not just trade volume.

**The safer the system, the more capital it needs to stay liquid.**

That’s the paradox every CCP faces — and exactly why deterministic liquidity modules will be critical in the MVP.

---

## Base-Layer Integration Roadmap

### Objective

Enable the Cartesi CCP to **query Ethereum base-layer state** (Uniswap, Chainlink, Aave, etc.) directly from inside the VM.

This allows risk models, mark-to-market, and settlement logic to operate on **real on-chain data** without relying on external oracles.

---

### Why It Matters

Base-layer access bridges the gap between simulation and reality.

### Priority Targets

1. **Uniswap TWAP via slot0 reads** (for mark price & LE model).
2. **Aave liquidity index** (for yield-linked perps and LSM hedging).
3. **Base-layer balance proofs** (for external LP verification).
4. **Block timestamp + gas metrics** (for latency-aware throttling).

---

### Expected Outcomes

- **Oracle independence:** direct, provable price access.
- **Auditable risk events:** anyone can replay block data.
- **Cross-protocol composability:** seamless integration with DeFi primitives.
- **Institutional credibility:** clear lineage from base-layer truth to CCP state.

---

### Long-Term Vision

> The Cartesi CCP becomes a self-contained clearinghouse that continuously prices, margins, and settles risk based on live Ethereum state — not oracle feeds.
> 
> 
> Each rollup batch = one step of deterministic, incremental market truth.
> 

| Subsystem | Before | After |
| --- | --- | --- |
| **Mark-to-Market** | Relies on external oracle | Reads UniswapV3 slot0 (√P) directly |
| **Aave yield** | Off-chain relayed index | Query Aave’s liquidityIndex natively |
| **LSM Auctions** | Synthetic pricing | Anchored to on-chain pool price |
| **Collateral verification** | Trust LP balance reports | Verify L1 balances directly |
|  |  |  |

## Discussion Points for the Team

---

### Market Experience

- How frequently are margin requirements updated — and how is this communicated to traders in real time?
- What will traders actually see when the system moves into SOFT or HARD mode? (e.g. banners, frozen inputs, visual cues.)
- How is the “reduce-only” mode handled in UI

### Pricing & PnL

- Can a trader verify the exact mark-to-market price that drove their PnL change?
- How visible should internal risk metrics (VaR, Δ_sys, breaker state) be in the interface?
- How do we ensure traders trust the composite mark price during thin liquidity moments?

### Capital & Fees

- How are trading and funding fees structured — per fill, per position, or per time interval?
- Can users predict required margin before opening a trade, given dynamic leverage?
- What withdrawal and settlement times are acceptable given Cartesi’s limitations?

---

### Core Architecture

- What’s the precise order of operations per block: `sequencer → VM → risk engine → margin update → settlement`?
- How frequently should the risk engine recompute VaR and margin buffers?
- What’s the performance target for simulating N traders and M instruments in the Cartesi Machine?

### Data Feeds & Oracles

- How do we retrieve Uniswap TWAPs or CEX mid-prices deterministically inside the rollup?
- What is the fallback mechanism if an external feed stalls or diverges significantly from internal price?

### Synthetic Liquidity Module

- How are SLM bots initialized, funded, and managed in code — as shared actors or discrete accounts?
- How do we enforce exposure caps and automatic PnL stop rules on them?
- How is their collateral updated and visible in rollup state for audits?

### Security & Testing

- What scenarios should our simulations cover? (e.g. 2021 ETH crash, oracle lag, extreme skew.)

---

### Capital & Risk

- How much seed collateral (USDC) is required to bootstrap the SLM and test auctions safely?
- Who provides and manages this pool — internal treasury, foundation, or DAO?
- What’s the maximum acceptable drawdown (in %) for internal liquidity funds before they must be topped up?

### Economics & Incentives

- How are fees and funding rates distributed among traders, liquidity providers, and CCP reserves?
- Should participants earn incentives for maintaining quotes or participating in auctions, not just trading volume?

## Supporting Evidence

The following references ground the proposal’s core mechanics — incremental risk computation, CCP architecture, and DeFi risk modeling — in established literature.

### Risk Metrics & Modelling

| Metric / Concept | Provenance | Supporting Literature |
| --- | --- | --- |
| **EWMA Volatility** | Incremental variance estimation for short-horizon risk | J.P. Morgan (1996) *RiskMetrics Technical Doc*; Poon & Granger (2003) |
| **GARCH Models** | Volatility clustering & persistence | Bollerslev (1986); Katsiampa (2017, crypto GARCH) |
| **Value-at-Risk (VaR)** | Core solvency / IM metric; backtesting frameworks | Kupiec (1995); Longin (2000) |
| **Systemic Δ / Γ Aggregation** | Portfolio-level exposure & convexity risk | Acharya & Richardson (2009) |
| **Liquidity Elasticity (LE)** | Depth-sensitive volatility / VaR correction | Amihud (2002); Kyle (1985) |
| **Circuit-Breaker FSM** | Deterministic stress-response (NORMAL→SOFT→HARD) | SEC Limit-Up/Limit-Down Report (2012) |
| **Incremental Risk Computation** | Rolling memory-bounded IM/MM updates | Basel FRTB (2016); CME SPAN 2 (2023); Eurex PRISMA (2022) |
|  |  |  |
| **Incremental Margin calculation** | Chebyshev Methods for Ultra-efficient Risk Calculations | Ardia et al. (2018) *arXiv:1805.00898* |

---

### Clearing & CCP Model Evidence

| Theme | Focus | Supporting Literature |
| --- | --- | --- |
| **Central Clearing Theory** | Why CCPs reduce counterparty risk via novation and netting | Pirrong (2011) *ISDA Discussion Paper* |
| **Systemic Liquidity Risk** | CCP liquidity demands under stress | King, M. (2023). *International Journal of Central Banking*. |
| **Default Waterfall Design** | Structured loss absorption logic | Allen (2012) *Stanford Law Review* |
| **Interconnected Risk** | Feedback loops between CCPs and banks | Faruqui et al. (2018) *BIS Quarterly Review* |
| **Regulatory Frameworks** | Basel III, FRTB risk capital and margin standards | Basel Committee on Banking Supervision (2016) |
| **Real-Time Margining** | Portfolio VaR margin systems in production | CME SPAN 2 (2023); Eurex PRISMA (2022) |

---

### DeFi & Crypto Risk Research

| Area | Insight | Supporting Literature |
| --- | --- | --- |
| **Crypto Volatility Models** | EWMA / GARCH remain robust under high volatility | Ječmínek et al. (2020); Katsiampa (2017) |
| **DeFi Risk Management** | Need for deterministic on-chain risk engines | Adamyk et al. (2025) *MDPI J. Risk & Fin.* |
| **On-Chain VaR Computation** | Real-time VaR for crypto-derivatives | Chen et al. (2023) *arXiv 2309.06393* |
| **Liquidity Shocks & Cascades** | How AMM depth and funding mechanics amplify volatility | Werner et al. (2022) *DeFi Liquidity Crisis Study* |
| **Keeper and Auction Dynamics** | Deterministic liquidations reduce chaos vs gas races | Gaunt et al. (2021) *DeFi Liquidation Mechanisms Paper* |