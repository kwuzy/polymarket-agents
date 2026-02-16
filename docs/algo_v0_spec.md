# Trading Algo v0 Spec (Step 1)

## Objective
Design a strategy that:
1. Finds favorable Polymarket opportunities
2. Sizes using a controlled `2^n` progression
3. Hard-stops losses based on account capacity
4. Standardizes trade sizing across different spreads/prices using a single unit definition

---

## 1) Candidate Selection Engine

### 1.1 Universe filter (from existing repo data)
Use `Gamma` + `Polymarket` market metadata and orderbook data to keep only markets that are:
- active, not closed, not archived
- spread <= `max_spread_threshold`
- sufficient depth at target fill size
- event resolution date within allowed horizon
- not restricted category (if configured)

### 1.2 Signal stack
Build an estimated probability `p_hat` per outcome using weighted sources:

- **S_market**: market-implied probability from current price/mid
- **S_cross**: consistency checks against related markets (same event family)
- **S_news**: structured news evidence score
- **S_social**: Reddit / sentiment score
- **S_trader**: tracked trader positioning signal (if available)

Aggregate with configurable weights:

`p_hat = w_market*S_market + w_cross*S_cross + w_news*S_news + w_social*S_social + w_trader*S_trader`

### 1.3 Entry edge
For a YES buy at ask price `a`:
- edge = `p_hat - a`

For NO (equivalent YES price `a_yes`, NO price `a_no`):
- evaluate symmetric edge and take higher expected value side.

Only enter if:
- edge >= `min_edge`
- expected value after fees/slippage > 0
- confidence score >= `min_confidence`

Rank candidates by:
`score = edge * confidence * liquidity_factor * time_decay_factor`

---

## 2) Unit Definition (Standardization)

Define **1 Unit = 1 Risk Unit (RU)** where RU is max-loss-at-resolution in USDC.

This solves “$1 at different spreads isn’t equal”.

### 2.1 Risk per share
- BUY YES at price `p`: risk/share = `p`
- BUY NO at price `q`: risk/share = `q`

### 2.2 Convert RU to shares
If RU = `$R` and risk/share = `r_ps`:
- `shares = R / r_ps`

So each base trade risks the same worst-case dollars regardless of odds.

---

## 3) Controlled 2^n Sizing

Let base stake be 1 RU.
Progression index `k` starts at 0 for a new ladder.

Position risk for trade `k`:
- `risk_k = RU * 2^k`

Rules:
- if prior trade in ladder **wins** -> reset `k = 0`
- if prior trade in ladder **loses** -> `k = k + 1`

Do not increase ladder on:
- unresolved trade
- low-confidence signal
- regime change (volatility/news shock)

---

## 4) Capacity-Based Failsafe

Let:
- `B` = current liquid bankroll usable for strategy
- `alpha` = capital utilization cap (e.g., 0.4 means only 40% of B can be ladder-risk)
- `R` = RU in USDC

Ladder cumulative risk through step `k`:
- `cum_risk(k) = R * (2^(k+1) - 1)`

Hard max ladder step:
- `k_max = floor(log2((alpha*B)/R + 1)) - 1`

Stop-loss constraints (all active):
1. **Ladder cap**: never trade if next step would exceed `k_max`
2. **Drawdown cap**: pause strategy if equity drawdown > `dd_max`
3. **Daily loss cap**: stop new entries once daily realized PnL < `-daily_loss_limit`
4. **Open risk cap**: sum(open worst-case risk) <= `open_risk_cap * B`

When cap hit:
- freeze progression
- reset to base RU only after cooldown and/or recovery criteria

---

## 5) Repo Integration Plan

### Existing files to leverage
- `agents/polymarket/gamma.py` (market/event metadata)
- `agents/polymarket/polymarket.py` (orderbook/price/execution)
- `agents/connectors/news.py` (news ingestion)
- `agents/connectors/chroma.py` (RAG retrieval)

### New modules to add
- `agents/strategy/signals.py` – source signals + probability estimate
- `agents/strategy/risk.py` – RU conversion + 2^n + failsafes
- `agents/strategy/candidate_ranker.py` – edge scoring and top-N selection
- `agents/strategy/state_store.py` – ladder state, bankroll, outcomes, cooldowns

### Output schema per candidate
```json
{
  "market_id": "...",
  "side": "BUY_YES|BUY_NO",
  "price": 0.42,
  "p_hat": 0.51,
  "edge": 0.09,
  "confidence": 0.71,
  "risk_unit_usdc": 10,
  "risk_step_k": 1,
  "risk_amount_usdc": 20,
  "shares": 47.6,
  "pass_checks": true,
  "reasons": ["edge>threshold", "spread_ok", "liquidity_ok"]
}
```

---

## 6) Data Needed from Kevin to Start Step 2 (fake-data testing)

### Risk + bankroll
1. Starting bankroll for paper testing (USDC)
2. `alpha` capital utilization cap
3. Max drawdown (`dd_max`)
4. Daily stop loss
5. Base RU (or allow algo to derive RU from bankroll, e.g. 0.5% of B)

### Trade policy
6. Allowed categories (politics/sports/crypto/etc.)
7. Max holding time per trade
8. Max concurrent ladders/trades
9. Minimum liquidity/depth requirement
10. Maximum spread threshold

### Signal configuration
11. Initial source weights (`w_market`, `w_cross`, `w_news`, `w_social`, `w_trader`)
12. News sources preferred / API keys
13. Reddit subreddits to monitor
14. “Other trader” source definition (wallet list? leaderboard? copied accounts?)

### Execution realism for simulation
15. Slippage model assumptions
16. Fee assumptions
17. Fill model (instant, partial, delayed)
18. Resolution timing assumptions and mark-to-market rules

### Logging + monitoring
19. What metrics matter most (Sharpe, hit rate, max DD, expectancy, utilization)
20. Alert thresholds for monitoring

---

## 7) First Defaults (if unspecified)

- RU = 0.5% of bankroll
- alpha = 0.35
- min_edge = 0.04
- max spread = 0.03
- confidence >= 0.6
- max concurrent ladders = 3
- dd_max = 12%
- daily loss cap = 3%

These are conservative starter values and should be tuned in simulation first.
