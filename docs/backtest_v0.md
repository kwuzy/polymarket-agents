# Deterministic Backtest v0

This backtest runner is config-driven and deterministic:

- Fixed config file
- Fixed snapshot file
- Fixed seed
- Same input => same output

## Files

- `configs/backtest_v0.json` - strategy + risk + weight sweep settings
- `scripts/python/backtest_runner.py` - deterministic runner
- `data/snapshots/gamma_markets_sample.json` - snapshot input (created by runner)
- `outputs/backtests/*.json` - run outputs

## Run

```bash
python3 scripts/python/backtest_runner.py --config configs/backtest_v0.json
```

Refresh the snapshot explicitly:

```bash
python3 scripts/python/backtest_runner.py --config configs/backtest_v0.json --refresh-snapshot
```

Run with mode comparison (`all_trades`, `single_ladder`, `multi_line` in one pass):

```bash
python3 scripts/python/backtest_runner.py --config configs/backtest_v0.json --compare-modes
```

## Current baseline in config

Weights are ordered as:

1. market
2. cross
3. news
4. reddit
5. trader

Baseline:

```json
{
  "market": 0.45,
  "cross": 0.3,
  "news": 0.1,
  "reddit": 0.05,
  "trader": 0.1
}
```

The runner includes this baseline in the experiment output and runs a weight sweep for alternatives.

## Execution modes

Set `execution.mode` in config:

- `single_ladder` (live-like): one active ladder, 2^n progression on losses, reset on win.
- `all_trades` (research): every candidate is simulated as independent step-0 trade to maximize sample size.
- `multi_line` (research, line-aware): deterministic routing across `execution.num_lines` independent ladder lines.

Set execution guardrails:

- `execution.max_concurrent_ladders`
  - In `single_ladder`, it must be `1` (enforced).
- `execution.num_lines`
  - Used by `multi_line` mode (e.g., 10 means 10 independent lines).
- `execution.line_assignment`
  - `round_robin` (default): candidate i goes to line `i % N`.
  - `market_hash`: deterministic market-id hashing to a stable line.

## Risk model notes

- Bankroll starts at 1000 (configurable)
- RU defaults to `risk_unit_pct * bankroll`
- RU is frozen inside a ladder and recalculated on ladder reset
- Ladder uses `2^n` sizing after losses (`single_ladder` mode)
- Capacity cap uses `alpha`
- Stops on max drawdown / daily loss cap

## Output diagnostics

Each run now reports:

- `stop_reason` (completed, capacity_limit, risk_cap, max_drawdown, daily_loss_cap)
- `max_loss_streak`
- `max_ladder_depth`
- `by_category` with trades/wins/pnl + `win_rate` + `avg_pnl`

And writes companion reporting files:

- `outputs/backtests/backtest_<timestamp>_leaderboard.csv`
- `outputs/backtests/backtest_<timestamp>_categories.csv`
- `outputs/backtests/backtest_<timestamp>_mode_compare.csv` (when `--compare-modes` is used)
- `outputs/backtests/backtest_<timestamp>_summary.md`
- `outputs/backtests/backtest_<timestamp>_live_recommendation.md`
- `outputs/backtests/backtest_<timestamp>_live_profile.json`

Category breakdown now uses explicit category fields when present, with text-based fallback inference from market/event metadata.
A guardrail-aware live profile recommendation is produced from `single_ladder` results.

Validation split is configurable via `validation.holdout_fraction` (default 0.25), and holdout robustness is emitted to `backtest_<timestamp>_holdout.md`.


Walk-forward config fields:
- `validation.walk_forward.enabled`
- `validation.walk_forward.train_size`
- `validation.walk_forward.test_size`
- `validation.walk_forward.step_size`

Each window trains on `[start:start+train_size)` and evaluates on the immediately following test slice, then advances by `step_size`.


Blocked fold robustness is configurable with `validation.folds` and exported to:
- `backtest_<timestamp>_fold_robustness.csv`
- `backtest_<timestamp>_fold_robustness.md`
- `backtest_<timestamp>_multiline.md` (per-line trades/pnl/avg risk + concentration metric)

Cost sensitivity scenarios are configurable under `validation.cost_scenarios` and exported to:
- `backtest_<timestamp>_cost_scenarios.csv`
- `backtest_<timestamp>_cost_scenarios.md`

Data quality diagnostics are exported to:
- `backtest_<timestamp>_data_quality.md`

Walk-forward robustness is configurable under `validation.walk_forward` and exported to:
- `backtest_<timestamp>_walk_forward.csv`
- `backtest_<timestamp>_walk_forward.md`

This includes candidate/executed counts, skip reasons, execution rate, and a warning when holdout executes zero baseline trades.


### Slippage/liquidity stress model

Execution slippage supports an optional dynamic model under `execution.slippage_model`:

- `enabled`: turn dynamic slippage on/off
- `spread_weight`: increases slippage as spread widens
- `depth_reference_usdc`: liquidity/volume reference for depth penalty
- `impact_reference_usdc`: trade-size reference for impact scaling
- `impact_power`: non-linear size impact exponent
- `max_slippage_bps`: cap on effective per-trade slippage

When enabled, effective slippage bps are derived from base `execution.slippage_bps` and scaled by spread, depth proxy, and trade size.
Run diagnostics now include `avg_effective_slippage_bps`.


`backtest_<timestamp>_live_profile.json` emits a machine-readable live configuration payload derived from the guardrail-aware single-ladder recommendation (weights, risk/execution settings, and selection metadata).

Feature source audit outputs are emitted to:
- `backtest_<timestamp>_feature_sources.md`
- `backtest_<timestamp>_feature_sources.json`

Configure feature provenance in `features.sources` to track live vs synthetic signals and weight share realism in each run.

Live readiness gate outputs are emitted to:
- `backtest_<timestamp>_live_readiness.md`
- `backtest_<timestamp>_live_readiness.json`

The readiness score combines baseline quality, holdout/walk-forward robustness, and feature-source realism (live vs synthetic share).

Deployment decision artifacts are emitted to:
- `backtest_<timestamp>_decision_card.md`
- `backtest_<timestamp>_decision_card.json`

These summarize GO/PAPER/NO-GO posture from readiness + robustness signals for supervised rollout decisions.

Guardrail alert artifacts are emitted to:
- `backtest_<timestamp>_guardrail_alerts.md`
- `backtest_<timestamp>_guardrail_alerts.json`

Configure thresholds in `validation.guardrail_alerts` (`max_drawdown`, `min_trades`, `min_win_rate`, `min_readiness_score`).

Run registry outputs:
- `backtest_<timestamp>_manifest.json` (artifact map + decision signals)
- `backtest_run_index.csv` (append-only index across runs)

Trend outputs (derived from run index):
- `backtest_run_trends.md`
- `backtest_run_trends.json`

These summarize recent run health (avg readiness, decision mix, and guardrail block rate).
