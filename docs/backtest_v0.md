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

Run with mode comparison (all_trades vs single_ladder in one pass):

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

Set `execution.max_concurrent_ladders` as a guardrail value:

- In `single_ladder`, it must be `1` (enforced).
- In `all_trades`, it can be >1 for research scenarios.

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

Category breakdown now uses explicit category fields when present, with text-based fallback inference from market/event metadata.
