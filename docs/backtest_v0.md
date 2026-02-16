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

## Risk model notes

- Bankroll starts at 1000 (configurable)
- RU defaults to `risk_unit_pct * bankroll`
- RU is frozen inside a ladder and recalculated on ladder reset
- Ladder uses `2^n` sizing after losses
- Capacity cap uses `alpha`
- Stops on max drawdown / daily loss cap
