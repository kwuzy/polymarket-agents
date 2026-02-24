# Ablation v1

Run feature-stack ablations on timestamped backtests.

## Command

```bash
python3 scripts/python/ablation_runner.py --config configs/backtest_v0_timestamped_2025.json
```

## Scenarios
- `baseline`
- `plus_real`
- `plus_real_whales`
- `plus_all` (real + whales + category models)

## Notes
Ablation effects depend on non-empty external datasets:
- `data/signals/news_events.json`
- `data/signals/social_events.json`
- `data/whales/whale_activity.json`

If these are missing/empty, scenario results may be identical.
