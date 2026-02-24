# Blend Tuning v1

Quick sweep on aligned config:
- `configs/backtest_v0_timestamped_updatedAt_2026.json`
- hydrated datasets (trade/news/social/whales)

Best quick setting found (grid 0.2/0.3/0.4):
- `news_external_weight`: 0.2
- `reddit_external_weight`: 0.4
- `trader_external_weight`: 0.4

Observed uplift in ablation baseline row:
- before (+real): baseline_pnl ~ -0.3390, baseline_win_rate 0.421
- after (+real tuned): baseline_pnl ~ -0.1790, baseline_win_rate 0.474

Artifacts:
- `outputs/backtests/blend_sweep_quick_20260224_193115.md`
- `outputs/backtests/ablation_20260224_193140.md`
