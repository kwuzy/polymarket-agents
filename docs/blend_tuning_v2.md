# Blend Tuning v2

Using aligned 2026 updatedAt window with hydrated external datasets and freshness guardrails.

Selected weights from sweep:
- news_external_weight: 0.1
- reddit_external_weight: 0.1
- trader_external_weight: 0.5

Ablation confirmation (`ablation_20260225_025745.md`):
- baseline row: baseline_pnl -0.3105, baseline_win_rate 0.421
- plus_real_whales row: baseline_pnl -0.0216, baseline_win_rate 0.526
- plus_all row: baseline_pnl -0.0216, baseline_win_rate 0.526

Interpretation:
- Whale/trader signal is currently strongest contributor.
- Real news/social still add value in best-weights search, but high external weighting degrades baseline robustness.
