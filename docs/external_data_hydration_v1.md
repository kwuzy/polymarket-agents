# External Data Hydration v1

Hydrates aligned datasets for ablation runs in a shared date window.

## Command

```bash
python3 scripts/python/hydrate_external_data.py \
  --snapshot data/snapshots/gamma_markets_fixture.json \
  --start 2025-01-01T00:00:00Z \
  --end 2025-12-31T23:59:59Z \
  --trade-ts-field createdAt \
  --wallets-file data/whales/watchlist.json

# Optional: auto-populate watchlist from leaderboard
python3 scripts/python/hydrate_external_data.py \
  --start 2026-01-01T00:00:00Z \
  --end 2026-12-31T23:59:59Z \
  --trade-ts-field updatedAt \
  --auto-leaderboard-wallets \
  --leaderboard-limit 25
```

## Outputs
- `data/signals/trade_events.json`
- `data/signals/news_events.json`
- `data/signals/social_events.json`
- `data/whales/whale_activity.json`
- `outputs/backtests/hydration_summary.json`

## Notes
- News: RSS feeds (Reuters/BBC)
- Social: Reddit `/new.json`
- Whale activity: Polymarket data-api `/activity?user=<wallet>` for wallets in `watchlist.json`
- If wallet list is empty, whale dataset remains empty.
