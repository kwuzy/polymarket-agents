# Whale Intelligence v1 (Point-in-Time Safe)

This module prepares whale/trader features without time leakage.

## Goals
- Track what top wallets are trading.
- Evaluate which categories each wallet performs well in.
- Compute wallet quality over date windows (7d / 30d / 90d).
- Guarantee features for trade time `t` only use whale data with timestamp `<= t`.

## Files
- `scripts/python/whale_intel.py`
- `tests/test_whale_intel_pit.py`

## Input schema (whale trades)
JSON array of rows:

```json
[
  {
    "wallet": "0xabc...",
    "market_id": "12345",
    "category": "sports",
    "pnl": 12.5,
    "won": true,
    "ts": "2025-06-01T12:00:00Z"
  }
]
```

## Usage
Build point-in-time whale features for backtest trade events:

```bash
python3 scripts/python/whale_intel.py \
  --whale-trades data/whales/whale_activity.json \
  --trade-events data/whales/trade_events.json \
  --out outputs/backtests/whale_features.json \
  --top-n 20
```

## Point-in-time safety rules
- Wallet stats are computed from `window_start <= whale_trade.ts <= trade.ts`.
- No future-trade contributions are allowed.
- Tests assert that future whale trades do not influence earlier trade features.

## Notes
This v1 module is the analytics/feature layer. Upstream ingestion from Polymarket leaderboard and activity endpoints is staged next.
