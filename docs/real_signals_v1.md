# Real Signals v1 (News + Social, Point-in-Time)

This adds a point-in-time-safe feature builder for real external signals.

## Scope
- News signal (timestamped, category-aware, sentiment-aware)
- Social signal (Reddit/X style posts, timestamped, category-aware, sentiment-aware)
- Trade-level feature generation with strict `item.ts <= trade.ts` rule

## File
- `scripts/python/real_signals.py`

## Input formats
### News JSON rows
```json
{"ts":"2025-07-01T12:00:00Z","category":"sports","sentiment":0.3,"text":"...","source":"rss"}
```

### Social JSON rows
```json
{"ts":"2025-07-01T12:00:00Z","category":"sports","sentiment":0.2,"text":"...","source":"reddit"}
```

### Trade events JSON rows
```json
{"trade_id":"A1","market_id":"123","category":"sports","ts":"2025-07-01T13:00:00Z"}
```

## Usage
```bash
python3 scripts/python/real_signals.py \
  --news data/signals/news_events.json \
  --social data/signals/social_events.json \
  --trades data/signals/trade_events.json \
  --out outputs/backtests/real_signal_features.json
```

## Outputs
Per trade:
- windows: 1d/7d/30d
- news count + mean sentiment
- social count + mean sentiment

## Safety
No future leakage:
- feature windows are bounded to `[trade_ts - window, trade_ts]`
- future news/social items are ignored for earlier trades
