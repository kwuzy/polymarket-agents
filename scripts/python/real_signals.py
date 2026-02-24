import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def parse_iso_ts(value: str) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except Exception:
        return None


@dataclass
class NewsItem:
    ts: datetime
    text: str
    category: str
    source: str
    sentiment: float


@dataclass
class SocialItem:
    ts: datetime
    text: str
    category: str
    source: str
    sentiment: float


def _norm_sentiment(v) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, x))


def load_news(path: str) -> List[NewsItem]:
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        ts = parse_iso_ts(r.get("ts") or r.get("published_at") or r.get("createdAt"))
        if ts is None:
            continue
        out.append(
            NewsItem(
                ts=ts,
                text=str(r.get("text") or r.get("title") or ""),
                category=str(r.get("category") or "other").lower(),
                source=str(r.get("source") or "news"),
                sentiment=_norm_sentiment(r.get("sentiment", 0.0)),
            )
        )
    out.sort(key=lambda x: x.ts)
    return out


def load_social(path: str) -> List[SocialItem]:
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        ts = parse_iso_ts(r.get("ts") or r.get("created_at") or r.get("createdAt"))
        if ts is None:
            continue
        out.append(
            SocialItem(
                ts=ts,
                text=str(r.get("text") or ""),
                category=str(r.get("category") or "other").lower(),
                source=str(r.get("source") or "social"),
                sentiment=_norm_sentiment(r.get("sentiment", 0.0)),
            )
        )
    out.sort(key=lambda x: x.ts)
    return out


def _window(items, as_of: datetime, days: int, category: str):
    start = as_of - timedelta(days=days)
    return [x for x in items if x.ts <= as_of and x.ts >= start and (category == "all" or x.category == category)]


def aggregate_real_signals(as_of: datetime, category: str, news: List[NewsItem], social: List[SocialItem], windows=(1, 7, 30)) -> Dict:
    out = {"as_of": as_of.isoformat(), "category": category, "news": {}, "social": {}}
    for d in windows:
        n = _window(news, as_of, d, category)
        s = _window(social, as_of, d, category)
        out["news"][f"{d}d"] = {
            "count": len(n),
            "sentiment_mean": (sum(x.sentiment for x in n) / len(n)) if n else 0.0,
        }
        out["social"][f"{d}d"] = {
            "count": len(s),
            "sentiment_mean": (sum(x.sentiment for x in s) / len(s)) if s else 0.0,
        }
    return out


def build_trade_features(trades: List[Dict], news: List[NewsItem], social: List[SocialItem], windows=(1, 7, 30)) -> List[Dict]:
    out = []
    for t in trades:
        ts = parse_iso_ts(t.get("ts"))
        if ts is None:
            continue
        cat = str(t.get("category") or "other").lower()
        agg = aggregate_real_signals(ts, cat, news, social, windows=windows)
        out.append({
            "trade_id": t.get("trade_id"),
            "market_id": t.get("market_id"),
            "ts": ts.isoformat(),
            "category": cat,
            "signals": agg,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Build point-in-time real news/social signals for trade events")
    parser.add_argument("--news", required=True)
    parser.add_argument("--social", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    news = load_news(args.news)
    social = load_social(args.social)
    trades = json.loads(Path(args.trades).read_text())
    feats = build_trade_features(trades, news, social)
    Path(args.out).write_text(json.dumps(feats, indent=2))
    print(f"wrote {args.out} rows={len(feats)}")


if __name__ == "__main__":
    main()
