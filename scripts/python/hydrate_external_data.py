import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


def parse_iso(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    v = str(value).strip()
    if not v:
        return None
    if v.isdigit():
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except Exception:
            return None
    if v.endswith('Z'):
        v = v[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(v)
    except Exception:
        return None


def to_iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def infer_category(text: str) -> str:
    t = (text or '').lower()
    if any(k in t for k in ['nba','nfl','mlb','nhl','soccer','football','basketball','final','playoff']):
        return 'sports'
    if any(k in t for k in ['trump','election','senate','house','president','congress','democrat','republican']):
        return 'politics'
    if any(k in t for k in ['bitcoin','btc','ethereum','eth','crypto','token']):
        return 'crypto'
    if any(k in t for k in ['stock','nasdaq','dow','s&p','earnings','revenue','inflation','gdp']):
        return 'business'
    return 'other'


def simple_sentiment(text: str) -> float:
    t = (text or '').lower()
    pos = sum(1 for k in ['win','beat','rise','up','surge','bull','growth','strong'] if k in t)
    neg = sum(1 for k in ['lose','drop','down','fall','bear','weak','risk','miss'] if k in t)
    raw = (pos - neg) / max(1, (pos + neg))
    return max(-1.0, min(1.0, raw))


def fetch_json(url: str):
    req = Request(url, headers={'User-Agent': 'polymarket-agents/hydrator'})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def fetch_text(url: str):
    req = Request(url, headers={'User-Agent': 'polymarket-agents/hydrator'})
    with urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', errors='ignore')


def load_snapshot(path):
    return json.loads(Path(path).read_text())


def build_trade_events(snapshot, start=None, end=None, ts_field='createdAt'):
    rows = []
    for m in snapshot:
        ts = parse_iso(m.get(ts_field) or m.get('createdAt') or m.get('startDate') or m.get('updatedAt'))
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        txt = ' '.join([str(m.get('question','')), str(m.get('description','')), str(m.get('slug',''))])
        rows.append({
            'trade_id': f"m{m.get('id')}",
            'market_id': str(m.get('id')),
            'category': infer_category(txt),
            'ts': to_iso(ts),
            'text': txt,
        })
    return rows


def fetch_rss_news(start=None, end=None):
    feeds = [
        'https://feeds.reuters.com/reuters/worldNews',
        'https://feeds.reuters.com/reuters/businessNews',
        'https://feeds.reuters.com/reuters/technologyNews',
        'https://feeds.bbci.co.uk/news/world/rss.xml',
    ]
    out = []
    for url in feeds:
        try:
            xml = fetch_text(url)
            root = ET.fromstring(xml)
            for item in root.findall('.//item')[:150]:
                title = (item.findtext('title') or '').strip()
                desc = (item.findtext('description') or '').strip()
                pub = item.findtext('pubDate') or item.findtext('published')
                ts = None
                if pub:
                    try:
                        from email.utils import parsedate_to_datetime
                        ts = parsedate_to_datetime(pub)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except Exception:
                        ts = None
                if ts is None:
                    continue
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                text = (title + ' ' + desc).strip()
                out.append({
                    'ts': to_iso(ts),
                    'text': text,
                    'category': infer_category(text),
                    'sentiment': simple_sentiment(text),
                    'source': 'rss',
                })
        except Exception:
            continue
    return out


def fetch_reddit_social(start=None, end=None):
    subs = ['politics', 'worldnews', 'CryptoCurrency', 'nba', 'sports', 'economics']
    out = []
    for s in subs:
        try:
            url = f"https://www.reddit.com/r/{s}/new.json?limit=100"
            payload = fetch_json(url)
            children = (((payload or {}).get('data') or {}).get('children') or [])
            for ch in children:
                d = ch.get('data') or {}
                ts = datetime.fromtimestamp(float(d.get('created_utc', 0)), tz=timezone.utc)
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                text = (d.get('title') or '') + ' ' + (d.get('selftext') or '')
                out.append({
                    'ts': to_iso(ts),
                    'text': text.strip(),
                    'category': infer_category(text),
                    'sentiment': simple_sentiment(text),
                    'source': 'reddit',
                })
        except Exception:
            continue
    return out


def fetch_leaderboard_wallets(limit=25):
    try:
        q = urlencode({'limit': str(limit), 'offset': '0'})
        url = f"https://data-api.polymarket.com/v1/leaderboard?{q}"
        rows = fetch_json(url)
        out = []
        for r in rows if isinstance(rows, list) else []:
            w = str(r.get('proxyWallet') or '').lower().strip()
            if w:
                out.append(w)
        return out
    except Exception:
        return []


def fetch_whale_activity(wallets, start=None, end=None):
    out = []
    for w in wallets:
        try:
            q = urlencode({'user': w, 'limit': '500'})
            url = f"https://data-api.polymarket.com/activity?{q}"
            rows = fetch_json(url)
            if isinstance(rows, dict):
                rows = rows.get('data', [])
            for r in rows or []:
                ts = parse_iso(r.get('timestamp') or r.get('createdAt') or r.get('time'))
                if ts is None:
                    continue
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                text = json.dumps(r)[:400]
                out.append({
                    'wallet': w.lower(),
                    'market_id': str(r.get('market') or r.get('marketId') or r.get('market_id') or ''),
                    'category': infer_category(text),
                    'pnl': float(r.get('pnl') or r.get('profit') or 0.0),
                    'won': r.get('won'),
                    'ts': to_iso(ts),
                })
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description='Hydrate external datasets for aligned ablation runs')
    ap.add_argument('--snapshot', default='data/snapshots/gamma_markets_fixture.json')
    ap.add_argument('--start', default='2025-01-01T00:00:00Z')
    ap.add_argument('--end', default='2025-12-31T23:59:59Z')
    ap.add_argument('--wallets-file', default='data/whales/watchlist.json')
    ap.add_argument('--trade-ts-field', default='createdAt', choices=['createdAt','updatedAt','startDate','endDate'])
    ap.add_argument('--auto-leaderboard-wallets', action='store_true')
    ap.add_argument('--leaderboard-limit', type=int, default=25)
    args = ap.parse_args()

    start = parse_iso(args.start)
    end = parse_iso(args.end)

    snapshot = load_snapshot(args.snapshot)
    trade_events = build_trade_events(snapshot, start=start, end=end, ts_field=args.trade_ts_field)
    news = fetch_rss_news(start=start, end=end)
    social = fetch_reddit_social(start=start, end=end)

    wallets = []
    wpath = Path(args.wallets_file)
    if wpath.exists():
        wallets = json.loads(wpath.read_text())
        if isinstance(wallets, dict):
            wallets = wallets.get('wallets', [])

    if (not wallets) and args.auto_leaderboard_wallets:
        wallets = fetch_leaderboard_wallets(limit=args.leaderboard_limit)
        wpath.parent.mkdir(parents=True, exist_ok=True)
        wpath.write_text(json.dumps({'wallets': wallets}, indent=2))

    whale_rows = fetch_whale_activity([str(x) for x in wallets], start=start, end=end) if wallets else []

    Path('data/signals').mkdir(parents=True, exist_ok=True)
    Path('data/whales').mkdir(parents=True, exist_ok=True)
    Path('data/signals/trade_events.json').write_text(json.dumps(trade_events, indent=2))
    Path('data/signals/news_events.json').write_text(json.dumps(news, indent=2))
    Path('data/signals/social_events.json').write_text(json.dumps(social, indent=2))
    Path('data/whales/whale_activity.json').write_text(json.dumps(whale_rows, indent=2))

    summary = {
        'trade_ts_field': args.trade_ts_field,
        'trade_events': len(trade_events),
        'news_events': len(news),
        'social_events': len(social),
        'whale_events': len(whale_rows),
        'wallets': len(wallets),
        'start': args.start,
        'end': args.end,
    }
    Path('outputs/backtests/hydration_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
