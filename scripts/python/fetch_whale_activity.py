import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def http_get_json(url: str):
    req = Request(url, headers={"User-Agent": "polymarket-agents/whale-intel-v1"})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_user_activity(user: str, limit: int = 500):
    base = "https://data-api.polymarket.com/activity"
    q = urlencode({"user": user, "limit": str(limit)})
    url = f"{base}?{q}"
    payload = http_get_json(url)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    return []


def normalize_activity(user: str, rows):
    out = []
    for r in rows:
        ts = r.get("timestamp") or r.get("createdAt") or r.get("time")
        market_id = str(r.get("market") or r.get("marketId") or r.get("market_id") or "")
        category = str(r.get("category") or "other").lower()
        pnl = float(r.get("pnl") or r.get("profit") or 0.0)
        won = r.get("won")
        out.append(
            {
                "wallet": user.lower(),
                "market_id": market_id,
                "category": category,
                "pnl": pnl,
                "won": won,
                "ts": ts,
                "raw": r,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Fetch whale activity snapshots from Polymarket data-api")
    parser.add_argument("--users", required=True, help="Comma-separated wallet addresses")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", default="data/whales/whale_activity.json")
    args = parser.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]
    all_rows = []
    for u in users:
        rows = fetch_user_activity(u, limit=args.limit)
        all_rows.extend(normalize_activity(u, rows))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_rows, indent=2))
    print(f"wrote {args.out} rows={len(all_rows)} users={len(users)} at={datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
