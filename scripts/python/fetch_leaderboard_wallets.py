import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch_json(url: str):
    req = Request(url, headers={"User-Agent": "polymarket-agents/leaderboard-wallets"})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Fetch top wallets from Polymarket leaderboard")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="data/whales/watchlist.json")
    args = ap.parse_args()

    q = urlencode({"limit": str(args.limit), "offset": str(args.offset)})
    url = f"https://data-api.polymarket.com/v1/leaderboard?{q}"
    rows = fetch_json(url)
    wallets = []
    for r in rows if isinstance(rows, list) else []:
        w = str(r.get("proxyWallet") or "").lower().strip()
        if w:
            wallets.append(w)

    payload = {"wallets": wallets}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} wallets={len(wallets)}")


if __name__ == "__main__":
    main()
