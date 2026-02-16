import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from urllib.parse import urlencode
from urllib.request import Request, urlopen


GAMMA_MARKETS_ENDPOINT = "https://gamma-api.polymarket.com/markets"


@dataclass
class TradeResult:
    market_id: str
    side: str
    price: float
    p_hat: float
    edge: float
    risk_usdc: float
    pnl: float
    won: bool


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def fetch_gamma_snapshot(path: str, limit: int = 500) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    params = {
        "active": "true",
        "closed": "false",
        "archived": "false",
        "limit": str(limit),
    }
    url = f"{GAMMA_MARKETS_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    with open(path, "w") as f:
        json.dump(data, f)


def load_snapshot(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def hash_to_unit(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    integer = int(digest[:12], 16)
    return (integer % 10_000_000) / 10_000_000


def parse_implied_prob(market: Dict) -> float:
    outcome_prices = market.get("outcomePrices")
    if isinstance(outcome_prices, str):
        outcome_prices = json.loads(outcome_prices)
    if not outcome_prices or len(outcome_prices) < 2:
        return None
    try:
        return float(outcome_prices[0])
    except Exception:
        return None


def parse_spread(market: Dict) -> float:
    try:
        return float(market.get("spread", 1.0))
    except Exception:
        return 1.0


def category_of(market: Dict) -> str:
    events = market.get("events") or []
    if events and isinstance(events, list):
        tags = events[0].get("tags") or []
        if tags:
            return (tags[0].get("label") or "unknown").lower()
    return "unknown"


def synth_signals(market: Dict, implied: float, seed: int) -> Dict[str, float]:
    mid = implied
    market_id = str(market.get("id", "0"))
    cat = category_of(market)

    u1 = hash_to_unit(f"{seed}:{market_id}:cross")
    u2 = hash_to_unit(f"{seed}:{market_id}:news")
    u3 = hash_to_unit(f"{seed}:{market_id}:reddit")
    u4 = hash_to_unit(f"{seed}:{market_id}:trader")
    uc = hash_to_unit(f"{seed}:{cat}:cat")

    return {
        "market": mid,
        "cross": clip(mid + (u1 - 0.5) * 0.16 + (uc - 0.5) * 0.05),
        "news": clip(mid + (u2 - 0.5) * 0.35),
        "reddit": clip(mid + (u3 - 0.5) * 0.45),
        "trader": clip(mid + (u4 - 0.5) * 0.25),
    }


def weighted_prob(signals: Dict[str, float], weights: Dict[str, float]) -> float:
    return clip(sum(signals[k] * weights[k] for k in weights.keys()))


def candidate_filter(market: Dict, implied: float, cfg: Dict) -> bool:
    if implied is None:
        return False
    if implied < cfg["filters"]["min_prob"] or implied > cfg["filters"]["max_prob"]:
        return False
    spread = parse_spread(market)
    if spread > cfg["filters"]["max_spread"]:
        return False
    return True


def generate_weight_variants(base: Dict[str, float], step: float = 0.05, max_variants: int = 200) -> List[Dict[str, float]]:
    keys = ["market", "cross", "news", "reddit", "trader"]
    values = [round(i * step, 4) for i in range(int(1 / step) + 1)]
    variants = [base]
    seen = {tuple(base[k] for k in keys)}

    for a in values:
        for b in values:
            for c in values:
                for d in values:
                    e = round(1 - (a + b + c + d), 4)
                    if e < 0 or e > 1:
                        continue
                    vec = (a, b, c, d, e)
                    if vec in seen:
                        continue
                    seen.add(vec)
                    variants.append(dict(zip(keys, vec)))
                    if len(variants) >= max_variants:
                        return variants
    return variants


def ladder_k_max(bankroll: float, ru: float, alpha: float) -> int:
    return math.floor(math.log2((alpha * bankroll) / ru + 1.0)) - 1


def run_backtest(snapshot: List[Dict], cfg: Dict, weights: Dict[str, float]) -> Dict:
    seed = cfg["seed"]
    risk_cfg = cfg["risk"]
    exec_cfg = cfg["execution"]

    bankroll = risk_cfg["starting_bankroll"]
    peak = bankroll
    daily_start = bankroll
    ru = bankroll * risk_cfg["risk_unit_pct"]
    alpha = risk_cfg["capital_utilization"]
    k = 0

    results: List[TradeResult] = []
    by_category = {}

    ordered = sorted(snapshot, key=lambda m: int(m.get("id", 0)))

    for market in ordered:
        implied = parse_implied_prob(market)
        if not candidate_filter(market, implied, cfg):
            continue

        k_max = ladder_k_max(bankroll, ru, alpha)
        if k > k_max:
            break

        risk_amount = ru * (2**k)
        if risk_amount > alpha * bankroll:
            break

        signals = synth_signals(market, implied, seed)
        p_hat = weighted_prob(signals, weights)

        if p_hat == implied:
            continue

        side = "BUY_YES" if p_hat > implied else "BUY_NO"
        price = implied if side == "BUY_YES" else 1 - implied
        edge = abs(p_hat - implied)
        if price <= 0:
            continue

        # Synthetic outcome generation (deterministic from hash)
        latent = clip(0.6 * p_hat + 0.4 * implied)
        u = hash_to_unit(f"{seed}:{market.get('id')}:outcome")
        yes_wins = u < latent
        won = yes_wins if side == "BUY_YES" else (not yes_wins)

        shares = risk_amount / price
        gross = shares * ((1 - price) if won else -price)

        fee = abs(gross) * (exec_cfg.get("fee_bps", 0) / 10_000)
        slippage = abs(gross) * (exec_cfg.get("slippage_bps", 0) / 10_000)
        pnl = gross - fee - slippage

        bankroll += pnl
        peak = max(peak, bankroll)

        if won:
            k = 0
            if not risk_cfg.get("freeze_ru_within_ladder", True):
                ru = bankroll * risk_cfg["risk_unit_pct"]
        else:
            k += 1

        # On reset conditions, recalc RU
        if won and risk_cfg.get("freeze_ru_within_ladder", True):
            ru = bankroll * risk_cfg["risk_unit_pct"]

        dd = (peak - bankroll) / peak if peak > 0 else 0
        day_loss = (daily_start - bankroll) / daily_start if daily_start > 0 else 0
        if dd >= risk_cfg["max_drawdown"] or day_loss >= risk_cfg["daily_loss_cap"]:
            break

        cat = category_of(market)
        by_category.setdefault(cat, {"trades": 0, "wins": 0, "pnl": 0.0})
        by_category[cat]["trades"] += 1
        by_category[cat]["wins"] += int(won)
        by_category[cat]["pnl"] += pnl

        results.append(
            TradeResult(
                market_id=str(market.get("id")),
                side=side,
                price=price,
                p_hat=p_hat,
                edge=edge,
                risk_usdc=risk_amount,
                pnl=pnl,
                won=won,
            )
        )

    wins = sum(1 for r in results if r.won)
    total = len(results)
    total_pnl = sum(r.pnl for r in results)
    max_dd = (peak - bankroll) / peak if peak > 0 else 0

    return {
        "weights": weights,
        "trades": total,
        "wins": wins,
        "win_rate": (wins / total) if total else 0,
        "pnl": total_pnl,
        "ending_bankroll": bankroll,
        "max_drawdown": max_dd,
        "by_category": by_category,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic Polymarket backtest runner")
    parser.add_argument("--config", default="configs/backtest_v0.json")
    parser.add_argument("--refresh-snapshot", action="store_true")
    parser.add_argument("--outdir", default="outputs/backtests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    snapshot_path = cfg["snapshot_path"]

    if args.refresh_snapshot or not os.path.exists(snapshot_path):
        fetch_gamma_snapshot(snapshot_path)

    snapshot = load_snapshot(snapshot_path)

    variants = [cfg["weights"]]
    if cfg.get("weight_sweep", {}).get("enabled", False):
        variants = generate_weight_variants(
            cfg["weights"],
            step=cfg["weight_sweep"].get("step", 0.05),
            max_variants=cfg["weight_sweep"].get("max_variants", 200),
        )

    leaderboard = []
    for w in variants:
        report = run_backtest(snapshot, cfg, w)
        leaderboard.append(report)

    leaderboard.sort(key=lambda x: (x["pnl"], x["win_rate"]), reverse=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_path = outdir / f"backtest_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(
            {
                "timestamp_utc": ts,
                "config": cfg,
                "top10": leaderboard[:10],
                "baseline": next((x for x in leaderboard if x["weights"] == cfg["weights"]), None),
                "all_results": leaderboard,
            },
            f,
            indent=2,
        )

    print(f"Wrote report: {full_path}")
    if leaderboard:
        best = leaderboard[0]
        print("Best:", best["weights"], "pnl=", round(best["pnl"], 4), "win_rate=", round(best["win_rate"], 4), "trades=", best["trades"])


if __name__ == "__main__":
    main()
