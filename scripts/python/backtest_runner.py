import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    return max(0, math.floor(math.log2((alpha * bankroll) / ru + 1.0)) - 1)


def compute_outcome(market: Dict, implied: float, p_hat: float, side: str, seed: int) -> bool:
    latent = clip(0.6 * p_hat + 0.4 * implied)
    u = hash_to_unit(f"{seed}:{market.get('id')}:outcome")
    yes_wins = u < latent
    return yes_wins if side == "BUY_YES" else (not yes_wins)


def enrich_category_summary(by_category: Dict[str, Dict]) -> Dict[str, Dict]:
    out = {}
    for cat, stats in by_category.items():
        trades = stats["trades"]
        wins = stats["wins"]
        out[cat] = {
            **stats,
            "win_rate": (wins / trades) if trades else 0.0,
            "avg_pnl": (stats["pnl"] / trades) if trades else 0.0,
        }
    return out


def validate_execution_config(exec_cfg: Dict) -> None:
    mode = exec_cfg.get("mode", "single_ladder")
    if mode not in {"single_ladder", "all_trades"}:
        raise ValueError("execution.mode must be one of: single_ladder, all_trades")

    max_ladders = int(exec_cfg.get("max_concurrent_ladders", 1))
    if max_ladders < 1:
        raise ValueError("execution.max_concurrent_ladders must be >= 1")

    # Live-mode guardrail: one active ladder only.
    if mode == "single_ladder" and max_ladders != 1:
        raise ValueError("single_ladder mode requires execution.max_concurrent_ladders == 1")


def run_backtest(snapshot: List[Dict], cfg: Dict, weights: Dict[str, float]) -> Dict:
    seed = cfg["seed"]
    risk_cfg = cfg["risk"]
    exec_cfg = cfg["execution"]

    validate_execution_config(exec_cfg)
    mode = exec_cfg.get("mode", "single_ladder")

    bankroll = risk_cfg["starting_bankroll"]
    peak = bankroll
    daily_start = bankroll
    ru = bankroll * risk_cfg["risk_unit_pct"]
    alpha = risk_cfg["capital_utilization"]

    ladder_step = 0
    current_loss_streak = 0
    max_loss_streak = 0
    max_ladder_depth = 0
    stop_reason = "completed"

    results: List[TradeResult] = []
    by_category = {}

    ordered = sorted(snapshot, key=lambda m: int(m.get("id", 0)))

    for market in ordered:
        implied = parse_implied_prob(market)
        if not candidate_filter(market, implied, cfg):
            continue

        # In all_trades mode each market is an independent step-0 trade.
        step_for_trade = ladder_step if mode == "single_ladder" else 0

        k_max = ladder_k_max(bankroll, ru, alpha)
        if step_for_trade > k_max:
            stop_reason = "capacity_limit"
            break

        risk_amount = ru * (2**step_for_trade)
        if risk_amount > alpha * bankroll:
            stop_reason = "risk_cap"
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

        won = compute_outcome(market, implied, p_hat, side, seed)

        shares = risk_amount / price
        gross = shares * ((1 - price) if won else -price)

        fee = abs(gross) * (exec_cfg.get("fee_bps", 0) / 10_000)
        slippage = abs(gross) * (exec_cfg.get("slippage_bps", 0) / 10_000)
        pnl = gross - fee - slippage

        bankroll += pnl
        peak = max(peak, bankroll)

        if won:
            current_loss_streak = 0
            if mode == "single_ladder":
                ladder_step = 0
                if risk_cfg.get("freeze_ru_within_ladder", True):
                    ru = bankroll * risk_cfg["risk_unit_pct"]
        else:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            if mode == "single_ladder":
                ladder_step += 1
                max_ladder_depth = max(max_ladder_depth, ladder_step)

        dd = (peak - bankroll) / peak if peak > 0 else 0
        day_loss = (daily_start - bankroll) / daily_start if daily_start > 0 else 0
        if dd >= risk_cfg["max_drawdown"]:
            stop_reason = "max_drawdown"
            break
        if day_loss >= risk_cfg["daily_loss_cap"]:
            stop_reason = "daily_loss_cap"
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
        "execution_mode": mode,
        "max_concurrent_ladders": int(exec_cfg.get("max_concurrent_ladders", 1)),
        "trades": total,
        "wins": wins,
        "win_rate": (wins / total) if total else 0,
        "pnl": total_pnl,
        "ending_bankroll": bankroll,
        "max_drawdown": max_dd,
        "max_loss_streak": max_loss_streak,
        "max_ladder_depth": max_ladder_depth,
        "stop_reason": stop_reason,
        "by_category": enrich_category_summary(by_category),
    }


def write_leaderboard_csv(path: Path, leaderboard: List[Dict]) -> None:
    keys = ["market", "cross", "news", "reddit", "trader"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "execution_mode",
                "max_concurrent_ladders",
                "trades",
                "wins",
                "win_rate",
                "pnl",
                "ending_bankroll",
                "max_drawdown",
                "max_loss_streak",
                "max_ladder_depth",
                "stop_reason",
                *[f"w_{k}" for k in keys],
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(leaderboard, start=1):
            out = {
                "rank": idx,
                "execution_mode": row["execution_mode"],
                "max_concurrent_ladders": row.get("max_concurrent_ladders", 1),
                "trades": row["trades"],
                "wins": row["wins"],
                "win_rate": row["win_rate"],
                "pnl": row["pnl"],
                "ending_bankroll": row["ending_bankroll"],
                "max_drawdown": row["max_drawdown"],
                "max_loss_streak": row["max_loss_streak"],
                "max_ladder_depth": row["max_ladder_depth"],
                "stop_reason": row["stop_reason"],
            }
            for k in keys:
                out[f"w_{k}"] = row["weights"].get(k, 0.0)
            writer.writerow(out)


def write_category_csv(path: Path, leaderboard: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "category", "trades", "wins", "win_rate", "pnl", "avg_pnl"],
        )
        writer.writeheader()
        for idx, row in enumerate(leaderboard, start=1):
            for cat, stats in sorted(row.get("by_category", {}).items()):
                writer.writerow(
                    {
                        "rank": idx,
                        "category": cat,
                        "trades": stats["trades"],
                        "wins": stats["wins"],
                        "win_rate": stats["win_rate"],
                        "pnl": stats["pnl"],
                        "avg_pnl": stats["avg_pnl"],
                    }
                )


def run_mode_pass(snapshot: List[Dict], cfg: Dict, force_mode: str = None) -> List[Dict]:
    cfg_local = json.loads(json.dumps(cfg))
    if force_mode is not None:
        exec_cfg = cfg_local.setdefault("execution", {})
        exec_cfg["mode"] = force_mode
        if force_mode == "single_ladder":
            exec_cfg["max_concurrent_ladders"] = 1

    variants = [cfg_local["weights"]]
    if cfg_local.get("weight_sweep", {}).get("enabled", False):
        variants = generate_weight_variants(
            cfg_local["weights"],
            step=cfg_local["weight_sweep"].get("step", 0.05),
            max_variants=cfg_local["weight_sweep"].get("max_variants", 200),
        )

    leaderboard = []
    for w in variants:
        report = run_backtest(snapshot, cfg_local, w)
        leaderboard.append(report)

    leaderboard.sort(key=lambda x: (x["pnl"], x["win_rate"]), reverse=True)
    return leaderboard


def write_mode_comparison_csv(path: Path, by_mode: Dict[str, Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "best_pnl",
                "best_win_rate",
                "best_trades",
                "baseline_pnl",
                "baseline_win_rate",
                "baseline_trades",
                "best_stop_reason",
                "baseline_stop_reason",
            ],
        )
        writer.writeheader()
        for mode, payload in by_mode.items():
            best = payload.get("best") or {}
            baseline = payload.get("baseline") or {}
            writer.writerow(
                {
                    "mode": mode,
                    "best_pnl": best.get("pnl", 0.0),
                    "best_win_rate": best.get("win_rate", 0.0),
                    "best_trades": best.get("trades", 0),
                    "baseline_pnl": baseline.get("pnl", 0.0),
                    "baseline_win_rate": baseline.get("win_rate", 0.0),
                    "baseline_trades": baseline.get("trades", 0),
                    "best_stop_reason": best.get("stop_reason", ""),
                    "baseline_stop_reason": baseline.get("stop_reason", ""),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Deterministic Polymarket backtest runner")
    parser.add_argument("--config", default="configs/backtest_v0.json")
    parser.add_argument("--refresh-snapshot", action="store_true")
    parser.add_argument("--outdir", default="outputs/backtests")
    parser.add_argument("--compare-modes", action="store_true", help="Run both all_trades and single_ladder passes and write comparison output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    snapshot_path = cfg["snapshot_path"]

    if args.refresh_snapshot or not os.path.exists(snapshot_path):
        fetch_gamma_snapshot(snapshot_path)

    snapshot = load_snapshot(snapshot_path)

    leaderboard = run_mode_pass(snapshot, cfg)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comparison_payload = None
    comparison_csv = None

    full_path = outdir / f"backtest_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(
            {
                "timestamp_utc": ts,
                "config": cfg,
                "top10": leaderboard[:10],
                "baseline": next((x for x in leaderboard if x["weights"] == cfg["weights"]), None),
                "all_results": leaderboard,
                "mode_comparison": comparison_payload,
            },
            f,
            indent=2,
        )

    leaderboard_csv = outdir / f"backtest_{ts}_leaderboard.csv"
    category_csv = outdir / f"backtest_{ts}_categories.csv"
    write_leaderboard_csv(leaderboard_csv, leaderboard)
    write_category_csv(category_csv, leaderboard)

    comparison_payload = None
    comparison_csv = None
    if args.compare_modes:
        mode_payload = {}
        for mode in ["all_trades", "single_ladder"]:
            mode_board = run_mode_pass(snapshot, cfg, force_mode=mode)
            mode_payload[mode] = {
                "best": mode_board[0] if mode_board else None,
                "baseline": next((x for x in mode_board if x["weights"] == cfg["weights"]), None),
            }
        comparison_payload = mode_payload
        comparison_csv = outdir / f"backtest_{ts}_mode_compare.csv"
        write_mode_comparison_csv(comparison_csv, mode_payload)

    print(f"Wrote report: {full_path}")
    print(f"Wrote leaderboard CSV: {leaderboard_csv}")
    print(f"Wrote category CSV: {category_csv}")
    if comparison_csv:
        print(f"Wrote mode comparison CSV: {comparison_csv}")
    if leaderboard:
        best = leaderboard[0]
        print(
            "Best:",
            best["weights"],
            "pnl=",
            round(best["pnl"], 4),
            "win_rate=",
            round(best["win_rate"], 4),
            "trades=",
            best["trades"],
            "mode=",
            best["execution_mode"],
        )


if __name__ == "__main__":
    main()
