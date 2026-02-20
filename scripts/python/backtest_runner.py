import argparse
import csv
import hashlib
import json
import math
import os
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


def parse_volume_num(market: Dict) -> float:
    for key in ["volumeNum", "volume", "volume24hr", "liquidityNum", "liquidity"]:
        raw = market.get(key)
        try:
            val = float(raw)
            if val >= 0:
                return val
        except Exception:
            continue
    return 0.0


def parse_liquidity_num(market: Dict) -> float:
    for key in ["liquidityNum", "liquidity", "volumeNum", "volume"]:
        raw = market.get(key)
        try:
            val = float(raw)
            if val >= 0:
                return val
        except Exception:
            continue
    return 0.0


def compute_effective_slippage_bps(market: Dict, risk_amount: float, exec_cfg: Dict) -> float:
    base = float(exec_cfg.get("slippage_bps", 0) or 0)
    model = exec_cfg.get("slippage_model", {}) or {}
    if not model.get("enabled", False):
        return base

    spread = parse_spread(market)
    volume = parse_volume_num(market)
    liquidity = parse_liquidity_num(market)

    spread_mult = max(0.25, min(3.0, 1.0 + spread * float(model.get("spread_weight", 0.5))))

    depth_ref = float(model.get("depth_reference_usdc", 50000.0) or 50000.0)
    depth_proxy = max(1.0, volume + liquidity)
    depth_mult = max(0.5, min(3.0, depth_ref / (depth_proxy + depth_ref)))

    impact_ref = float(model.get("impact_reference_usdc", 50.0) or 50.0)
    impact_power = float(model.get("impact_power", 0.5) or 0.5)
    impact_mult = max(0.5, min(3.0, (max(risk_amount, 0.01) / impact_ref) ** impact_power))

    eff = base * spread_mult * depth_mult * impact_mult
    cap = float(model.get("max_slippage_bps", max(10.0, base * 4.0)) or max(10.0, base * 4.0))
    return max(0.0, min(cap, eff))


def infer_category_from_text(text: str) -> str:
    t = (text or "").lower()
    keyword_map = {
        "politics": ["trump", "election", "senate", "house", "president", "white house", "congress", "democrat", "republican", "governor", "fed", "powell"],
        "sports": ["nba", "nfl", "mlb", "nhl", "ufc", "soccer", "football", "super bowl", "championship", "final", "playoff", "world cup", "f1", "nascar", "tennis", "golf"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "token", "binance", "coinbase", "blockchain"],
        "business": ["stock", "nasdaq", "s&p", "dow", "tesla", "apple", "microsoft", "earnings", "ipo", "revenue", "inflation", "gdp", "rate cut"],
        "entertainment": ["movie", "oscar", "grammy", "emmy", "album", "box office", "netflix", "youtube", "tiktok", "celebrity"],
        "science": ["space", "nasa", "spacex", "climate", "earthquake", "hurricane", "weather", "ai", "openai", "research"],
    }
    for cat, words in keyword_map.items():
        if any(w in t for w in words):
            return cat
    return "other"


def category_of(market: Dict) -> str:
    # Prefer explicit category when available.
    for key in ["category", "group", "topic"]:
        raw = market.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()

    events = market.get("events") or []
    if events and isinstance(events, list):
        ev0 = events[0]
        for key in ["category", "group", "topic"]:
            raw = ev0.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip().lower()

        # Fallback to text inference from event fields.
        text = " ".join(
            [
                str(ev0.get("title", "")),
                str(ev0.get("slug", "")),
                str(ev0.get("ticker", "")),
                str(ev0.get("description", "")),
            ]
        )
        inferred = infer_category_from_text(text)
        if inferred != "other":
            return inferred

    # Final fallback to market text inference.
    market_text = " ".join(
        [
            str(market.get("question", "")),
            str(market.get("slug", "")),
            str(market.get("description", "")),
        ]
    )
    return infer_category_from_text(market_text)


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


def signal_source_map(cfg: Dict) -> Dict[str, str]:
    fmap = (((cfg or {}).get("features") or {}).get("sources") or {})
    default = {
        "market": "live_market_price",
        "cross": "synthetic_proxy",
        "news": "synthetic_proxy",
        "reddit": "synthetic_proxy",
        "trader": "synthetic_proxy",
    }
    default.update({k: str(v) for k, v in fmap.items() if k in default})
    return default


def write_feature_sources_md(path: Path, source_map: Dict[str, str], weights: Dict[str, float]) -> None:
    lines = ["# Feature Source Audit", ""]
    synthetic_weight = 0.0
    live_weight = 0.0
    for k in ["market", "cross", "news", "reddit", "trader"]:
        src = source_map.get(k, "unknown")
        w = float(weights.get(k, 0.0) or 0.0)
        lines.append(f"- {k}: source={src}, weight={w:.3f}")
        if src.startswith("live"):
            live_weight += w
        else:
            synthetic_weight += w
    lines.append("")
    lines.append(f"- live_weight_share={live_weight:.3f}")
    lines.append(f"- synthetic_weight_share={synthetic_weight:.3f}")
    if synthetic_weight > 0.0:
        lines.append("- ⚠️ Synthetic features present: treat this as framework validation, not production alpha.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_feature_sources_json(path: Path, source_map: Dict[str, str], weights: Dict[str, float]) -> None:
    live_weight = 0.0
    synthetic_weight = 0.0
    rows = []
    for k in ["market", "cross", "news", "reddit", "trader"]:
        src = source_map.get(k, "unknown")
        w = float(weights.get(k, 0.0) or 0.0)
        rows.append({"feature": k, "source": src, "weight": w})
        if src.startswith("live"):
            live_weight += w
        else:
            synthetic_weight += w
    payload = {
        "features": rows,
        "totals": {
            "live_weight_share": live_weight,
            "synthetic_weight_share": synthetic_weight,
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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
    if mode not in {"single_ladder", "all_trades", "multi_line"}:
        raise ValueError("execution.mode must be one of: single_ladder, all_trades, multi_line")

    max_ladders = int(exec_cfg.get("max_concurrent_ladders", 1))
    if max_ladders < 1:
        raise ValueError("execution.max_concurrent_ladders must be >= 1")

    # Live-mode guardrail: one active ladder only.
    if mode == "single_ladder" and max_ladders != 1:
        raise ValueError("single_ladder mode requires execution.max_concurrent_ladders == 1")

    num_lines = int(exec_cfg.get("num_lines", 1))
    if num_lines < 1:
        raise ValueError("execution.num_lines must be >= 1")

    assignment = exec_cfg.get("line_assignment", "round_robin")
    if assignment not in {"round_robin", "market_hash"}:
        raise ValueError("execution.line_assignment must be one of: round_robin, market_hash")


def run_backtest(snapshot: List[Dict], cfg: Dict, weights: Dict[str, float]) -> Dict:
    seed = cfg["seed"]
    risk_cfg = cfg["risk"]
    exec_cfg = cfg["execution"]

    validate_execution_config(exec_cfg)
    mode = exec_cfg.get("mode", "single_ladder")
    num_lines = int(exec_cfg.get("num_lines", 1))

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

    # multi_line mode: independent line states and bankroll buckets
    line_bankrolls = []
    line_rus = []
    line_steps = []
    line_peaks = []
    line_loss_streaks = []
    line_trade_counts = []
    line_pnls = []
    line_risk_sums = []
    line_assignment = exec_cfg.get("line_assignment", "round_robin")

    if mode == "multi_line":
        per_line = bankroll / num_lines
        line_bankrolls = [per_line for _ in range(num_lines)]
        line_rus = [per_line * risk_cfg["risk_unit_pct"] for _ in range(num_lines)]
        line_steps = [0 for _ in range(num_lines)]
        line_peaks = [per_line for _ in range(num_lines)]
        line_loss_streaks = [0 for _ in range(num_lines)]
        line_trade_counts = [0 for _ in range(num_lines)]
        line_pnls = [0.0 for _ in range(num_lines)]
        line_risk_sums = [0.0 for _ in range(num_lines)]

    results: List[TradeResult] = []
    effective_slippage_bps_sum = 0.0
    by_category = {}
    diagnostics = {
        "total_markets": 0,
        "candidate_markets": 0,
        "skipped_filter": 0,
        "skipped_no_edge": 0,
        "skipped_invalid_price": 0,
        "avg_effective_slippage_bps": 0.0,
    }

    ordered = sorted(snapshot, key=lambda m: int(m.get("id", 0)))

    for i, market in enumerate(ordered):
        diagnostics["total_markets"] += 1
        implied = parse_implied_prob(market)
        if not candidate_filter(market, implied, cfg):
            diagnostics["skipped_filter"] += 1
            continue
        diagnostics["candidate_markets"] += 1

        if mode == "single_ladder":
            active_bankroll = bankroll
            active_ru = ru
            step_for_trade = ladder_step
        elif mode == "all_trades":
            active_bankroll = bankroll
            active_ru = ru
            step_for_trade = 0
        else:  # multi_line
            if line_assignment == "market_hash":
                line_idx = int(hash_to_unit(f"line:{market.get('id')}") * num_lines) % num_lines
            else:
                line_idx = i % num_lines
            active_bankroll = line_bankrolls[line_idx]
            active_ru = line_rus[line_idx]
            step_for_trade = line_steps[line_idx]

        k_max = ladder_k_max(active_bankroll, active_ru, alpha)
        if step_for_trade > k_max:
            if mode == "multi_line":
                continue
            stop_reason = "capacity_limit"
            break

        risk_amount = active_ru * (2**step_for_trade)
        if risk_amount > alpha * active_bankroll:
            if mode == "multi_line":
                continue
            stop_reason = "risk_cap"
            break

        signals = synth_signals(market, implied, seed)
        p_hat = weighted_prob(signals, weights)

        if p_hat == implied:
            diagnostics["skipped_no_edge"] += 1
            continue

        side = "BUY_YES" if p_hat > implied else "BUY_NO"
        price = implied if side == "BUY_YES" else 1 - implied
        edge = abs(p_hat - implied)
        if price <= 0:
            diagnostics["skipped_invalid_price"] += 1
            continue

        won = compute_outcome(market, implied, p_hat, side, seed)

        shares = risk_amount / price
        gross = shares * ((1 - price) if won else -price)

        fee = abs(gross) * (exec_cfg.get("fee_bps", 0) / 10_000)
        eff_slippage_bps = compute_effective_slippage_bps(market, risk_amount, exec_cfg)
        slippage = abs(gross) * (eff_slippage_bps / 10_000)
        pnl = gross - fee - slippage

        effective_slippage_bps_sum += eff_slippage_bps

        if mode == "multi_line":
            line_trade_counts[line_idx] += 1
            line_pnls[line_idx] += pnl
            line_risk_sums[line_idx] += risk_amount
            line_bankrolls[line_idx] += pnl
            line_peaks[line_idx] = max(line_peaks[line_idx], line_bankrolls[line_idx])
            if won:
                line_loss_streaks[line_idx] = 0
                line_steps[line_idx] = 0
                if risk_cfg.get("freeze_ru_within_ladder", True):
                    line_rus[line_idx] = line_bankrolls[line_idx] * risk_cfg["risk_unit_pct"]
            else:
                line_loss_streaks[line_idx] += 1
                max_loss_streak = max(max_loss_streak, line_loss_streaks[line_idx])
                line_steps[line_idx] += 1
                max_ladder_depth = max(max_ladder_depth, line_steps[line_idx])
            bankroll = sum(line_bankrolls)
            peak = max(peak, bankroll)
        else:
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

    out = {
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
        "diagnostics": {
            **diagnostics,
            "executed_trades": total,
            "execution_rate": (total / diagnostics["candidate_markets"]) if diagnostics["candidate_markets"] else 0.0,
            "avg_effective_slippage_bps": (effective_slippage_bps_sum / total) if total else 0.0,
        },
    }
    if mode == "multi_line":
        out["line_summary"] = {
            "num_lines": num_lines,
            "line_assignment": line_assignment,
            "line_bankrolls": line_bankrolls,
            "line_steps": line_steps,
            "line_trade_counts": line_trade_counts,
            "line_pnls": line_pnls,
            "line_risk_sums": line_risk_sums,
        }
    return out


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


def split_snapshot(snapshot: List[Dict], holdout_fraction: float) -> Tuple[List[Dict], List[Dict]]:
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        return snapshot, []
    cut = max(1, int(len(snapshot) * (1 - holdout_fraction)))
    return snapshot[:cut], snapshot[cut:]


def build_blocked_folds(snapshot: List[Dict], folds: int) -> List[Tuple[List[Dict], List[Dict]]]:
    if folds <= 1 or len(snapshot) < 4:
        return []
    n = len(snapshot)
    fold_size = max(1, n // folds)
    out = []
    for i in range(folds):
        start = i * fold_size
        end = n if i == folds - 1 else min(n, (i + 1) * fold_size)
        test = snapshot[start:end]
        train = snapshot[:start] + snapshot[end:]
        if train and test:
            out.append((train, test))
    return out


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
                "best_num_lines",
                "baseline_num_lines",
                "best_line_assignment",
                "baseline_line_assignment",
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
                    "best_num_lines": ((best.get("line_summary") or {}).get("num_lines", 1)),
                    "baseline_num_lines": ((baseline.get("line_summary") or {}).get("num_lines", 1)),
                    "best_line_assignment": ((best.get("line_summary") or {}).get("line_assignment", "")),
                    "baseline_line_assignment": ((baseline.get("line_summary") or {}).get("line_assignment", "")),
                }
            )


def _top_category_rows(by_category: Dict[str, Dict], top_n: int = 8) -> List[str]:
    rows = sorted(
        by_category.items(),
        key=lambda kv: (kv[1].get("pnl", 0.0), kv[1].get("trades", 0)),
        reverse=True,
    )[:top_n]
    out = []
    for cat, stats in rows:
        out.append(
            f"- {cat}: trades={stats.get('trades',0)}, win_rate={stats.get('win_rate',0):.3f}, pnl={stats.get('pnl',0.0):.4f}, avg_pnl={stats.get('avg_pnl',0.0):.4f}"
        )
    return out


def evaluate_holdout(
    train_results: List[Dict],
    test_snapshot: List[Dict],
    cfg: Dict,
    mode: str = "single_ladder",
) -> Dict:
    if not train_results or not test_snapshot:
        return None

    best_train = train_results[0]
    baseline_train = next((x for x in train_results if x["weights"] == cfg["weights"]), None)

    cfg_local = json.loads(json.dumps(cfg))
    cfg_local.setdefault("execution", {})["mode"] = mode
    if mode == "single_ladder":
        cfg_local["execution"]["max_concurrent_ladders"] = 1

    best_test = run_backtest(test_snapshot, cfg_local, best_train["weights"])
    baseline_test = run_backtest(test_snapshot, cfg_local, cfg["weights"])

    return {
        "mode": mode,
        "train_best": best_train,
        "train_baseline": baseline_train,
        "test_best": best_test,
        "test_baseline": baseline_test,
    }


def evaluate_cost_scenarios(
    snapshot: List[Dict],
    cfg: Dict,
    baseline_weights: Dict,
    recommended_weights: Dict,
    scenarios: List[Dict],
    mode: str = "single_ladder",
) -> List[Dict]:
    if not scenarios:
        return []

    out = []
    for i, s in enumerate(scenarios):
        fee_bps = float(s.get("fee_bps", 0))
        slippage_bps = float(s.get("slippage_bps", 0))
        label = s.get("label") or f"scenario_{i+1}"

        cfg_local = json.loads(json.dumps(cfg))
        exec_cfg = cfg_local.setdefault("execution", {})
        exec_cfg["mode"] = mode
        exec_cfg["fee_bps"] = fee_bps
        exec_cfg["slippage_bps"] = slippage_bps
        if mode == "single_ladder":
            exec_cfg["max_concurrent_ladders"] = 1

        baseline = run_backtest(snapshot, cfg_local, baseline_weights)
        reco = run_backtest(snapshot, cfg_local, recommended_weights) if recommended_weights else None

        out.append(
            {
                "label": label,
                "mode": mode,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "baseline": baseline,
                "recommended": reco,
            }
        )
    return out


def write_cost_scenarios_csv(path: Path, scenarios: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "mode",
                "fee_bps",
                "slippage_bps",
                "baseline_pnl",
                "baseline_win_rate",
                "recommended_pnl",
                "recommended_win_rate",
                "delta_pnl",
            ],
        )
        writer.writeheader()
        for s in scenarios:
            b = s.get("baseline") or {}
            r = s.get("recommended") or {}
            writer.writerow(
                {
                    "label": s.get("label", ""),
                    "mode": s.get("mode", ""),
                    "fee_bps": s.get("fee_bps", 0),
                    "slippage_bps": s.get("slippage_bps", 0),
                    "baseline_pnl": b.get("pnl", 0.0),
                    "baseline_win_rate": b.get("win_rate", 0.0),
                    "recommended_pnl": r.get("pnl", 0.0),
                    "recommended_win_rate": r.get("win_rate", 0.0),
                    "delta_pnl": (r.get("pnl", 0.0) - b.get("pnl", 0.0)) if r else 0.0,
                }
            )


def write_cost_scenarios_md(path: Path, scenarios: List[Dict]) -> None:
    lines = ["# Cost Sensitivity", ""]
    if not scenarios:
        lines.append("No cost scenarios configured.")
    else:
        for s in scenarios:
            b = s.get("baseline") or {}
            r = s.get("recommended") or {}
            delta = (r.get("pnl", 0.0) - b.get("pnl", 0.0)) if r else 0.0
            lines.append(
                f"- {s.get('label')}: fee={s.get('fee_bps')}bps, slip={s.get('slippage_bps')}bps | baseline_pnl={b.get('pnl',0.0):.4f}, recommended_pnl={r.get('pnl',0.0):.4f}, delta={delta:.4f}"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_multiline_md(path: Path, baseline_multi: Dict) -> None:
    lines = ["# Multi-line Diagnostics", ""]
    if not baseline_multi:
        lines.append("No multi-line baseline result available.")
    else:
        ls = baseline_multi.get("line_summary") or {}
        lines.append(f"- num_lines={ls.get('num_lines', 0)}")
        lines.append(f"- assignment={ls.get('line_assignment', 'round_robin')}")
        lines.append(f"- baseline_pnl={baseline_multi.get('pnl',0.0):.4f}")
        lines.append(f"- baseline_win_rate={baseline_multi.get('win_rate',0.0):.3f}")
        lines.append(f"- baseline_trades={baseline_multi.get('trades',0)}")
        banks = ls.get("line_bankrolls", [])
        steps = ls.get("line_steps", [])
        counts = ls.get("line_trade_counts", [])
        pnls = ls.get("line_pnls", [])
        risks = ls.get("line_risk_sums", [])
        if banks:
            lines.append("")
            lines.append("## Per-line stats")
            total_trades = sum(counts) if counts else 0
            if total_trades > 0:
                hhi = sum(((c / total_trades) ** 2) for c in counts)
                lines.append(f"- line_trade_concentration_hhi={hhi:.4f} (lower is more balanced)")
            for i, b in enumerate(banks, start=1):
                st = steps[i-1] if i-1 < len(steps) else 0
                c = counts[i-1] if i-1 < len(counts) else 0
                p = pnls[i-1] if i-1 < len(pnls) else 0.0
                r = risks[i-1] if i-1 < len(risks) else 0.0
                avg_r = (r / c) if c else 0.0
                lines.append(f"- line_{i}: trades={c}, pnl={p:.4f}, avg_risk={avg_r:.4f}, bankroll={b:.4f}, ladder_step={st}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_data_quality_md(path: Path, dataset_meta: Dict, baseline: Dict, holdout: Dict) -> None:
    lines = ["# Data Quality Diagnostics", ""]
    lines.append(
        f"- Dataset split: total={dataset_meta.get('total_markets',0)}, train={dataset_meta.get('train_markets',0)}, holdout={dataset_meta.get('holdout_markets',0)}"
    )
    lines.append("")

    bdiag = (baseline or {}).get("diagnostics") or {}
    lines.append("## Train baseline diagnostics")
    lines.append(f"- candidate_markets={bdiag.get('candidate_markets',0)}")
    lines.append(f"- executed_trades={bdiag.get('executed_trades',0)}")
    lines.append(f"- execution_rate={bdiag.get('execution_rate',0.0):.3f}")
    lines.append(f"- skipped_filter={bdiag.get('skipped_filter',0)}")
    lines.append(f"- skipped_no_edge={bdiag.get('skipped_no_edge',0)}")
    lines.append("")

    if holdout:
        hb = holdout.get("test_baseline") or {}
        hdiag = hb.get("diagnostics") or {}
        lines.append("## Holdout baseline diagnostics")
        lines.append(f"- candidate_markets={hdiag.get('candidate_markets',0)}")
        lines.append(f"- executed_trades={hdiag.get('executed_trades',0)}")
        lines.append(f"- execution_rate={hdiag.get('execution_rate',0.0):.3f}")
        lines.append(f"- skipped_filter={hdiag.get('skipped_filter',0)}")
        lines.append(f"- skipped_no_edge={hdiag.get('skipped_no_edge',0)}")

        if hdiag.get("executed_trades", 0) == 0:
            lines.append("")
            lines.append("⚠️ Warning: holdout baseline produced zero trades; treat robustness conclusions as low confidence.")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_fold_robustness(snapshot: List[Dict], cfg: Dict, folds: int, mode: str = "single_ladder") -> List[Dict]:
    fold_pairs = build_blocked_folds(snapshot, folds)
    out = []
    for idx, (train, test) in enumerate(fold_pairs, start=1):
        board = run_mode_pass(train, cfg, force_mode=mode)
        if not board:
            continue
        best_train = board[0]
        baseline_train = next((x for x in board if x["weights"] == cfg["weights"]), None)

        cfg_local = json.loads(json.dumps(cfg))
        cfg_local.setdefault("execution", {})["mode"] = mode
        if mode == "single_ladder":
            cfg_local["execution"]["max_concurrent_ladders"] = 1

        best_test = run_backtest(test, cfg_local, best_train["weights"])
        baseline_test = run_backtest(test, cfg_local, cfg["weights"])

        out.append(
            {
                "fold": idx,
                "train_size": len(train),
                "test_size": len(test),
                "train_best": best_train,
                "train_baseline": baseline_train,
                "test_best": best_test,
                "test_baseline": baseline_test,
            }
        )
    return out


def write_fold_robustness_csv(path: Path, folds: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "train_size",
                "test_size",
                "test_best_pnl",
                "test_best_win_rate",
                "test_best_trades",
                "test_baseline_pnl",
                "test_baseline_win_rate",
                "test_baseline_trades",
                "delta_pnl",
            ],
        )
        writer.writeheader()
        for r in folds:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            writer.writerow(
                {
                    "fold": r.get("fold"),
                    "train_size": r.get("train_size"),
                    "test_size": r.get("test_size"),
                    "test_best_pnl": tb.get("pnl", 0.0),
                    "test_best_win_rate": tb.get("win_rate", 0.0),
                    "test_best_trades": tb.get("trades", 0),
                    "test_baseline_pnl": bl.get("pnl", 0.0),
                    "test_baseline_win_rate": bl.get("win_rate", 0.0),
                    "test_baseline_trades": bl.get("trades", 0),
                    "delta_pnl": tb.get("pnl", 0.0) - bl.get("pnl", 0.0),
                }
            )


def write_fold_robustness_md(path: Path, folds: List[Dict]) -> None:
    lines = ["# Fold Robustness", ""]
    if not folds:
        lines.append("Fold validation disabled or insufficient data.")
    else:
        deltas = []
        for r in folds:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            delta = tb.get("pnl", 0.0) - bl.get("pnl", 0.0)
            deltas.append(delta)
            lines.append(
                f"- fold {r.get('fold')}: test_best_pnl={tb.get('pnl',0.0):.4f}, test_baseline_pnl={bl.get('pnl',0.0):.4f}, delta={delta:.4f}, best_trades={tb.get('trades',0)}, baseline_trades={bl.get('trades',0)}"
            )
        lines.append("")
        pos = sum(1 for d in deltas if d > 0)
        lines.append(f"- Positive delta folds: {pos}/{len(deltas)}")
        lines.append(f"- Avg delta pnl: {sum(deltas)/len(deltas):.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")




def build_walk_forward_windows(snapshot: List[Dict], train_size: int, test_size: int, step_size: int) -> List[Tuple[List[Dict], List[Dict]]]:
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        return []
    if len(snapshot) < (train_size + test_size):
        return []

    out = []
    start = 0
    n = len(snapshot)
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n:
            break
        train = snapshot[start:train_end]
        test = snapshot[train_end:test_end]
        if train and test:
            out.append((train, test))
        start += step_size
    return out


def evaluate_walk_forward(snapshot: List[Dict], cfg: Dict, mode: str = "single_ladder") -> List[Dict]:
    wf_cfg = (cfg.get("validation", {}) or {}).get("walk_forward", {}) or {}
    if not wf_cfg.get("enabled", False):
        return []

    train_size = int(wf_cfg.get("train_size", 0) or 0)
    test_size = int(wf_cfg.get("test_size", 0) or 0)
    step_size = int(wf_cfg.get("step_size", test_size or 0) or 0)

    windows = build_walk_forward_windows(snapshot, train_size, test_size, step_size)
    if not windows:
        return []

    out = []
    for i, (train, test) in enumerate(windows, start=1):
        board = run_mode_pass(train, cfg, force_mode=mode)
        if not board:
            continue

        best_train = board[0]
        baseline_train = next((x for x in board if x["weights"] == cfg["weights"]), None)

        cfg_local = json.loads(json.dumps(cfg))
        cfg_local.setdefault("execution", {})["mode"] = mode
        if mode == "single_ladder":
            cfg_local["execution"]["max_concurrent_ladders"] = 1

        best_test = run_backtest(test, cfg_local, best_train["weights"])
        baseline_test = run_backtest(test, cfg_local, cfg["weights"])

        out.append(
            {
                "window": i,
                "train_start": (i - 1) * step_size,
                "train_end": (i - 1) * step_size + len(train),
                "test_end": (i - 1) * step_size + len(train) + len(test),
                "train_size": len(train),
                "test_size": len(test),
                "train_best": best_train,
                "train_baseline": baseline_train,
                "test_best": best_test,
                "test_baseline": baseline_test,
            }
        )
    return out


def write_walk_forward_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window",
                "train_start",
                "train_end",
                "test_end",
                "train_size",
                "test_size",
                "test_best_pnl",
                "test_best_win_rate",
                "test_best_trades",
                "test_baseline_pnl",
                "test_baseline_win_rate",
                "test_baseline_trades",
                "delta_pnl",
            ],
        )
        writer.writeheader()
        for r in rows:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            writer.writerow(
                {
                    "window": r.get("window"),
                    "train_start": r.get("train_start"),
                    "train_end": r.get("train_end"),
                    "test_end": r.get("test_end"),
                    "train_size": r.get("train_size"),
                    "test_size": r.get("test_size"),
                    "test_best_pnl": tb.get("pnl", 0.0),
                    "test_best_win_rate": tb.get("win_rate", 0.0),
                    "test_best_trades": tb.get("trades", 0),
                    "test_baseline_pnl": bl.get("pnl", 0.0),
                    "test_baseline_win_rate": bl.get("win_rate", 0.0),
                    "test_baseline_trades": bl.get("trades", 0),
                    "delta_pnl": tb.get("pnl", 0.0) - bl.get("pnl", 0.0),
                }
            )


def write_walk_forward_md(path: Path, rows: List[Dict]) -> None:
    lines = ["# Walk-Forward Robustness", ""]
    if not rows:
        lines.append("Walk-forward disabled or insufficient data/windows.")
    else:
        deltas = []
        for r in rows:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            delta = tb.get("pnl", 0.0) - bl.get("pnl", 0.0)
            deltas.append(delta)
            lines.append(
                f"- window {r.get('window')}: test_best_pnl={tb.get('pnl',0.0):.4f}, test_baseline_pnl={bl.get('pnl',0.0):.4f}, delta={delta:.4f}, best_trades={tb.get('trades',0)}, baseline_trades={bl.get('trades',0)}"
            )
        lines.append("")
        lines.append(f"- Positive delta windows: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
        lines.append(f"- Avg delta pnl: {sum(deltas)/len(deltas):.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def choose_live_recommendation(results: List[Dict]) -> Dict:
    """Pick a guardrail-aware profile from single_ladder results.

    Score favors positive pnl with lower drawdown and lower loss-streak risk.
    """
    if not results:
        return None

    eligible = [
        r
        for r in results
        if r.get("execution_mode") == "single_ladder"
        and r.get("trades", 0) >= 5
        and r.get("win_rate", 0.0) >= 0.40
        and r.get("max_drawdown", 1.0) <= 0.20
    ]
    if not eligible:
        return None

    def score(r: Dict) -> float:
        pnl = float(r.get("pnl", 0.0))
        dd = float(r.get("max_drawdown", 0.0))
        streak = float(r.get("max_loss_streak", 0.0))
        return pnl - (dd * 2.0) - (streak * 0.02)

    return sorted(eligible, key=score, reverse=True)[0]


def write_holdout_md(path: Path, holdout_report: Dict) -> None:
    lines = ["# Holdout Robustness", ""]
    if not holdout_report:
        lines.append("Holdout disabled or insufficient data.")
    else:
        mode = holdout_report.get("mode", "single_ladder")
        tb = holdout_report.get("train_best") or {}
        tbase = holdout_report.get("train_baseline") or {}
        hb = holdout_report.get("test_best") or {}
        hbase = holdout_report.get("test_baseline") or {}
        lines += [
            f"- Mode: `{mode}`",
            "",
            "## Train",
            f"- Best pnl={tb.get('pnl',0.0):.4f}, win_rate={tb.get('win_rate',0.0):.3f}",
            f"- Baseline pnl={tbase.get('pnl',0.0):.4f}, win_rate={tbase.get('win_rate',0.0):.3f}",
            "",
            "## Holdout Test",
            f"- Best(train-selected) pnl={hb.get('pnl',0.0):.4f}, win_rate={hb.get('win_rate',0.0):.3f}",
            f"- Baseline pnl={hbase.get('pnl',0.0):.4f}, win_rate={hbase.get('win_rate',0.0):.3f}",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")




def build_live_profile_config(cfg: Dict, recommendation: Dict) -> Dict:
    if not recommendation:
        return {}

    risk = cfg.get("risk", {}) or {}
    execution = cfg.get("execution", {}) or {}

    return {
        "version": "backtest-live-profile/v1",
        "selected_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_basis": {
            "method": "guardrail-aware-single-ladder",
            "score_components": ["pnl", "max_drawdown", "max_loss_streak"],
            "constraints": {
                "min_trades": 5,
                "min_win_rate": 0.40,
                "max_drawdown": 0.20,
            },
        },
        "weights": recommendation.get("weights", {}),
        "expected_backtest_metrics": {
            "pnl": recommendation.get("pnl", 0.0),
            "win_rate": recommendation.get("win_rate", 0.0),
            "trades": recommendation.get("trades", 0),
            "max_drawdown": recommendation.get("max_drawdown", 0.0),
            "max_loss_streak": recommendation.get("max_loss_streak", 0),
        },
        "execution": {
            "mode": "single_ladder",
            "max_concurrent_ladders": 1,
            "fee_bps": execution.get("fee_bps", 0),
            "slippage_bps": execution.get("slippage_bps", 0),
            "slippage_model": execution.get("slippage_model", {}),
        },
        "risk": {
            "starting_bankroll": risk.get("starting_bankroll", 0.0),
            "risk_unit_pct": risk.get("risk_unit_pct", 0.0),
            "capital_utilization": risk.get("capital_utilization", 0.0),
            "max_drawdown": risk.get("max_drawdown", 0.0),
            "daily_loss_cap": risk.get("daily_loss_cap", 0.0),
            "freeze_ru_within_ladder": risk.get("freeze_ru_within_ladder", True),
        },
        "notes": [
            "Backtest recommendation only; not financial advice.",
            "Verify with forward tests/paper trading before live deployment.",
        ],
    }


def write_live_profile_json(path: Path, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def write_live_recommendation_md(path: Path, recommendation: Dict) -> None:
    lines = ["# Live Profile Recommendation", ""]
    if not recommendation:
        lines.append("No eligible recommendation found under current guardrails.")
    else:
        lines.extend(
            [
                "Selected from `single_ladder` results using guardrail-aware score.",
                "",
                f"- Weights: `{recommendation.get('weights', {})}`",
                f"- pnl={recommendation.get('pnl',0.0):.4f}",
                f"- win_rate={recommendation.get('win_rate',0.0):.3f}",
                f"- trades={recommendation.get('trades',0)}",
                f"- max_drawdown={recommendation.get('max_drawdown',0.0):.3f}",
                f"- max_loss_streak={recommendation.get('max_loss_streak',0)}",
                f"- stop_reason={recommendation.get('stop_reason','')}",
            ]
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")




def compute_live_readiness(report: Dict) -> Dict:
    baseline = report.get("baseline") or {}
    holdout = report.get("holdout") or {}
    walk_forward = report.get("walk_forward") or []
    feature_sources = report.get("feature_sources") or {}
    live_readiness = report.get("live_readiness") or {}
    decision_card = report.get("decision_card") or {}

    score = 100
    reasons = []

    b_trades = int(baseline.get("trades", 0) or 0)
    b_dd = float(baseline.get("max_drawdown", 1.0) or 1.0)
    b_wr = float(baseline.get("win_rate", 0.0) or 0.0)

    if b_trades < 20:
        score -= 20
        reasons.append("low_trade_count")
    if b_dd > 0.2:
        score -= 25
        reasons.append("high_drawdown")
    if b_wr < 0.45:
        score -= 10
        reasons.append("weak_win_rate")

    hb = holdout.get("test_best") or {}
    hbase = holdout.get("test_baseline") or {}
    if hb and hbase:
        if float(hb.get("pnl", 0.0)) < float(hbase.get("pnl", 0.0)):
            score -= 20
            reasons.append("holdout_underperforms_baseline")
    else:
        score -= 10
        reasons.append("missing_holdout_signal")

    if walk_forward:
        deltas = []
        for r in walk_forward:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            deltas.append(float(tb.get("pnl", 0.0)) - float(bl.get("pnl", 0.0)))
        pos = sum(1 for d in deltas if d > 0)
        if pos < max(1, len(deltas) // 2):
            score -= 15
            reasons.append("walk_forward_instability")
    else:
        score -= 10
        reasons.append("missing_walk_forward")

    synthetic_weight = 0.0
    for k, src in feature_sources.items():
        w = float((baseline.get("weights") or {}).get(k, 0.0) or 0.0)
        if not str(src).startswith("live"):
            synthetic_weight += w
    if synthetic_weight > 0.5:
        score -= 25
        reasons.append("high_synthetic_feature_share")

    score = max(0, min(100, score))
    tier = "green" if score >= 75 else ("yellow" if score >= 50 else "red")

    return {
        "score": score,
        "tier": tier,
        "reasons": reasons,
        "checks": {
            "baseline_trades": b_trades,
            "baseline_drawdown": b_dd,
            "baseline_win_rate": b_wr,
            "synthetic_weight_share": synthetic_weight,
            "walk_forward_windows": len(walk_forward),
        },
    }


def write_live_readiness_md(path: Path, payload: Dict) -> None:
    lines = ["# Live Readiness Gate", ""]
    lines.append(f"- score={payload.get('score', 0)}/100")
    lines.append(f"- tier={payload.get('tier', 'red')}")
    lines.append(f"- reasons={payload.get('reasons', [])}")
    checks = payload.get("checks") or {}
    lines.append("")
    lines.append("## Checks")
    for k, v in checks.items():
        lines.append(f"- {k}={v}")
    lines.append("")
    if payload.get("tier") != "green":
        lines.append("- ⚠️ Not live-ready: continue paper testing and improve weak checks.")
    else:
        lines.append("- ✅ Candidate live-ready (still require supervised rollout).")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_live_readiness_json(path: Path, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)




def build_deployment_decision_card(report: Dict) -> Dict:
    baseline = report.get("baseline") or {}
    holdout = report.get("holdout") or {}
    live_readiness = report.get("live_readiness") or {}
    decision_card = report.get("decision_card") or {}
    live_profile = report.get("live_profile") or {}

    hb = holdout.get("test_best") or {}
    hbase = holdout.get("test_baseline") or {}
    holdout_delta = float(hb.get("pnl", 0.0)) - float(hbase.get("pnl", 0.0)) if hb and hbase else 0.0

    readiness_tier = live_readiness.get("tier", "red")
    recommendation = "NO_GO"
    if readiness_tier == "green":
        recommendation = "GO_PAPER"
    elif readiness_tier == "yellow":
        recommendation = "PAPER_ONLY"

    next_actions = [
        "Run forward paper test for at least 2 weeks with identical risk controls.",
        "Track realized slippage vs modeled slippage and recalibrate slippage_model.",
        "Require manual approval before any live capital deployment.",
    ]

    return {
        "recommendation": recommendation,
        "readiness_tier": readiness_tier,
        "readiness_score": live_readiness.get("score", 0),
        "headline": {
            "baseline_pnl": baseline.get("pnl", 0.0),
            "baseline_win_rate": baseline.get("win_rate", 0.0),
            "baseline_trades": baseline.get("trades", 0),
            "holdout_delta_pnl": holdout_delta,
        },
        "risk_flags": live_readiness.get("reasons", []),
        "profile_snapshot": {
            "weights": live_profile.get("weights", {}),
            "execution": live_profile.get("execution", {}),
            "risk": live_profile.get("risk", {}),
        },
        "next_actions": next_actions,
    }


def write_deployment_decision_card_md(path: Path, payload: Dict) -> None:
    lines = ["# Deployment Decision Card", ""]
    lines.append(f"- recommendation={payload.get('recommendation', 'NO_GO')}")
    lines.append(f"- readiness_tier={payload.get('readiness_tier', 'red')}")
    lines.append(f"- readiness_score={payload.get('readiness_score', 0)}")
    head = payload.get("headline") or {}
    lines.append("")
    lines.append("## Headline metrics")
    lines.append(f"- baseline_pnl={head.get('baseline_pnl', 0.0):.4f}")
    lines.append(f"- baseline_win_rate={head.get('baseline_win_rate', 0.0):.3f}")
    lines.append(f"- baseline_trades={head.get('baseline_trades', 0)}")
    lines.append(f"- holdout_delta_pnl={head.get('holdout_delta_pnl', 0.0):.4f}")

    lines.append("")
    lines.append("## Risk flags")
    flags = payload.get("risk_flags") or []
    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Next actions")
    for a in payload.get("next_actions", []):
        lines.append(f"- {a}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_deployment_decision_card_json(path: Path, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_run_summary_md(path: Path, report: Dict) -> None:
    baseline = report.get("baseline") or {}
    top = (report.get("top10") or [{}])[0]
    mode_cmp = report.get("mode_comparison") or {}
    live_reco = report.get("live_recommendation") or {}
    live_profile = report.get("live_profile") or {}
    holdout = report.get("holdout") or {}
    cost_scenarios = report.get("cost_scenarios") or []
    fold_robustness = report.get("fold_robustness") or []
    walk_forward = report.get("walk_forward") or []
    multi_line_baseline = report.get("multi_line_baseline") or {}
    feature_sources = report.get("feature_sources") or {}
    live_readiness = report.get("live_readiness") or {}
    decision_card = report.get("decision_card") or {}

    lines = []
    lines.append("# Backtest Run Summary")
    lines.append("")
    lines.append(f"- Timestamp UTC: `{report.get('timestamp_utc','')}`")
    lines.append(f"- Execution mode (primary run): `{top.get('execution_mode','')}`")
    lines.append("")

    lines.append("## Baseline vs Best (primary run)")
    lines.append("")
    lines.append(f"- Baseline weights: `{baseline.get('weights',{})}`")
    lines.append(f"- Baseline: pnl={baseline.get('pnl',0.0):.4f}, win_rate={baseline.get('win_rate',0):.3f}, trades={baseline.get('trades',0)}, stop_reason={baseline.get('stop_reason','')} ")
    lines.append(f"- Best weights: `{top.get('weights',{})}`")
    lines.append(f"- Best: pnl={top.get('pnl',0.0):.4f}, win_rate={top.get('win_rate',0):.3f}, trades={top.get('trades',0)}, stop_reason={top.get('stop_reason','')} ")
    lines.append("")

    if mode_cmp:
        lines.append("## Mode comparison")
        lines.append("")
        for mode in ["all_trades", "single_ladder", "multi_line"]:
            payload = mode_cmp.get(mode) or {}
            best = payload.get("best") or {}
            base = payload.get("baseline") or {}
            lines.append(
                f"- {mode}: best_pnl={best.get('pnl',0.0):.4f}, best_win_rate={best.get('win_rate',0):.3f}, baseline_pnl={base.get('pnl',0.0):.4f}, baseline_win_rate={base.get('win_rate',0):.3f}"
            )
        lines.append("")

    lines.append("## Baseline category/regime snapshot")
    lines.append("")
    lines.extend(_top_category_rows(baseline.get("by_category", {}), top_n=10) or ["- no categories captured"])
    lines.append("")

    lines.append("## Suggested live profile (guardrail-aware)")
    lines.append("")
    if live_reco:
        lines.append(f"- Weights: `{live_reco.get('weights', {})}`")
        lines.append(f"- pnl={live_reco.get('pnl',0.0):.4f}, win_rate={live_reco.get('win_rate',0.0):.3f}, trades={live_reco.get('trades',0)}")
        lines.append(f"- max_drawdown={live_reco.get('max_drawdown',0.0):.3f}, max_loss_streak={live_reco.get('max_loss_streak',0)}")
    else:
        lines.append("- No eligible live recommendation under current guardrails.")

    lines.append("")
    lines.append("## Live profile config emitter")
    lines.append("")
    if live_profile:
        lines.append(f"- version={live_profile.get('version','')}")
        lines.append(f"- mode={((live_profile.get('execution') or {}).get('mode',''))}")
        lines.append(f"- max_concurrent_ladders={((live_profile.get('execution') or {}).get('max_concurrent_ladders',0))}")
        lines.append(f"- selected_weights={live_profile.get('weights',{})}")
    else:
        lines.append("- No live profile emitted (no eligible recommendation).")

    lines.append("")
    lines.append("## Holdout robustness (single_ladder)")
    lines.append("")
    if holdout:
        hb = holdout.get("test_best") or {}
        hbase = holdout.get("test_baseline") or {}
        lines.append(f"- Test best(train-selected): pnl={hb.get('pnl',0.0):.4f}, win_rate={hb.get('win_rate',0.0):.3f}, trades={hb.get('trades',0)}")
        lines.append(f"- Test baseline: pnl={hbase.get('pnl',0.0):.4f}, win_rate={hbase.get('win_rate',0.0):.3f}, trades={hbase.get('trades',0)}")
        if hbase.get('trades', 0) == 0:
            lines.append("- ⚠️ Holdout has zero baseline trades; robustness signal is weak.")
    else:
        lines.append("- Holdout disabled or insufficient data.")

    lines.append("")
    lines.append("## Cost sensitivity")
    lines.append("")
    if cost_scenarios:
        for s in cost_scenarios:
            b = s.get("baseline") or {}
            r = s.get("recommended") or {}
            delta = (r.get("pnl", 0.0) - b.get("pnl", 0.0)) if r else 0.0
            lines.append(
                f"- {s.get('label')}: fee={s.get('fee_bps')}bps, slip={s.get('slippage_bps')}bps, baseline_pnl={b.get('pnl',0.0):.4f}, reco_pnl={r.get('pnl',0.0):.4f}, delta={delta:.4f}"
            )
    else:
        lines.append("- No cost scenarios configured.")

    lines.append("")
    lines.append("## Fold robustness")
    lines.append("")
    if fold_robustness:
        deltas = []
        for r in fold_robustness:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            delta = tb.get("pnl", 0.0) - bl.get("pnl", 0.0)
            deltas.append(delta)
            lines.append(f"- fold {r.get('fold')}: delta_pnl={delta:.4f}, best_trades={tb.get('trades',0)}, baseline_trades={bl.get('trades',0)}")
        lines.append(f"- positive folds: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
        lines.append(f"- avg delta pnl: {sum(deltas)/len(deltas):.4f}")
    else:
        lines.append("- Fold validation disabled or insufficient data.")

    lines.append("")
    lines.append("## Walk-forward robustness")
    lines.append("")
    if walk_forward:
        wf_deltas = []
        for r in walk_forward:
            tb = r.get("test_best") or {}
            bl = r.get("test_baseline") or {}
            delta = tb.get("pnl", 0.0) - bl.get("pnl", 0.0)
            wf_deltas.append(delta)
            lines.append(f"- window {r.get('window')}: delta_pnl={delta:.4f}, best_trades={tb.get('trades',0)}, baseline_trades={bl.get('trades',0)}")
        lines.append(f"- positive windows: {sum(1 for d in wf_deltas if d > 0)}/{len(wf_deltas)}")
        lines.append(f"- avg delta pnl: {sum(wf_deltas)/len(wf_deltas):.4f}")
    else:
        lines.append("- Walk-forward disabled or insufficient windows.")

    lines.append("")
    lines.append("## Feature source audit")
    lines.append("")
    if feature_sources:
        for k in ["market", "cross", "news", "reddit", "trader"]:
            lines.append(f"- {k}: source={feature_sources.get(k, 'unknown')}")
    else:
        lines.append("- Feature sources not provided.")

    lines.append("")
    lines.append("## Live readiness gate")
    lines.append("")
    if live_readiness:
        lines.append(f"- score={live_readiness.get('score',0)}/100")
        lines.append(f"- tier={live_readiness.get('tier','red')}")
        lines.append(f"- reasons={live_readiness.get('reasons',[])}")
    else:
        lines.append("- No readiness assessment generated.")

    lines.append("")
    lines.append("## Deployment decision")
    lines.append("")
    if decision_card:
        lines.append(f"- recommendation={decision_card.get('recommendation','NO_GO')}")
        lines.append(f"- readiness_tier={decision_card.get('readiness_tier','red')}")
    else:
        lines.append("- No deployment decision card generated.")

    lines.append("")
    lines.append("## Multi-line baseline")
    lines.append("")
    if multi_line_baseline:
        lines.append(f"- pnl={multi_line_baseline.get('pnl',0.0):.4f}, win_rate={multi_line_baseline.get('win_rate',0.0):.3f}, trades={multi_line_baseline.get('trades',0)}")
        ls = multi_line_baseline.get("line_summary") or {}
        lines.append(f"- num_lines={ls.get('num_lines',0)}")
    else:
        lines.append("- No multi-line baseline result available.")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


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

    holdout_fraction = cfg.get("validation", {}).get("holdout_fraction", 0.0)
    train_snapshot, holdout_snapshot = split_snapshot(snapshot, holdout_fraction)

    leaderboard = run_mode_pass(train_snapshot, cfg)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comparison_payload = None
    comparison_csv = None
    if args.compare_modes:
        mode_payload = {}
        for mode in ["all_trades", "single_ladder", "multi_line"]:
            mode_board = run_mode_pass(train_snapshot, cfg, force_mode=mode)
            mode_payload[mode] = {
                "best": mode_board[0] if mode_board else None,
                "baseline": next((x for x in mode_board if x["weights"] == cfg["weights"]), None),
            }
        comparison_payload = mode_payload
        comparison_csv = outdir / f"backtest_{ts}_mode_compare.csv"
        write_mode_comparison_csv(comparison_csv, mode_payload)

    single_ladder_board = run_mode_pass(train_snapshot, cfg, force_mode="single_ladder")
    multi_line_board = run_mode_pass(train_snapshot, cfg, force_mode="multi_line")
    multi_line_baseline = next((x for x in multi_line_board if x["weights"] == cfg["weights"]), None)

    live_recommendation = choose_live_recommendation(single_ladder_board)
    live_profile = build_live_profile_config(cfg, live_recommendation)
    holdout_report = evaluate_holdout(single_ladder_board, holdout_snapshot, cfg, mode="single_ladder")

    cost_scenarios_cfg = cfg.get("validation", {}).get("cost_scenarios", [])
    cost_scenarios = evaluate_cost_scenarios(
        train_snapshot,
        cfg,
        baseline_weights=cfg["weights"],
        recommended_weights=(live_recommendation or {}).get("weights"),
        scenarios=cost_scenarios_cfg,
        mode="single_ladder",
    )

    fold_count = int(cfg.get("validation", {}).get("folds", 0) or 0)
    fold_robustness = evaluate_fold_robustness(snapshot, cfg, fold_count, mode="single_ladder")
    walk_forward = evaluate_walk_forward(snapshot, cfg, mode="single_ladder")

    dataset_meta = {
        "total_markets": len(snapshot),
        "train_markets": len(train_snapshot),
        "holdout_markets": len(holdout_snapshot),
        "holdout_fraction": holdout_fraction,
    }
    baseline_result = next((x for x in leaderboard if x["weights"] == cfg["weights"]), None)
    feature_sources = signal_source_map(cfg)

    full_path = outdir / f"backtest_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(
            {
                "timestamp_utc": ts,
                "config": cfg,
                "dataset": dataset_meta,
                "top10": leaderboard[:10],
                "baseline": baseline_result,
                "all_results": leaderboard,
                "mode_comparison": comparison_payload,
                "holdout": holdout_report,
                "cost_scenarios": cost_scenarios,
                "fold_robustness": fold_robustness,
                "walk_forward": walk_forward,
                "multi_line_baseline": multi_line_baseline,
        "feature_sources": feature_sources,
                "feature_sources": feature_sources,
                "live_readiness": compute_live_readiness({"baseline": baseline_result, "holdout": holdout_report, "walk_forward": walk_forward, "feature_sources": feature_sources}),
                "decision_card": build_deployment_decision_card({"baseline": baseline_result, "holdout": holdout_report, "live_profile": live_profile, "live_readiness": compute_live_readiness({"baseline": baseline_result, "holdout": holdout_report, "walk_forward": walk_forward, "feature_sources": feature_sources})}),
            },
            f,
            indent=2,
        )

    leaderboard_csv = outdir / f"backtest_{ts}_leaderboard.csv"
    category_csv = outdir / f"backtest_{ts}_categories.csv"
    summary_md = outdir / f"backtest_{ts}_summary.md"
    write_leaderboard_csv(leaderboard_csv, leaderboard)
    write_category_csv(category_csv, leaderboard)

    temp_report = {
        "timestamp_utc": ts,
        "top10": leaderboard[:10],
        "baseline": baseline_result,
        "mode_comparison": comparison_payload,
        "live_recommendation": live_recommendation,
        "live_profile": live_profile,
        "holdout": holdout_report,
        "cost_scenarios": cost_scenarios,
        "fold_robustness": fold_robustness,
        "walk_forward": walk_forward,
        "multi_line_baseline": multi_line_baseline,
        "feature_sources": feature_sources,
    }
    live_readiness = compute_live_readiness(temp_report)
    decision_card = build_deployment_decision_card({**temp_report, "live_readiness": live_readiness})
    summary_report = {**temp_report, "live_readiness": live_readiness, "decision_card": decision_card}
    write_run_summary_md(summary_md, summary_report)

    live_reco_md = outdir / f"backtest_{ts}_live_recommendation.md"
    live_profile_json = outdir / f"backtest_{ts}_live_profile.json"
    holdout_md = outdir / f"backtest_{ts}_holdout.md"
    cost_csv = outdir / f"backtest_{ts}_cost_scenarios.csv"
    cost_md = outdir / f"backtest_{ts}_cost_scenarios.md"
    fold_csv = outdir / f"backtest_{ts}_fold_robustness.csv"
    fold_md = outdir / f"backtest_{ts}_fold_robustness.md"
    multiline_md = outdir / f"backtest_{ts}_multiline.md"
    walk_forward_csv = outdir / f"backtest_{ts}_walk_forward.csv"
    walk_forward_md = outdir / f"backtest_{ts}_walk_forward.md"
    data_quality_md = outdir / f"backtest_{ts}_data_quality.md"
    feature_sources_md = outdir / f"backtest_{ts}_feature_sources.md"
    feature_sources_json = outdir / f"backtest_{ts}_feature_sources.json"
    live_readiness_md = outdir / f"backtest_{ts}_live_readiness.md"
    live_readiness_json = outdir / f"backtest_{ts}_live_readiness.json"
    decision_card_md = outdir / f"backtest_{ts}_decision_card.md"
    decision_card_json = outdir / f"backtest_{ts}_decision_card.json"
    write_live_recommendation_md(live_reco_md, live_recommendation)
    write_live_profile_json(live_profile_json, live_profile)
    write_holdout_md(holdout_md, holdout_report)
    write_cost_scenarios_csv(cost_csv, cost_scenarios)
    write_cost_scenarios_md(cost_md, cost_scenarios)
    write_fold_robustness_csv(fold_csv, fold_robustness)
    write_fold_robustness_md(fold_md, fold_robustness)
    write_walk_forward_csv(walk_forward_csv, walk_forward)
    write_walk_forward_md(walk_forward_md, walk_forward)
    write_multiline_md(multiline_md, multi_line_baseline)
    write_data_quality_md(data_quality_md, dataset_meta, baseline_result, holdout_report)
    write_feature_sources_md(feature_sources_md, feature_sources, cfg.get("weights", {}))
    write_feature_sources_json(feature_sources_json, feature_sources, cfg.get("weights", {}))
    write_live_readiness_md(live_readiness_md, summary_report.get("live_readiness", {}))
    write_live_readiness_json(live_readiness_json, summary_report.get("live_readiness", {}))
    write_deployment_decision_card_md(decision_card_md, summary_report.get("decision_card", {}))
    write_deployment_decision_card_json(decision_card_json, summary_report.get("decision_card", {}))

    print(f"Wrote report: {full_path}")
    print(f"Wrote leaderboard CSV: {leaderboard_csv}")
    print(f"Wrote category CSV: {category_csv}")
    print(f"Wrote summary MD: {summary_md}")
    print(f"Wrote live recommendation MD: {live_reco_md}")
    print(f"Wrote live profile JSON: {live_profile_json}")
    print(f"Wrote holdout MD: {holdout_md}")
    print(f"Wrote cost scenarios CSV: {cost_csv}")
    print(f"Wrote cost scenarios MD: {cost_md}")
    print(f"Wrote fold robustness CSV: {fold_csv}")
    print(f"Wrote fold robustness MD: {fold_md}")
    print(f"Wrote walk-forward CSV: {walk_forward_csv}")
    print(f"Wrote walk-forward MD: {walk_forward_md}")
    print(f"Wrote multi-line MD: {multiline_md}")
    print(f"Wrote data quality MD: {data_quality_md}")
    print(f"Wrote feature sources MD: {feature_sources_md}")
    print(f"Wrote feature sources JSON: {feature_sources_json}")
    print(f"Wrote live readiness MD: {live_readiness_md}")
    print(f"Wrote live readiness JSON: {live_readiness_json}")
    print(f"Wrote decision card MD: {decision_card_md}")
    print(f"Wrote decision card JSON: {decision_card_json}")
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
