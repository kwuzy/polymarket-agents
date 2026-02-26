import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

try:
    from backtest_runner import load_config, load_snapshot, apply_date_range_filter, split_snapshot, run_mode_pass
except ModuleNotFoundError:
    from scripts.python.backtest_runner import load_config, load_snapshot, apply_date_range_filter, split_snapshot, run_mode_pass


def parse_iso(v):
    if not v:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def set_feature_flags(cfg, real=False, whales=False, category=False):
    c = copy.deepcopy(cfg)
    feats = c.setdefault("features", {})
    feats.setdefault("real_signals", {})["enabled"] = bool(real)
    feats.setdefault("whales", {})["enabled"] = bool(whales)
    feats.setdefault("category_models", {})["enabled"] = bool(category)
    return c


def summarize(board, baseline_weights):
    best = board[0] if board else {}
    baseline = next((x for x in board if x.get("weights") == baseline_weights), {})
    return {
        "best_pnl": best.get("pnl", 0.0),
        "best_win_rate": best.get("win_rate", 0.0),
        "best_trades": best.get("trades", 0),
        "baseline_pnl": baseline.get("pnl", 0.0),
        "baseline_win_rate": baseline.get("win_rate", 0.0),
        "baseline_trades": baseline.get("trades", 0),
    }


def date_meta_quality(meta):
    start = parse_iso(meta.get("applied_min"))
    end = parse_iso(meta.get("applied_max"))
    span_hours = 0.0
    if start and end:
        span_hours = max(0.0, (end - start).total_seconds() / 3600.0)
    return {
        "markets": int(meta.get("filtered_count", 0) or 0),
        "span_hours": span_hours,
    }


def maybe_apply_fallback(cfg, snapshot, date_meta):
    rules = ((cfg.get("validation") or {}).get("ablation_guardrails") or {})
    enabled = bool(rules.get("enabled", True))
    if not enabled:
        return cfg, snapshot, date_meta, {"triggered": False, "reason": "disabled"}

    q = date_meta_quality(date_meta)
    min_markets = int(rules.get("min_markets", 100) or 100)
    min_span_hours = float(rules.get("min_span_hours", 24.0) or 24.0)

    ok = (q["markets"] >= min_markets) and (q["span_hours"] >= min_span_hours)
    if ok:
        return cfg, snapshot, date_meta, {
            "triggered": False,
            "reason": "passed",
            "min_markets": min_markets,
            "min_span_hours": min_span_hours,
            "actual_markets": q["markets"],
            "actual_span_hours": q["span_hours"],
        }

    fb = rules.get("fallback_date_range") or {}
    if not fb:
        return cfg, snapshot, date_meta, {
            "triggered": False,
            "reason": "failed_no_fallback",
            "min_markets": min_markets,
            "min_span_hours": min_span_hours,
            "actual_markets": q["markets"],
            "actual_span_hours": q["span_hours"],
        }

    c2 = copy.deepcopy(cfg)
    c2.setdefault("validation", {}).setdefault("date_range", {})
    c2["validation"]["date_range"]["enabled"] = True
    c2["validation"]["date_range"]["field"] = fb.get("field", c2["validation"]["date_range"].get("field", "createdAt"))
    c2["validation"]["date_range"]["start"] = fb.get("start")
    c2["validation"]["date_range"]["end"] = fb.get("end")

    s2, m2 = apply_date_range_filter(load_snapshot(c2["snapshot_path"]), c2)
    q2 = date_meta_quality(m2)
    return c2, s2, m2, {
        "triggered": True,
        "reason": "failed_primary_used_fallback",
        "min_markets": min_markets,
        "min_span_hours": min_span_hours,
        "actual_markets": q["markets"],
        "actual_span_hours": q["span_hours"],
        "fallback_markets": q2["markets"],
        "fallback_span_hours": q2["span_hours"],
        "fallback_range": c2["validation"]["date_range"],
    }


def main():
    ap = argparse.ArgumentParser(description="Ablation runner for feature stacks")
    ap.add_argument("--config", default="configs/backtest_v0_timestamped_2025.json")
    ap.add_argument("--outdir", default="outputs/backtests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    snapshot = load_snapshot(cfg["snapshot_path"])
    snapshot, date_meta = apply_date_range_filter(snapshot, cfg)

    cfg, snapshot, date_meta, guardrail_info = maybe_apply_fallback(cfg, snapshot, date_meta)

    train, _ = split_snapshot(snapshot, cfg.get("validation", {}).get("holdout_fraction", 0.25))

    scenarios = [
        ("baseline", dict(real=False, whales=False, category=False)),
        ("plus_real", dict(real=True, whales=False, category=False)),
        ("plus_real_whales", dict(real=True, whales=True, category=False)),
        ("plus_all", dict(real=True, whales=True, category=True)),
    ]

    rows = []
    for name, flags in scenarios:
        c = set_feature_flags(cfg, **flags)
        board = run_mode_pass(train, c)
        summary = summarize(board, c["weights"])
        rows.append({"scenario": name, **flags, **summary})

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / f"ablation_{ts}.json"
    mpath = outdir / f"ablation_{ts}.md"

    payload = {
        "timestamp_utc": ts,
        "config": args.config,
        "date_filter": date_meta,
        "guardrail": guardrail_info,
        "rows": rows,
    }
    jpath.write_text(json.dumps(payload, indent=2))

    lines = ["# Feature Ablation Report", "", f"- timestamp_utc: `{ts}`", f"- config: `{args.config}`", "", "## Dataset window", ""]
    lines.append(f"- enabled: {date_meta.get('enabled')}")
    lines.append(f"- requested: {date_meta.get('start')} -> {date_meta.get('end')}")
    lines.append(f"- applied: {date_meta.get('applied_min')} -> {date_meta.get('applied_max')}")
    lines.append(f"- markets: {date_meta.get('filtered_count')}/{date_meta.get('input_count')}")
    lines.append("")
    lines.append("## Ablation guardrail")
    lines.append("")
    lines.append(f"- triggered: {guardrail_info.get('triggered')}")
    lines.append(f"- reason: {guardrail_info.get('reason')}")
    lines.append(f"- min_markets: {guardrail_info.get('min_markets')}")
    lines.append(f"- min_span_hours: {guardrail_info.get('min_span_hours')}")
    lines.append(f"- actual_markets: {guardrail_info.get('actual_markets')}")
    lines.append(f"- actual_span_hours: {guardrail_info.get('actual_span_hours')}")
    if guardrail_info.get("triggered"):
        lines.append(f"- fallback_markets: {guardrail_info.get('fallback_markets')}")
        lines.append(f"- fallback_span_hours: {guardrail_info.get('fallback_span_hours')}")
        lines.append(f"- fallback_range: {guardrail_info.get('fallback_range')}")

    lines += ["", "## Scenarios (all_trades leaderboard summary)", "", "| scenario | best_pnl | best_win_rate | baseline_pnl | baseline_win_rate |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['best_pnl']:.4f} | {r['best_win_rate']:.3f} | {r['baseline_pnl']:.4f} | {r['baseline_win_rate']:.3f} |")
    mpath.write_text("\n".join(lines) + "\n")

    print(jpath)
    print(mpath)


if __name__ == "__main__":
    main()
