import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

from backtest_runner import load_config, load_snapshot, apply_date_range_filter, split_snapshot, run_mode_pass


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


def main():
    ap = argparse.ArgumentParser(description="Ablation runner for feature stacks")
    ap.add_argument("--config", default="configs/backtest_v0_timestamped_2025.json")
    ap.add_argument("--outdir", default="outputs/backtests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    snapshot = load_snapshot(cfg["snapshot_path"])
    snapshot, date_meta = apply_date_range_filter(snapshot, cfg)
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
        "rows": rows,
    }
    jpath.write_text(json.dumps(payload, indent=2))

    lines = ["# Feature Ablation Report", "", f"- timestamp_utc: `{ts}`", f"- config: `{args.config}`", "", "## Dataset window", ""]
    lines.append(f"- enabled: {date_meta.get('enabled')}")
    lines.append(f"- requested: {date_meta.get('start')} -> {date_meta.get('end')}")
    lines.append(f"- applied: {date_meta.get('applied_min')} -> {date_meta.get('applied_max')}")
    lines.append(f"- markets: {date_meta.get('filtered_count')}/{date_meta.get('input_count')}")
    lines += ["", "## Scenarios (all_trades leaderboard summary)", "", "| scenario | best_pnl | best_win_rate | baseline_pnl | baseline_win_rate |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['best_pnl']:.4f} | {r['best_win_rate']:.3f} | {r['baseline_pnl']:.4f} | {r['baseline_win_rate']:.3f} |")
    mpath.write_text("\n".join(lines) + "\n")

    print(jpath)
    print(mpath)


if __name__ == "__main__":
    main()
