import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

from backtest_runner import load_config, load_snapshot, apply_date_range_filter, split_snapshot, run_mode_pass


def run_once(cfg):
    snapshot = load_snapshot(cfg["snapshot_path"])
    snapshot, _ = apply_date_range_filter(snapshot, cfg)
    train, _ = split_snapshot(snapshot, cfg.get("validation", {}).get("holdout_fraction", 0.25))
    board = run_mode_pass(train, cfg)
    baseline = next((x for x in board if x.get("weights") == cfg["weights"]), {})
    best = board[0] if board else {}
    return {
        "best_pnl": best.get("pnl", 0.0),
        "best_win_rate": best.get("win_rate", 0.0),
        "baseline_pnl": baseline.get("pnl", 0.0),
        "baseline_win_rate": baseline.get("win_rate", 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep external blend weights for real/whale signal integration")
    ap.add_argument("--config", default="configs/backtest_v0_timestamped_updatedAt_2026.json")
    ap.add_argument("--outdir", default="outputs/backtests")
    args = ap.parse_args()

    base = load_config(args.config)
    base.setdefault("features", {}).setdefault("real_signals", {})["enabled"] = True
    base.setdefault("features", {}).setdefault("whales", {})["enabled"] = True

    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for nw in grid:
        for rw in grid:
            for tw in grid:
                cfg = copy.deepcopy(base)
                b = cfg.setdefault("features", {}).setdefault("blend", {})
                b["news_external_weight"] = nw
                b["reddit_external_weight"] = rw
                b["trader_external_weight"] = tw
                r = run_once(cfg)
                rows.append({"news_w": nw, "reddit_w": rw, "trader_w": tw, **r})

    rows.sort(key=lambda x: (x["baseline_pnl"], x["baseline_win_rate"]), reverse=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / f"blend_sweep_{ts}.json"
    mp = outdir / f"blend_sweep_{ts}.md"
    jp.write_text(json.dumps({"timestamp_utc": ts, "rows": rows[:30]}, indent=2))

    lines = ["# Blend Sweep", "", "Top 15 by baseline_pnl", "", "| rank | news_w | reddit_w | trader_w | baseline_pnl | baseline_win_rate | best_pnl |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows[:15], 1):
        lines.append(f"| {i} | {r['news_w']:.1f} | {r['reddit_w']:.1f} | {r['trader_w']:.1f} | {r['baseline_pnl']:.4f} | {r['baseline_win_rate']:.3f} | {r['best_pnl']:.4f} |")
    mp.write_text("\n".join(lines) + "\n")
    print(jp)
    print(mp)


if __name__ == "__main__":
    main()
