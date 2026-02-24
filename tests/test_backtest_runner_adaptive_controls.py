import unittest
from unittest.mock import patch

from scripts.python.backtest_runner import run_backtest


def _snapshot(n=30):
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "id": str(i),
                "outcomePrices": [0.5, 0.5],
                "spread": 0.01,
                "question": f"Market {i}",
                "volume": 100000,
                "liquidity": 100000,
            }
        )
    return rows


def _cfg():
    return {
        "seed": 42,
        "filters": {"min_prob": 0.01, "max_prob": 0.99, "max_spread": 1.0},
        "risk": {
            "starting_bankroll": 1000.0,
            "risk_unit_pct": 0.0001,
            "capital_utilization": 1.0,
            "max_drawdown": 0.95,
            "daily_loss_cap": 0.95,
            "freeze_ru_within_ladder": True,
        },
        "execution": {
            "mode": "single_ladder",
            "max_concurrent_ladders": 1,
            "num_lines": 1,
            "line_assignment": "round_robin",
            "fee_bps": 0,
            "slippage_bps": 0,
            "slippage_model": {"enabled": False},
            "adaptive_controls": {
                "loss_streak_for_throttle": 2,
                "ru_throttle_multiplier": 0.5,
                "soft_ladder_cap": True,
                "regime_pause_enabled": False,
                "regime_window": 5,
                "regime_min_win_rate": 0.4,
                "regime_cooldown_markets": 5,
            },
        },
        "weights": {"market": 1.0, "cross": 0.0, "news": 0.0, "reddit": 0.0, "trader": 0.0},
    }


class TestAdaptiveControls(unittest.TestCase):
    @patch("scripts.python.backtest_runner.compute_outcome", return_value=False)
    @patch("scripts.python.backtest_runner.weighted_prob", return_value=0.6)
    @patch("scripts.python.backtest_runner.synth_signals", return_value={"market": 0.6, "cross": 0.6, "news": 0.6, "reddit": 0.6, "trader": 0.6})
    def test_ru_throttle_applies_after_loss_streak(self, *_):
        cfg = _cfg()
        out = run_backtest(_snapshot(20), cfg, cfg["weights"])
        self.assertGreater(out["diagnostics"]["ru_throttle_trades"], 0)

    @patch("scripts.python.backtest_runner.compute_outcome", return_value=False)
    @patch("scripts.python.backtest_runner.weighted_prob", return_value=0.6)
    @patch("scripts.python.backtest_runner.synth_signals", return_value={"market": 0.6, "cross": 0.6, "news": 0.6, "reddit": 0.6, "trader": 0.6})
    def test_regime_pause_skips_candidates(self, *_):
        cfg = _cfg()
        controls = cfg["execution"]["adaptive_controls"]
        controls["regime_pause_enabled"] = True
        controls["regime_window"] = 3
        controls["regime_min_win_rate"] = 0.67
        controls["regime_cooldown_markets"] = 5
        out = run_backtest(_snapshot(30), cfg, cfg["weights"])
        self.assertGreater(out["diagnostics"]["regime_pause_triggers"], 0)
        self.assertGreater(out["diagnostics"]["skipped_regime_pause"], 0)

    @patch("scripts.python.backtest_runner.compute_outcome", return_value=False)
    @patch("scripts.python.backtest_runner.weighted_prob", return_value=0.6)
    @patch("scripts.python.backtest_runner.synth_signals", return_value={"market": 0.6, "cross": 0.6, "news": 0.6, "reddit": 0.6, "trader": 0.6})
    def test_soft_cap_prevents_hard_capacity_break(self, *_):
        cfg = _cfg()
        cfg["risk"]["risk_unit_pct"] = 0.05  # keeps k_max small and exercises one-step-early cap
        out = run_backtest(_snapshot(20), cfg, cfg["weights"])
        self.assertGreater(out["diagnostics"]["soft_cap_hits"], 0)


if __name__ == "__main__":
    unittest.main()
