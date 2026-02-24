import unittest

from scripts.python.backtest_runner import synth_signals


class TestMicroCategorySignals(unittest.TestCase):
    def test_category_multiplier_applies(self):
        mkt = {
            "id": "1",
            "question": "NBA finals winner",
            "outcomePrices": [0.5, 0.5],
            "spread": 0.01,
            "volume": 100000,
            "liquidity": 100000,
        }
        cfg_on = {
            "features": {
                "category_models": {
                    "enabled": True,
                    "default_multiplier": 1.0,
                    "multipliers": {"sports": 1.1},
                }
            }
        }
        cfg_off = {"features": {"category_models": {"enabled": False}}}
        a = synth_signals(mkt, 0.5, 42, cfg=cfg_on)
        b = synth_signals(mkt, 0.5, 42, cfg=cfg_off)
        self.assertGreaterEqual(a["cross"], b["cross"])

    def test_microstructure_affects_signal(self):
        good = {
            "id": "2",
            "question": "NBA finals winner",
            "outcomePrices": [0.5, 0.5],
            "spread": 0.005,
            "volume": 200000,
            "liquidity": 200000,
        }
        bad = {
            "id": "2",
            "question": "NBA finals winner",
            "outcomePrices": [0.5, 0.5],
            "spread": 0.6,
            "volume": 10,
            "liquidity": 10,
        }
        sg = synth_signals(good, 0.5, 42)
        sb = synth_signals(bad, 0.5, 42)
        self.assertGreaterEqual(sg["cross"], sb["cross"])


if __name__ == "__main__":
    unittest.main()
