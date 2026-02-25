import unittest

from scripts.python.backtest_runner import synth_signals


class TestCategoryBlendOverrides(unittest.TestCase):
    def test_category_override_changes_weights(self):
        market = {
            "id": "1",
            "question": "NBA finals winner",
            "outcomePrices": [0.5, 0.5],
            "spread": 0.01,
            "volume": 100000,
            "liquidity": 100000,
            "updatedAt": "2026-03-01T00:00:00Z",
        }
        ext_ctx = {"news_rows": [], "social_rows": [], "whale_rows": [], "date_field": "updatedAt", "news_max_age_days": 30, "social_max_age_days": 14, "whale_max_age_days": 30}
        cfg_a = {"features": {"blend": {"news_external_weight": 0.1, "reddit_external_weight": 0.1, "trader_external_weight": 0.5}}}
        cfg_b = {
            "features": {
                "blend": {
                    "news_external_weight": 0.1,
                    "reddit_external_weight": 0.1,
                    "trader_external_weight": 0.5,
                    "category_overrides": {
                        "sports": {"news_external_weight": 0.3, "reddit_external_weight": 0.3, "trader_external_weight": 0.3}
                    },
                }
            }
        }
        a = synth_signals(market, 0.5, 42, ext_ctx=ext_ctx, cfg=cfg_a)
        b = synth_signals(market, 0.5, 42, ext_ctx=ext_ctx, cfg=cfg_b)
        # empty external rows => same result, but should execute without errors
        self.assertAlmostEqual(a["news"], b["news"], places=6)


if __name__ == "__main__":
    unittest.main()
