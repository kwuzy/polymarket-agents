import unittest

from scripts.python.backtest_runner import build_external_signal_context, signal_source_map, synth_signals


class TestRealWhaleIntegration(unittest.TestCase):
    def test_signal_sources_flip_when_enabled(self):
        cfg = {
            "features": {
                "real_signals": {"enabled": True},
                "whales": {"enabled": True},
            }
        }
        sm = signal_source_map(cfg)
        self.assertIn("live_news", sm["news"])
        self.assertIn("live_social", sm["reddit"])
        self.assertIn("live_whale", sm["trader"])

    def test_synth_signals_accept_external_context(self):
        cfg = {
            "validation": {"date_range": {"field": "createdAt"}},
            "features": {
                "real_signals": {
                    "enabled": True,
                    "news_path": "data/signals/news_events.json",
                    "social_path": "data/signals/social_events.json",
                },
                "whales": {
                    "enabled": True,
                    "source": {"activity_cache": "data/whales/whale_activity.json"},
                },
            },
        }
        ctx = build_external_signal_context(cfg)
        mkt = {
            "id": "123",
            "createdAt": "2025-06-04T00:00:00Z",
            "outcomePrices": [0.5, 0.5],
            "spread": 0.02,
            "question": "NBA finals winner?",
        }
        sig = synth_signals(mkt, 0.5, 42, ext_ctx=ctx)
        self.assertTrue(0.0 <= sig["news"] <= 1.0)
        self.assertTrue(0.0 <= sig["reddit"] <= 1.0)
        self.assertTrue(0.0 <= sig["trader"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
