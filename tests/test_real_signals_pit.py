import unittest

from scripts.python.real_signals import build_trade_features, load_news, load_social


class TestRealSignalsPIT(unittest.TestCase):
    def test_future_items_not_used_for_earlier_trade(self):
        news_rows = [
            {"ts": "2025-01-01T00:00:00Z", "category": "sports", "sentiment": 1.0, "text": "old"},
            {"ts": "2025-12-01T00:00:00Z", "category": "sports", "sentiment": -1.0, "text": "future"},
        ]
        social_rows = [
            {"ts": "2025-01-02T00:00:00Z", "category": "sports", "sentiment": 0.5, "text": "old social"},
            {"ts": "2025-12-02T00:00:00Z", "category": "sports", "sentiment": -0.5, "text": "future social"},
        ]
        import json, tempfile
        with tempfile.TemporaryDirectory() as d:
            import pathlib
            np = pathlib.Path(d) / "n.json"
            sp = pathlib.Path(d) / "s.json"
            np.write_text(json.dumps(news_rows))
            sp.write_text(json.dumps(social_rows))
            news = load_news(str(np))
            social = load_social(str(sp))

        trades = [
            {"trade_id": "A", "market_id": "m1", "category": "sports", "ts": "2025-01-10T00:00:00Z"},
            {"trade_id": "B", "market_id": "m2", "category": "sports", "ts": "2025-12-15T00:00:00Z"},
        ]
        feats = build_trade_features(trades, news, social)
        a = next(x for x in feats if x["trade_id"] == "A")
        b = next(x for x in feats if x["trade_id"] == "B")

        self.assertGreater(a["signals"]["news"]["30d"]["sentiment_mean"], 0)
        self.assertGreater(a["signals"]["social"]["30d"]["sentiment_mean"], 0)
        self.assertAlmostEqual(b["signals"]["news"]["30d"]["sentiment_mean"], -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
