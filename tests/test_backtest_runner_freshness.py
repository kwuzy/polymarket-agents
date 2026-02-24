import json
import tempfile
import unittest
from pathlib import Path

from scripts.python.backtest_runner import build_external_signal_context, synth_signals


class TestFreshnessGuardrails(unittest.TestCase):
    def test_stale_news_social_whale_ignored_by_max_age(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            news = d / "news.json"
            social = d / "social.json"
            whales = d / "whales.json"
            news.write_text(json.dumps([
                {"ts": "2025-01-01T00:00:00Z", "category": "sports", "sentiment": 1.0}
            ]))
            social.write_text(json.dumps([
                {"ts": "2025-01-01T00:00:00Z", "category": "sports", "sentiment": 1.0}
            ]))
            whales.write_text(json.dumps([
                {"ts": "2025-01-01T00:00:00Z", "category": "sports", "pnl": 100.0}
            ]))

            cfg = {
                "validation": {"date_range": {"field": "updatedAt"}},
                "features": {
                    "real_signals": {"enabled": True, "news_path": str(news), "social_path": str(social)},
                    "whales": {"enabled": True, "source": {"activity_cache": str(whales)}},
                    "freshness": {"enabled": True, "news_max_age_days": 7, "social_max_age_days": 7, "whale_max_age_days": 7},
                },
            }
            ctx = build_external_signal_context(cfg)
            mkt = {
                "id": "1",
                "updatedAt": "2026-02-24T00:00:00Z",
                "question": "NBA finals",
                "outcomePrices": [0.5, 0.5],
                "spread": 0.01,
                "volume": 100000,
                "liquidity": 100000,
            }
            sig = synth_signals(mkt, 0.5, 42, ext_ctx=ctx, cfg=cfg)
            base = synth_signals(mkt, 0.5, 42, ext_ctx=None, cfg={"features": {}})

            # stale external rows should not materially alter baseline signal
            self.assertAlmostEqual(sig["news"], base["news"], places=6)
            self.assertAlmostEqual(sig["reddit"], base["reddit"], places=6)
            self.assertAlmostEqual(sig["trader"], base["trader"], places=6)


if __name__ == "__main__":
    unittest.main()
