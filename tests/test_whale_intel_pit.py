import unittest
from datetime import datetime, timezone

from scripts.python.whale_intel import WhaleTrade, build_wallet_snapshot, build_trade_level_whale_features


class TestWhaleIntelPointInTime(unittest.TestCase):
    def test_wallet_snapshot_excludes_future_trades(self):
        as_of = datetime(2025, 6, 15, tzinfo=timezone.utc)
        rows = [
            WhaleTrade(wallet="w1", market_id="m1", category="sports", pnl=10, won=True, ts=datetime(2025, 6, 1, tzinfo=timezone.utc)),
            WhaleTrade(wallet="w1", market_id="m2", category="sports", pnl=-5, won=False, ts=datetime(2025, 6, 20, tzinfo=timezone.utc)),  # future
        ]
        snap = build_wallet_snapshot(rows, as_of, windows=[30], categories=["sports"])
        self.assertIn("w1", snap)
        self.assertEqual(snap["w1"]["windows"]["30d"]["trades"], 1)
        self.assertEqual(snap["w1"]["windows"]["30d"]["pnl"], 10)

    def test_trade_level_features_are_point_in_time(self):
        whale = [
            WhaleTrade(wallet="w1", market_id="m1", category="sports", pnl=2, won=True, ts=datetime(2025, 1, 1, tzinfo=timezone.utc)),
            WhaleTrade(wallet="w1", market_id="m2", category="sports", pnl=100, won=True, ts=datetime(2025, 12, 1, tzinfo=timezone.utc)),  # future for trade A
            WhaleTrade(wallet="w2", market_id="m3", category="sports", pnl=1, won=True, ts=datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]
        trades = [
            {"trade_id": "A", "market_id": "x", "category": "sports", "ts": "2025-01-10T00:00:00Z"},
            {"trade_id": "B", "market_id": "y", "category": "sports", "ts": "2025-12-15T00:00:00Z"},
        ]
        feats = build_trade_level_whale_features(trades, whale, windows=[7, 30, 90], categories=["sports"], top_n=2)
        self.assertEqual(len(feats), 2)

        a = next(x for x in feats if x["trade_id"] == "A")
        b = next(x for x in feats if x["trade_id"] == "B")

        # For A, future +100 trade should not be present in 30d pnl
        a_w1 = next(x for x in a["whale_top_n"] if x["wallet"] == "w1")
        self.assertLessEqual(a_w1["w30"]["pnl"], 2)

        # For B, that later trade can be counted
        b_w1 = next(x for x in b["whale_top_n"] if x["wallet"] == "w1")
        self.assertGreaterEqual(b_w1["w30"]["pnl"], 100)


if __name__ == "__main__":
    unittest.main()
