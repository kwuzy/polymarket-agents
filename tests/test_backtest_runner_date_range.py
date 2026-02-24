import unittest

from scripts.python.backtest_runner import apply_date_range_filter


class TestDateRangeFilter(unittest.TestCase):
    def test_filters_snapshot_by_created_at(self):
        snapshot = [
            {"id": "1", "createdAt": "2025-01-01T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
            {"id": "2", "createdAt": "2025-06-01T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
            {"id": "3", "createdAt": "2026-01-01T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
        ]
        cfg = {
            "validation": {
                "date_range": {
                    "enabled": True,
                    "field": "createdAt",
                    "start": "2025-03-01T00:00:00Z",
                    "end": "2025-12-31T23:59:59Z",
                }
            }
        }

        filtered, meta = apply_date_range_filter(snapshot, cfg)
        self.assertEqual([m["id"] for m in filtered], ["2"])
        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["filtered_count"], 1)

    def test_disabled_filter_keeps_all_rows(self):
        snapshot = [
            {"id": "1", "createdAt": "2025-01-01T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
            {"id": "2", "createdAt": "2025-06-01T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
        ]
        cfg = {"validation": {"date_range": {"enabled": False, "field": "createdAt", "start": None, "end": None}}}
        filtered, meta = apply_date_range_filter(snapshot, cfg)
        self.assertEqual(len(filtered), 2)
        self.assertFalse(meta["enabled"])


if __name__ == "__main__":
    unittest.main()
