import csv
import tempfile
import unittest
from pathlib import Path

from scripts.python.backtest_runner import summarize_run_index


class TestRunIndexTrends(unittest.TestCase):
    def _write_rows(self, rows):
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "backtest_run_index.csv"
        fieldnames = [
            "timestamp_utc",
            "run_json",
            "summary_md",
            "leaderboard_csv",
            "mode_compare_csv",
            "readiness_score",
            "readiness_tier",
            "decision",
            "guardrail_blocked",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        return td, path

    def test_empty_index_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            payload = summarize_run_index(Path(d) / "missing.csv")
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["alert_level"], "none")
        self.assertEqual(payload["blocked_streak"], 0)

    def test_degrading_critical_when_blocked_streak_and_delta(self):
        rows = [
            {"timestamp_utc": "t1", "readiness_score": "70", "decision": "NO_GO", "guardrail_blocked": "false"},
            {"timestamp_utc": "t2", "readiness_score": "68", "decision": "NO_GO", "guardrail_blocked": "true"},
            {"timestamp_utc": "t3", "readiness_score": "60", "decision": "NO_GO", "guardrail_blocked": "true"},
            {"timestamp_utc": "t4", "readiness_score": "50", "decision": "NO_GO", "guardrail_blocked": "true"},
            {"timestamp_utc": "t5", "readiness_score": "45", "decision": "NO_GO", "guardrail_blocked": "true"},
        ]
        td, path = self._write_rows(rows)
        try:
            payload = summarize_run_index(path)
        finally:
            td.cleanup()

        self.assertEqual(payload["trend_signal"], "degrading")
        self.assertEqual(payload["blocked_streak"], 4)
        self.assertEqual(payload["alert_level"], "critical")
        self.assertEqual(payload["decision_streak"], {"decision": "NO_GO", "length": 5})


if __name__ == "__main__":
    unittest.main()
