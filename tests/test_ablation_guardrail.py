import json
import tempfile
import unittest
from pathlib import Path

from scripts.python.ablation_runner import maybe_apply_fallback


class TestAblationGuardrail(unittest.TestCase):
    def test_triggers_fallback_on_tiny_window(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            snap = d / "snap.json"
            snap.write_text(json.dumps([
                {"id": "1", "updatedAt": "2026-03-01T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
                {"id": "2", "updatedAt": "2026-03-02T00:00:00Z", "outcomePrices": [0.5, 0.5], "spread": 0.01},
            ]))

            cfg = {
                "snapshot_path": str(snap),
                "validation": {
                    "date_range": {"enabled": True, "field": "updatedAt", "start": "2026-03-01T00:00:00Z", "end": "2026-03-01T00:10:00Z"},
                    "ablation_guardrails": {
                        "enabled": True,
                        "min_markets": 2,
                        "min_span_hours": 24,
                        "fallback_date_range": {"field": "updatedAt", "start": "2026-03-01T00:00:00Z", "end": "2026-12-31T23:59:59Z"},
                    },
                },
            }

            # emulate tiny primary meta
            primary_meta = {
                "enabled": True,
                "applied_min": "2026-03-01T00:00:00+00:00",
                "applied_max": "2026-03-01T00:10:00+00:00",
                "filtered_count": 1,
                "input_count": 2,
                "start": "2026-03-01T00:00:00Z",
                "end": "2026-03-01T00:10:00Z",
                "field": "updatedAt",
            }

            c2, s2, m2, g = maybe_apply_fallback(cfg, [], primary_meta)
            self.assertTrue(g.get("triggered"))
            self.assertGreaterEqual(int(m2.get("filtered_count", 0)), 1)


if __name__ == "__main__":
    unittest.main()
