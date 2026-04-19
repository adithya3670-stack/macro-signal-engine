import json
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.minor_3d_signal_manager import Minor3DSignalManager


class TestMinor3DSignalManager(unittest.TestCase):
    def _write_combo_results(self, folder: str) -> str:
        rows = [
            {"combo": "vote_2of3", "period": "2019-2021", "alerts": 1, "tp": 1, "fp": 0, "covered_events": 1, "total_events": 4},
            {"combo": "vote_2of3", "period": "2024-2025", "alerts": 3, "tp": 1, "fp": 2, "covered_events": 1, "total_events": 3},
            {"combo": "or_any", "period": "2019-2021", "alerts": 5, "tp": 1, "fp": 4, "covered_events": 1, "total_events": 4},
            {"combo": "or_any", "period": "2024-2025", "alerts": 9, "tp": 1, "fp": 8, "covered_events": 1, "total_events": 3},
        ]
        path = os.path.join(folder, "minor_3d_rule_combo_results.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_latest(self, folder: str) -> str:
        payload = {
            "as_of_date": "2026-03-06",
            "signals": {
                "top_rules": [{"rule": "r1", "latest_alert": True}],
                "combo_latest": {
                    "vote_2of3": {"latest_alert": True},
                    "or_any": {"latest_alert": True},
                    "all_3": {"latest_alert": False},
                },
            },
        }
        path = os.path.join(folder, "minor_3d_latest_signals.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def test_policy_enables_actionable_when_precision_above_floor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_combo_results(tmpdir)
            mgr = Minor3DSignalManager(
                research_dir=tmpdir,
                features_file="unused.csv",
                events_file="unused.csv",
                script_path="unused.py",
            )
            policy = mgr._compute_policy(precision_floor=0.2)
            self.assertTrue(policy["actionable"]["enabled"])
            self.assertEqual(policy["actionable"]["combo"], "vote_2of3")
            self.assertEqual(policy["actionable"]["reason_code"], "USE_ACTIONABLE")

    def test_policy_disables_actionable_when_precision_below_floor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_combo_results(tmpdir)
            mgr = Minor3DSignalManager(
                research_dir=tmpdir,
                features_file="unused.csv",
                events_file="unused.csv",
                script_path="unused.py",
            )
            policy = mgr._compute_policy(precision_floor=0.7)
            self.assertFalse(policy["actionable"]["enabled"])
            self.assertEqual(policy["actionable"]["reason_code"], "LOW_PRECISION")

    def test_status_maps_to_watchlist_when_actionable_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_combo_results(tmpdir)
            self._write_latest(tmpdir)
            mgr = Minor3DSignalManager(
                research_dir=tmpdir,
                features_file="unused.csv",
                events_file="unused.csv",
                script_path="unused.py",
            )
            status = mgr.load_latest_status(auto_refresh=False, precision_floor=0.7)
            self.assertEqual(status["state"], "WATCHLIST")
            self.assertFalse(status["actionable"]["enabled"])
            self.assertTrue(status["watchlist"]["raw_signal"])


if __name__ == "__main__":
    unittest.main()
