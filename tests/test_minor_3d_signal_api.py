import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class TestMinor3DSignalApi(unittest.TestCase):
    def test_minor_3d_endpoints(self):
        app.testing = True
        client = app.test_client()

        mgr = MagicMock()
        mgr.load_latest_status.return_value = {"status": "success", "state": "WATCHLIST"}
        mgr.load_policy.return_value = {"status": "success", "policy": {"actionable": {"enabled": True}}}
        mgr.rebuild.return_value = {"status": "success", "policy": {"actionable": {"enabled": True}}}

        with patch("routes.analysis.Minor3DSignalManager", return_value=mgr):
            status_resp = client.get("/api/minor_3d/status_latest?auto_refresh=false")
            self.assertEqual(status_resp.status_code, 200)
            self.assertEqual(status_resp.get_json()["status"], "success")
            self.assertIn("state", status_resp.get_json())

            policy_resp = client.get("/api/minor_3d/policy_latest?auto_refresh=false")
            self.assertEqual(policy_resp.status_code, 200)
            self.assertEqual(policy_resp.get_json()["status"], "success")
            self.assertIn("policy", policy_resp.get_json())

            rebuild_resp = client.post("/api/minor_3d/rebuild", json={"precision_floor": 0.25})
            self.assertEqual(rebuild_resp.status_code, 200)
            self.assertEqual(rebuild_resp.get_json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
