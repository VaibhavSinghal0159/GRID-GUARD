import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from api import app
from electricity_theft_detector_v4 import run_theft_screening


class TestGridGuard(unittest.TestCase):
    def test_screening_is_deterministic_and_valid(self):
        r1 = run_theft_screening()
        r2 = run_theft_screening()
        self.assertEqual(len(r1["houses"]), 50)
        self.assertEqual(
            [h["risk_score"] for h in r1["houses"]],
            [h["risk_score"] for h in r2["houses"]],
        )
        self.assertTrue(r1["illustrative_synthetic_loss"])
        self.assertIn("not proof of theft", r1["disclaimer"].lower())

    def test_api_endpoints_and_safety(self):
        client = TestClient(app)
        res_houses = client.get("/api/houses")
        self.assertEqual(res_houses.status_code, 200)
        self.assertEqual(len(res_houses.json()["houses"]), 50)

        res_metrics = client.get("/api/model-metrics")
        self.assertEqual(res_metrics.status_code, 200)
        metrics = res_metrics.json()
        self.assertIn("pr_auc", metrics)
        self.assertIn("trivial_baseline", metrics)

        res_exp = client.get("/api/explanation?house_id=1")
        self.assertEqual(res_exp.status_code, 200)
        self.assertNotIn("fir", res_exp.text.lower())


if __name__ == "__main__":
    unittest.main()