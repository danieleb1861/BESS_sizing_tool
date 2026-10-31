import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bess_nsps.kpi import compute_kpis
from bess_nsps.models import BESSpec


class KPIRegressionTests(unittest.TestCase):
    def setUp(self):
        self.bes = BESSpec(100.0, 100.0)

    def evaluate(self, soc, power=None, dt=None):
        soc = np.asarray(soc, float)
        if power is None:
            delta = np.diff(soc, prepend=soc[..., :1], axis=-1)
            power = np.where(delta < 0, -delta * 100 * self.bes.eta_d,
                             -delta * 100 / self.bes.eta_c)
        if dt is None:
            dt = np.full(soc.shape, 60.0)
            dt[..., 0] = 0.0
        return compute_kpis(power, dt, soc, self.bes, batt_pmax_kw=100.0)

    def test_repeated_cycles_double_efc_not_mean_depth(self):
        one = self.evaluate([0.5, 0.2, 0.5])
        two = self.evaluate([0.5, 0.2, 0.5, 0.2, 0.5])
        self.assertAlmostEqual(one.dod_mean, 0.3)
        self.assertAlmostEqual(two.dod_mean, 0.3)
        self.assertAlmostEqual(one.efc_per_year, 17.5)
        self.assertAlmostEqual(two.efc_per_year, 35.0)
        self.assertAlmostEqual(two.energy_throughput_kwh, 2 * one.energy_throughput_kwh)

    def test_plateaus_and_partial_discharge(self):
        result = self.evaluate([0.7, 0.6, 0.6, 0.4, 0.6, 0.5])
        self.assertAlmostEqual(result.dod_mean, 0.2)
        self.assertAlmostEqual(result.efc_per_year, 35 * 0.4 / 0.6)

    def test_rows_are_independent_manoeuvres(self):
        result = self.evaluate([[0.5, 0.2, 0.5], [0.8, 0.5, 0.8]])
        self.assertAlmostEqual(result.dod_mean, 0.3)
        self.assertAlmostEqual(result.efc_per_year, 17.5)

    def test_idle_and_time_weighted_c_rate(self):
        idle = self.evaluate([0.5, 0.5, 0.5])
        self.assertEqual(idle.dod_mean, 0.0)
        self.assertEqual(idle.efc_per_year, 0.0)
        result = self.evaluate([0.5, 0.5, 0.5], [100, 10, 30], [0, 1, 3])
        self.assertAlmostEqual(result.c_rate_mean_per_h, 0.25)


if __name__ == "__main__":
    unittest.main()
