import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bess_nsps.kpi import compute_kpis
from bess_nsps.models import BESSpec
from bess_nsps.data import get_mat_window, load_profile
from bess_nsps.window import slice_by_indices


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


class MATRegressionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="bess_test_")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.mat"

    def test_missing_time_uses_declared_interval(self):
        savemat(self.path, {"pow_load": [100., 200., 150.]})
        with self.assertWarnsRegex(UserWarning, "0.25 min/sample"):
            load, time = load_profile(str(self.path), step_min=0.25)
        np.testing.assert_array_equal(load, [100., 200., 150.])
        np.testing.assert_array_equal(time, [0., 0.25, 0.5])

    def test_malformed_time_is_rejected(self):
        for time in ([0., 1.], [0., 2., 1.], [0., 0., 1.], [0., 1., np.nan]):
            with self.subTest(time=time):
                savemat(self.path, {"pow_load": [100., 200., 150.], "t_plot": time})
                with self.assertRaises(ValueError):
                    load_profile(str(self.path))

    def test_unknown_load_does_not_select_time(self):
        savemat(self.path, {"t_plot": np.arange(12), "unknown_load": np.full(12, 100.)})
        load, time = load_profile(str(self.path))
        np.testing.assert_array_equal(load, np.full(12, 100.))
        savemat(self.path, {"t_plot": np.arange(12)})
        with self.assertRaises(KeyError):
            load_profile(str(self.path))

    def test_ambiguous_load_is_rejected(self):
        savemat(self.path, {"a": np.arange(12), "b": np.arange(12), "t_plot": np.arange(12)})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_profile(str(self.path))

    def test_nested_matlab_struct(self):
        savemat(self.path, {"profile": {"pow_load": [100., 200., 150.], "t_plot": [0., 1., 2.]}})
        load, time = load_profile(str(self.path))
        np.testing.assert_array_equal(load, [100., 200., 150.])
        np.testing.assert_array_equal(time, [0., 1., 2.])

    def test_matlab_window_includes_final_sample(self):
        savemat(self.path, {"i0": 2, "i1": 3})
        window = get_mat_window(str(self.path))
        load, time = slice_by_indices([10, 20, 30, 40], [0, 1, 2, 3],
                                      window["i0"], window["i1"])
        np.testing.assert_array_equal(load, [20, 30])


if __name__ == "__main__":
    unittest.main()
