import sys
import tempfile
import unittest
from pathlib import Path
from itertools import product
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bess_nsps.kpi import compute_kpis
from bess_nsps.models import BESSpec, DGSpec, sfoc_interp_g_per_kwh
from bess_nsps.dp import DPConfig, run_dp
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


class DPRegressionTests(unittest.TestCase):
    def setUp(self):
        self.dg = DGSpec(100., pmax_frac=1., p_grid_kw=np.array([0., 100.]),
                         ramp_up_kw_per_min=6000., ramp_dn_kw_per_min=-6000.,
                         sfoc_g_per_kwh=np.array([300., 200.]))
        self.bes = BESSpec(40., 1000., eta_c=1., eta_d=1.)
        self.cfg = DPConfig(2, 3, -1., 1.)

    def brute_force(self, load, time):
        # Independent enumeration of complete schedules, including per-unit ramps.
        dt = np.diff(time, prepend=time[0])
        best = np.inf
        actions = list(product((1, 2), (-40., 0., 40.)))
        for schedule in product(actions, repeat=len(load)):
            n, pb = map(np.asarray, zip(*schedule))
            if pb[0] != 0:
                continue
            pg = (load - pb) / n
            if not np.all(((pg >= 20.) & (pg <= 100.)) | (pg == 0)):
                continue
            units = np.where(np.arange(2)[None, :] < n[:, None], pg[:, None], 0.)
            ramps = np.diff(units, axis=0)
            if (np.any(ramps > self.dg.ramp_up_kw_per_min * dt[1:, None])
                    or np.any(ramps < self.dg.ramp_dn_kw_per_min * dt[1:, None])):
                continue
            ds = np.where(pb >= 0, pb / self.bes.eta_d, pb * self.bes.eta_c) * dt / 60 / self.bes.e_kwh
            soc = 0.5 - np.cumsum(ds)
            if np.any(soc < .2) or np.any(soc > .8) or abs(soc[-1] - .5) > 1e-12:
                continue
            cost = np.sum(n * pg * (300. - pg) * dt / 60000.)
            best = min(best, cost)
        return best

    def test_cost_matches_exhaustive_search(self):
        for eta_c, eta_d in [(1., 1.), (.8, .625)]:
            self.bes.eta_c, self.bes.eta_d = eta_c, eta_d
            for load in ([120., 80., 160., 80.], [80., 160., 40., 120.]):
                with self.subTest(load=load, efficiencies=(eta_c, eta_d)):
                    load = np.asarray(load)
                    time = np.arange(4) / 60.
                    result = run_dp(load, time, self.dg, self.bes, self.cfg)
                    self.assertAlmostEqual(result.cost_kg, self.brute_force(load, time), places=12)
                    np.testing.assert_allclose(result.n_active_opt * result.p_dg_opt_kw + result.p_bess_opt_kw, load)
                    self.assertEqual(result.p_bess_opt_kw[0], 0.)
                    self.assertAlmostEqual(result.soc_opt[-1], result.soc_opt[0], places=12)

    def test_terminal_tolerance_cannot_supply_free_energy(self):
        result = run_dp([80., 80.], [0., 1 / 60.], self.dg, self.bes, self.cfg)
        np.testing.assert_array_equal(result.p_bess_opt_kw, [0., 0.])

    def test_resource_limits_raise_instead_of_pruning(self):
        for name, cap, error in [("STATES_CAP", 1, MemoryError),
                                 ("TOTAL_STATES_CAP", 1, MemoryError),
                                 ("TRANSITIONS_CAP", 1, RuntimeError)]:
            with self.subTest(limit=name), patch("bess_nsps.dp." + name, cap):
                with self.assertRaisesRegex(error, "resource|exceeds"):
                    run_dp([120., 80., 160., 80.], np.arange(4), self.dg, self.bes, self.cfg)

    def test_start_and_stop_ramps_are_enforced(self):
        self.dg.pmin_frac = .6
        self.dg.ramp_up_kw_per_min = 1.
        self.dg.ramp_dn_kw_per_min = -1.
        self.bes.pmax_kw = 0.
        for load in ([80., 160.], [160., 80.]):
            with self.subTest(load=load), self.assertRaisesRegex(RuntimeError, "No feasible"):
                run_dp(load, [0., 1.], self.dg, self.bes, self.cfg)

    def test_invalid_inputs_fail_before_solving(self):
        for time in ([0., 0.], [1., 0.], [0., np.nan]):
            with self.subTest(time=time), self.assertRaises(ValueError):
                run_dp([80., 80.], time, self.dg, self.bes, self.cfg)
        self.cfg.alpha_max = 2.
        with self.assertRaises(ValueError):
            run_dp([80., 80.], [0., 1.], self.dg, self.bes, self.cfg)


class ModelRegressionTests(unittest.TestCase):
    def test_pchip_preserves_knots_and_shape(self):
        # PCHIP for (0,0), (1,1), (2,0) has slopes 2,0,-2.
        actual = sfoc_interp_g_per_kwh(np.array([0., .5, 1., 1.5, 2.]),
                                      np.array([0., 1., 2.]), np.array([0., 1., 0.]))
        np.testing.assert_allclose(actual, [0., .75, 1., .75, 0.])

    def test_default_ramps_match_matlab_values(self):
        dg = DGSpec(2200.)
        self.assertAlmostEqual(dg.ramp_up_kw_per_min, 2200. / .33)
        self.assertAlmostEqual(dg.ramp_dn_kw_per_min, -2200. / .2)

    def test_invalid_fuel_curve_is_rejected(self):
        with self.assertRaises(ValueError):
            sfoc_interp_g_per_kwh(np.array([50.]), np.array([0., 100.]), np.array([200., np.nan]))


if __name__ == "__main__":
    unittest.main()
