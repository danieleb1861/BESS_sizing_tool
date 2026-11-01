import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
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
            ramps = np.diff(pg)
            if (np.any(ramps > self.dg.ramp_up_kw_per_min * dt[1:])
                    or np.any(ramps < self.dg.ramp_dn_kw_per_min * dt[1:])):
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

    def test_ramps_limit_power_per_unit_across_commitment_changes(self):
        # MATLAB reference, select_best_parent.m: diff = pdg - prev_states.pdg.
        self.dg.pmin_frac = .6
        self.dg.ramp_up_kw_per_min = 1.
        self.dg.ramp_dn_kw_per_min = -1.
        self.bes.pmax_kw = 0.
        result = run_dp([80., 160.], [0., 1.], self.dg, self.bes, self.cfg)
        np.testing.assert_array_equal(result.n_active_opt, [1, 2])
        result = run_dp([160., 80.], [0., 1.], self.dg, self.bes, self.cfg)
        np.testing.assert_array_equal(result.n_active_opt, [2, 1])
        with self.assertRaisesRegex(RuntimeError, "No feasible"):
            run_dp([80., 100.], [0., 1.], self.dg, self.bes, self.cfg)

    def lossy_case(self):
        # Grid units cannot cancel with eta_c * eta_d = 0.9118, so grid-only closure means idle.
        self.bes = BESSpec(40., .5, eta_c=.97, eta_d=.94)
        self.cfg = DPConfig(2, 5, -1., 1.)
        self.dg = DGSpec(100., pmax_frac=1., p_grid_kw=np.array([0., 50., 100.]),
                         sfoc_g_per_kwh=np.array([300., 300., 200.]),
                         ramp_up_kw_per_min=6000., ramp_dn_kw_per_min=-6000.)
        return np.array([60., 100., 40., 90., 60., 80.]), np.arange(6) / 60.

    def test_lossy_battery_closes_exactly_and_saves_fuel(self):
        load, time = self.lossy_case()
        result = run_dp(load, time, self.dg, self.bes, self.cfg)
        idle = run_dp(load, time, self.dg, BESSpec(0., .5), self.cfg)
        self.assertGreater(np.abs(result.p_bess_opt_kw).max(), 0.)
        self.assertLess(result.cost_kg, idle.cost_kg)
        self.assertEqual(result.soc_opt[-1], result.soc_opt[0])

    def test_trajectory_obeys_the_soc_dynamics_and_the_power_balance(self):
        load, time = self.lossy_case()
        result = run_dp(load, time, self.dg, self.bes, self.cfg)
        dt = np.diff(time, prepend=time[0])
        power = result.p_bess_opt_kw
        beta = np.where(power > 0, 1 / self.bes.eta_d, self.bes.eta_c)
        np.testing.assert_allclose(.5 - np.cumsum(beta * power * dt / 60 / self.bes.e_kwh), result.soc_opt, atol=1e-12)
        np.testing.assert_allclose(result.n_active_opt * result.p_dg_opt_kw + power, load)
        fuel = result.n_active_opt * result.p_dg_opt_kw * dt / 60000. * sfoc_interp_g_per_kwh(
            result.p_dg_opt_kw, self.dg.p_grid_kw, self.dg.sfoc_g_per_kwh)
        self.assertAlmostEqual(result.cost_kg, fuel.sum(), places=12)

    def test_never_worse_than_an_idle_battery(self):
        rng = np.random.default_rng(1861)
        self.bes = BESSpec(40., 2., eta_c=.97, eta_d=.94)
        self.cfg = DPConfig(3, 21, -1., 1.)
        for _ in range(5):
            load = rng.uniform(60., 180., 40)
            time = np.arange(40) / 60.
            result = run_dp(load, time, self.dg, self.bes, self.cfg)
            idle = run_dp(load, time, self.dg, BESSpec(0., 2.), self.cfg)
            self.assertLessEqual(result.cost_kg, idle.cost_kg + 1e-12)

    def test_cost_is_close_to_exhaustive_search_with_closure_power(self):
        load, time = self.lossy_case()
        dt = np.diff(time, prepend=time[0]) / 60.
        best = np.inf
        for actions in product((-40., -20., 0., 20., 40.), repeat=len(load) - 2):
            soc, cost = .5, 0.
            for j, power in enumerate((*actions, None), start=1):
                if power is None:
                    need = .5 - soc
                    power = -need * .5 / dt[j] * (self.bes.eta_d if need < 0 else 1 / self.bes.eta_c)
                    if abs(power) > 40.:
                        cost = np.inf
                        break
                soc -= power * (1 / self.bes.eta_d if power > 0 else self.bes.eta_c) * dt[j] / .5
                options = [n * (load[j] - power) / n * float(sfoc_interp_g_per_kwh(
                               np.array([(load[j] - power) / n]), self.dg.p_grid_kw, self.dg.sfoc_g_per_kwh)[0])
                           for n in (1, 2) if 20. <= (load[j] - power) / n <= 100. or load[j] == power]
                if not options or not .2 <= soc <= .8:
                    cost = np.inf
                    break
                cost += min(options) * dt[j] / 1000.
            best = min(best, cost)
        result = run_dp(load, time, self.dg, self.bes, self.cfg)
        self.assertGreaterEqual(result.cost_kg, best - 1e-12)
        self.assertLess(result.cost_kg, best * 1.001)

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


class CLIRegressionTests(unittest.TestCase):
    def test_full_sweep_keeps_pareto_per_profile(self):
        import h5py
        import pandas as pd
        from bess_nsps import main_opt

        with tempfile.TemporaryDirectory(prefix="bess_cli_test_") as folder:
            root = Path(folder)
            profiles = []
            for name, load in [("a", 100), ("b", 150)]:
                path = root / (name + ".csv")
                path.write_text("date,time,pow_kw\n" + "".join(
                    f"2024-01-01,00:0{i}:00,{load}\n" for i in range(3)))
                profiles.append(str(path))
            output = root / "out" / "results.csv"
            argv = ["main_opt", "--profiles", *profiles, "--dg-pmax", "200", "--ndg", "2",
                    "--bess-pmin", "100", "--bess-pmax", "100", "--bess-pesteps", "1",
                    "--bess-emin", "100", "--bess-emax", "101", "--bess-esteps", "2",
                    "--nbatt-steps", "3", "--save", str(output), "--plots", "--save-traces"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main_opt.main()
            result = pd.read_csv(output)
            self.assertEqual(len(result), 4)
            for _, group in result.groupby("profile"):
                self.assertEqual(group.loc[group.pareto, "bess_e_kwh"].tolist(), [101.])
            self.assertEqual(len(list(output.parent.glob("plots/*/*.png"))), 3)
            dg = main_opt.build_dg(200.)
            with h5py.File(output.parent / "traces.h5", "r") as traces:
                for name in ("a", "b"):
                    trace = traces[name]["p100_e101"]
                    power = trace["p_dg_kw"][:]
                    dt = np.diff(trace["t_min"][:], prepend=0.)
                    cost = np.sum(power * trace["n_active"][:] * dt
                                  * sfoc_interp_g_per_kwh(power, dg.p_grid_kw, dg.sfoc_g_per_kwh) / 60000.)
                    self.assertAlmostEqual(trace.attrs["fuel_kg"], cost)


if __name__ == "__main__":
    unittest.main()
