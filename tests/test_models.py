from __future__ import annotations

import unittest

import numpy as np

from surface_impedance.models import (
    _rough_stack_transitions_from_layers,
    compute_profile,
    compute_surface_impedance,
    waveimpedance,
    zs_next,
)


class TestModels(unittest.TestCase):
    def test_normal_skin_returns_positive_real_and_imaginary_parts(self) -> None:
        freq = np.array([1e6, 1e7, 1e8])
        zs = compute_surface_impedance("normal-skin", freq, {"sigma": 5.8e7, "mu_r": 1.0})
        self.assertEqual(zs.shape, freq.shape)
        self.assertTrue(np.all(zs.real > 0.0))
        self.assertTrue(np.all(zs.imag > 0.0))

    def test_half_space_accepts_zero_tau(self) -> None:
        freq = np.array([1e6, 1e8, 1e10])
        zs = compute_surface_impedance(
            "half-space",
            freq,
            {
                "sigma": 5.8e7,
                "tau": 0.0,
                "epsr_real": 1.0,
                "epsr_imag": 0.0,
                "mur_real": 1.0,
                "mur_imag": 0.0,
            },
        )
        self.assertTrue(np.all(np.isfinite(zs.real)))
        self.assertTrue(np.all(np.isfinite(zs.imag)))

    def test_multi_layer_returns_finite_values(self) -> None:
        freq = np.array([1e6, 1e8, 1e10])
        zs = compute_surface_impedance(
            "multi-layer",
            freq,
            {
                "base_layer": {
                    "epsilon": (1.0, 0.0),
                    "mu": (1.0, 0.0),
                    "sigma": 5.8e7,
                    "tau": 0.0,
                },
                "layers": [
                    {
                        "epsilon": (1.0, 0.0),
                        "mu": (1.0, 0.0),
                        "sigma": 5.8e7,
                        "tau": 0.0,
                        "thickness": 1e-6,
                    }
                ],
            },
        )
        self.assertTrue(np.all(np.isfinite(zs.real)))
        self.assertTrue(np.all(np.isfinite(zs.imag)))

    def test_multi_layer_uses_last_listed_layer_as_air_facing(self) -> None:
        freq = np.array([1e9])
        base_layer = {
            "epsilon": (1.0, 0.0),
            "mu": (1.0, 0.0),
            "sigma": 2.0e6,
            "tau": 0.0,
        }
        layer_1 = {
            "epsilon": (1.0, 0.0),
            "mu": (1.0, 0.0),
            "sigma": 5.0e6,
            "tau": 0.0,
            "thickness": 2e-6,
        }
        layer_2 = {
            "epsilon": (1.0, 0.0),
            "mu": (1.0, 0.0),
            "sigma": 9.0e6,
            "tau": 0.0,
            "thickness": 1e-6,
        }
        zs = compute_surface_impedance(
            "multi-layer",
            freq,
            {"base_layer": base_layer, "layers": [layer_1, layer_2]},
        )
        z_start = waveimpedance(freq, base_layer["epsilon"], base_layer["mu"], base_layer["sigma"], 0.0)
        expected = zs_next(
            zs_next(
                z_start,
                freq,
                layer_2["epsilon"],
                layer_2["mu"],
                layer_2["sigma"],
                0.0,
                layer_2["thickness"],
            ),
            freq,
            layer_1["epsilon"],
            layer_1["mu"],
            layer_1["sigma"],
            0.0,
            layer_1["thickness"],
        )
        np.testing.assert_allclose(zs, expected)

    def test_rough_single_returns_finite_values(self) -> None:
        freq = np.array([1e6, 1e7])
        zs = compute_surface_impedance(
            "rough-single",
            freq,
            {
                "sigma_metal": 5.8e7,
                "rq": 0.5e-6,
                "mu_r": 1.0,
                "step_size": 10e-9,
                "xmin_factor": -5.0,
                "domain_factor": 50.0,
            },
        )
        self.assertTrue(np.all(np.isfinite(zs.real)))
        self.assertTrue(np.all(np.isfinite(zs.imag)))

    def test_rough_multi_returns_finite_values(self) -> None:
        freq = np.array([1e6, 1e7])
        zs = compute_surface_impedance(
            "rough-multi",
            freq,
            {
                "sigma1": 5.8e7,
                "sigma2": 4.2e7,
                "rq01": 0.4e-6,
                "rq12": 0.2e-6,
                "t1": 1.5e-6,
                "mu_r": 1.0,
                "step_size": 10e-9,
                "xmin_factor": -5.0,
                "domain_factor": 50.0,
            },
        )
        self.assertTrue(np.all(np.isfinite(zs.real)))
        self.assertTrue(np.all(np.isfinite(zs.imag)))

    def test_rough_multi_json_stack_returns_finite_values(self) -> None:
        freq = np.array([1e6, 1e7])
        zs = compute_surface_impedance(
            "rough-multi",
            freq,
            {
                "base_layer": {
                    "epsilon": (1.0, 0.0),
                    "mu": (1.0, 0.0),
                    "sigma": 1.35e6,
                    "tau": 0.0,
                    "rq": 0.08e-6,
                },
                "layers": [
                    {
                        "epsilon": (1.0, 0.0),
                        "mu": (1.0, 0.0),
                        "sigma": 5.8e7,
                        "tau": 0.0,
                        "thickness": 75e-6,
                        "rq": 0.4e-6,
                    },
                    {
                        "epsilon": (1.0, 0.0),
                        "mu": (1.0, 0.0),
                        "sigma": 2.4e6,
                        "tau": 0.0,
                        "thickness": 50e-9,
                        "rq": 0.15e-6,
                    },
                ],
                "step_size": 10e-9,
                "xmin_factor": -5.0,
                "domain_factor": 50.0,
            },
        )
        self.assertTrue(np.all(np.isfinite(zs.real)))
        self.assertTrue(np.all(np.isfinite(zs.imag)))

    def test_rough_multi_uses_last_listed_layer_as_air_facing(self) -> None:
        transitions = _rough_stack_transitions_from_layers(
            {
                "epsilon": (1.0, 0.0),
                "mu": (1.0, 0.0),
                "sigma": 1.0e6,
                "tau": 0.0,
                "rq": 30e-9,
            },
            [
                {
                    "epsilon": (1.0, 0.0),
                    "mu": (1.0, 0.0),
                    "sigma": 2.0e6,
                    "tau": 0.0,
                    "thickness": 200e-9,
                    "rq": 20e-9,
                },
                {
                    "epsilon": (1.0, 0.0),
                    "mu": (1.0, 0.0),
                    "sigma": 3.0e6,
                    "tau": 0.0,
                    "thickness": 100e-9,
                    "rq": 10e-9,
                },
            ],
        )
        self.assertEqual(transitions[0]["sigma_next"], 3.0e6)
        self.assertEqual(transitions[0]["rq"], 10e-9)
        self.assertEqual(transitions[1]["sigma_next"], 2.0e6)
        self.assertEqual(transitions[1]["position"], 100e-9)
        self.assertEqual(transitions[2]["sigma_next"], 1.0e6)
        self.assertEqual(transitions[2]["position"], 300e-9)

    def test_compute_profile_for_rough_single_returns_normalized_arrays(self) -> None:
        profile = compute_profile(
            "rough-single",
            1e9,
            {
                "sigma_metal": 5.8e7,
                "rq": 0.5e-6,
                "mu_r": 1.0,
                "step_size": 10e-9,
                "xmin_factor": -5.0,
                "domain_factor": 10.0,
            },
        )
        self.assertGreater(len(profile.x_m), 10)
        self.assertEqual(profile.x_m.shape, profile.normalized_conductivity.shape)
        self.assertTrue(np.all(profile.normalized_conductivity >= 0.0))
        self.assertTrue(np.all(profile.normalized_magnetic_field >= 0.0))
        self.assertTrue(np.all(profile.normalized_power_loss_density >= 0.0))

    def test_compute_profile_for_rough_multi_returns_normalized_arrays(self) -> None:
        profile = compute_profile(
            "rough-multi",
            1e9,
            {
                "sigma1": 5.8e7,
                "sigma2": 4.2e7,
                "rq01": 0.4e-6,
                "rq12": 0.2e-6,
                "t1": 1.5e-6,
                "mu_r": 1.0,
                "step_size": 10e-9,
                "xmin_factor": -5.0,
                "domain_factor": 10.0,
            },
        )
        self.assertGreater(len(profile.x_m), 10)
        self.assertEqual(profile.x_m.shape, profile.normalized_conductivity.shape)
        self.assertTrue(np.all(profile.normalized_conductivity >= 0.0))
        self.assertTrue(np.all(profile.normalized_magnetic_field >= 0.0))
        self.assertTrue(np.all(profile.normalized_power_loss_density >= 0.0))

    def test_compute_profile_for_rough_multi_json_stack_returns_normalized_arrays(self) -> None:
        profile = compute_profile(
            "rough-multi",
            1e9,
            {
                "base_layer": {
                    "epsilon": (1.0, 0.0),
                    "mu": (1.0, 0.0),
                    "sigma": 1.35e6,
                    "tau": 0.0,
                    "rq": 0.08e-6,
                },
                "layers": [
                    {
                        "epsilon": (1.0, 0.0),
                        "mu": (1.0, 0.0),
                        "sigma": 5.8e7,
                        "tau": 0.0,
                        "thickness": 75e-6,
                        "rq": 0.4e-6,
                    },
                    {
                        "epsilon": (1.0, 0.0),
                        "mu": (1.0, 0.0),
                        "sigma": 2.4e6,
                        "tau": 0.0,
                        "thickness": 50e-9,
                        "rq": 0.15e-6,
                    },
                ],
                "step_size": 10e-9,
                "xmin_factor": -5.0,
                "domain_factor": 10.0,
            },
        )
        self.assertGreater(len(profile.x_m), 10)
        self.assertEqual(profile.x_m.shape, profile.normalized_conductivity.shape)
        self.assertTrue(np.all(profile.normalized_conductivity >= 0.0))
        self.assertTrue(np.all(profile.normalized_magnetic_field >= 0.0))
        self.assertTrue(np.all(profile.normalized_power_loss_density >= 0.0))

    def test_compute_profile_for_half_space_returns_finite_arrays(self) -> None:
        profile = compute_profile(
            "half-space",
            1e9,
            {
                "sigma": 5.8e7,
                "tau": 0.0,
                "epsr_real": 1.0,
                "epsr_imag": 0.0,
                "mur_real": 1.0,
                "mur_imag": 0.0,
            },
        )
        self.assertTrue(np.all(np.isfinite(profile.normalized_conductivity)))
        self.assertTrue(np.all(np.isfinite(profile.normalized_magnetic_field)))
        self.assertTrue(np.all(np.isfinite(profile.normalized_power_loss_density)))

    def test_compute_profile_for_multi_layer_returns_finite_arrays(self) -> None:
        profile = compute_profile(
            "multi-layer",
            1e9,
            {
                "base_layer": {
                    "epsilon": (1.0, 0.0),
                    "mu": (1.0, 0.0),
                    "sigma": 1.4e6,
                    "tau": 0.0,
                },
                "layers": [
                    {
                        "epsilon": (1.0, 0.0),
                        "mu": (1.0, 0.0),
                        "sigma": 5.8e7,
                        "tau": 0.0,
                        "thickness": 75e-6,
                    }
                ],
            },
        )
        self.assertGreater(len(profile.x_m), 10)
        self.assertLess(float(np.min(profile.x_m)), 0.0)
        self.assertGreater(float(np.max(profile.x_m)), 75e-6)
        self.assertEqual(profile.x_m.shape, profile.normalized_conductivity.shape)
        self.assertTrue(np.all(np.isfinite(profile.normalized_conductivity)))
        self.assertTrue(np.all(np.isfinite(profile.normalized_magnetic_field)))
        self.assertTrue(np.all(np.isfinite(profile.normalized_power_loss_density)))
        self.assertGreater(float(np.max(profile.normalized_conductivity)), 0.0)

    def test_half_space_returns_finite_values(self) -> None:
        freq = np.array([1e6, 1e8, 1e10])
        zs = compute_surface_impedance(
            "half-space",
            freq,
            {
                "sigma": 5.8e7,
                "tau": 2.5e-14,
                "epsr_real": 1.0,
                "epsr_imag": 0.0,
                "mur_real": 1.0,
                "mur_imag": 0.0,
            },
        )
        self.assertTrue(np.all(np.isfinite(zs.real)))
        self.assertTrue(np.all(np.isfinite(zs.imag)))


if __name__ == "__main__":
    unittest.main()
