from __future__ import annotations

import unittest

import numpy as np

from surface_impedance.models import compute_profile, compute_surface_impedance


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
