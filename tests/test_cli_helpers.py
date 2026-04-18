from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from surface_impedance.cli import (
    _frequency_unit_factor,
    _impedance_unit_factor,
    build_parser,
    collect_cases,
    default_params_for_model,
    default_rough_multi_step_size,
    default_rough_stack_step_size,
    default_rough_single_step_size,
    make_frequency_grid,
    profile_layer_centers_um,
    ComparisonCase,
)
from surface_impedance.io_utils import build_output_rows, export_data


class TestCliHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path("test-artifacts")
        self.output_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for path in self.output_dir.glob("*"):
            path.unlink()
        self.output_dir.rmdir()

    def test_make_frequency_grid_is_log_spaced(self) -> None:
        grid = make_frequency_grid(1e3, 1e9, 4)
        expected = np.array([1e3, 1e5, 1e7, 1e9])
        np.testing.assert_allclose(grid, expected)

    def test_collect_cases_parses_multiple_comparisons(self) -> None:
        class Args:
            case = [
                "label=copper,model=normal-skin,sigma=5.8e7",
                "label=bulk,model=half-space,sigma=4.2e7,tau=0,epsr_real=1.0,epsr_imag=0.0,mur_real=1.0,mur_imag=0.0",
            ]
            model = "normal-skin"
            sigma = 5.8e7
            sigma_metal = 5.8e7
            mu_r = 1.0
            tau = 2.5e-14
            epsr_real = 1.0
            epsr_imag = 0.0
            mur_real = 1.0
            mur_imag = 0.0

        cases = collect_cases(Args())
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].label, "copper")
        self.assertEqual(cases[1].model, "half-space")
        self.assertEqual(cases[1].params["sigma"], 4.2e7)

    def test_rough_single_case_uses_its_own_rq_for_default_step_size(self) -> None:
        class Args:
            case = [
                "label=rough-a,model=rough-single,sigma_metal=5.8e7,rq=1e-12",
                "label=rough-b,model=rough-single,sigma_metal=5.8e7,rq=1e-15",
            ]
            model = "normal-skin"
            sigma = 5.8e7
            sigma_metal = 5.8e7
            mu_r = 1.0
            tau = 2.5e-14
            rq = 0.5e-6
            step_size = None
            xmin_factor = -5.0
            domain_factor = 10.0
            epsr_real = 1.0
            epsr_imag = 0.0
            mur_real = 1.0
            mur_imag = 0.0
            layers_file = None

        cases = collect_cases(Args())
        self.assertEqual(cases[0].params["step_size"], 1e-12 / 25.0)
        self.assertEqual(cases[1].params["step_size"], 1e-15 / 25.0)

    def test_parser_accepts_plot_scale(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--plot-scale", "semilogx"])
        self.assertEqual(args.plot_scale, "semilogx")

    def test_parser_accepts_frequency_unit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--frequency-unit", "GHz"])
        self.assertEqual(args.frequency_unit, "GHz")

    def test_parser_accepts_impedance_unit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--impedance-unit", "mOhm"])
        self.assertEqual(args.impedance_unit, "mOhm")

    def test_parser_accepts_profile_plot_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--profile-frequency",
                "1e9",
                "--profile-quantity",
                "conductivity",
                "--profile-quantity",
                "current-density",
                "--profile-quantity",
                "power-loss",
                "--profile-plot",
                "profile.png",
            ]
        )
        self.assertEqual(args.profile_frequency, 1e9)
        self.assertEqual(args.profile_quantity, ["conductivity", "current-density", "power-loss"])
        self.assertEqual(args.profile_plot, Path("profile.png"))

    def test_parser_accepts_profile_x_limits_in_um(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--profile-x-min-um",
                "-0.5",
                "--profile-x-max-um",
                "2.0",
            ]
        )
        self.assertEqual(args.profile_x_min_um, -0.5)
        self.assertEqual(args.profile_x_max_um, 2.0)

    def test_parser_accepts_profile_overlay_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--profile-overlay-cases",
                "--profile-normalize-to",
                "smooth",
            ]
        )
        self.assertTrue(args.profile_overlay_cases)
        self.assertEqual(args.profile_normalize_to, "smooth")

    def test_profile_layer_centers_use_air_facing_last_layer_convention(self) -> None:
        case = ComparisonCase(
            label="rough",
            model="rough-multi",
            params={
                "base_layer": {"sigma": 2.1e7, "rq": 60e-9},
                "layers": [
                    {"name": "Layer 1", "material": "Cu", "thickness": 0.75e-6, "sigma": 5.8e7, "rq": 60e-9},
                    {"name": "Layer 2", "material": "Ni", "thickness": 2.75e-6, "sigma": 1.8e6, "rq": 60e-9},
                    {"name": "Layer 3", "material": "Au", "thickness": 1.25e-6, "sigma": 8.6e6, "rq": 60e-9},
                ],
            },
        )
        centers = profile_layer_centers_um(case)
        self.assertEqual(len(centers), 3)
        self.assertEqual(centers[0][1], "Au")
        self.assertAlmostEqual(centers[0][0], 0.625)
        self.assertEqual(centers[1][1], "Ni")
        self.assertAlmostEqual(centers[1][0], 2.625)
        self.assertEqual(centers[2][1], "Cu")
        self.assertAlmostEqual(centers[2][0], 4.375)

    def test_profile_layer_centers_support_multi_layer_cases(self) -> None:
        case = ComparisonCase(
            label="smooth",
            model="multi-layer",
            params={
                "base_layer": {"sigma": 1.4e6},
                "layers": [
                    {"name": "Layer 1", "material": "Copper", "thickness": 75e-6, "sigma": 5.8e7},
                    {"name": "Layer 2", "material": "Nickel", "thickness": 2e-6, "sigma": 1.5e7},
                ],
            },
        )
        centers = profile_layer_centers_um(case)
        self.assertEqual(len(centers), 2)
        self.assertEqual(centers[0][1], "Nickel")
        self.assertAlmostEqual(centers[0][0], 1.0)
        self.assertEqual(centers[1][1], "Copper")
        self.assertAlmostEqual(centers[1][0], 39.5)

    def test_frequency_unit_factor_maps_units(self) -> None:
        self.assertEqual(_frequency_unit_factor("Hz"), 1.0)
        self.assertEqual(_frequency_unit_factor("MHz"), 1e6)

    def test_impedance_unit_factor_maps_units(self) -> None:
        self.assertEqual(_impedance_unit_factor("Ohm"), 1.0)
        self.assertEqual(_impedance_unit_factor("uOhm"), 1e-6)

    def test_default_rough_single_step_size_scales_with_rq(self) -> None:
        self.assertEqual(default_rough_single_step_size(1e-6), 4e-8)

    def test_default_rough_multi_step_size_uses_smaller_roughness(self) -> None:
        self.assertEqual(default_rough_multi_step_size(1e-6, 5e-7), 2e-8)

    def test_default_rough_stack_step_size_uses_smallest_json_roughness(self) -> None:
        self.assertAlmostEqual(
            default_rough_stack_step_size(
                {
                    "base_layer": {"rq": 2e-7},
                    "layers": [{"rq": 1e-6}, {"rq": 5e-7}],
                }
            ),
            8e-9,
        )

    def test_rough_single_defaults_use_requested_convention(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--model", "rough-single", "--rq", "1e-6"])
        params = default_params_for_model("rough-single", args)
        self.assertEqual(params["step_size"], 4e-8)
        self.assertEqual(params["xmin_factor"], -5.0)
        self.assertEqual(params["domain_factor"], 10.0)

    def test_rough_multi_defaults_use_requested_convention(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "rough-multi",
                "--rq01",
                "1e-6",
                "--rq12",
                "5e-7",
                "--t1",
                "2e-6",
            ]
        )
        params = default_params_for_model("rough-multi", args)
        self.assertEqual(params["step_size"], 2e-8)
        self.assertEqual(params["t1"], 2e-6)
        self.assertEqual(params["xmin_factor"], -5.0)
        self.assertEqual(params["domain_factor"], 10.0)

    def test_rough_multi_defaults_can_load_layers_file(self) -> None:
        target = self.output_dir / "rough-layers.json"
        target.write_text(
            json.dumps(
                {
                    "base_layer": {
                        "epsilon": [1.0, 0.0],
                        "mu": [1.0, 0.0],
                        "sigma": 1.35e6,
                        "tau": 0.0,
                        "rq": 0.1e-6,
                    },
                    "layers": [
                        {
                            "epsilon": [1.0, 0.0],
                            "mu": [1.0, 0.0],
                            "sigma": 5.8e7,
                            "tau": 0.0,
                            "thickness": 75e-6,
                            "rq": 0.5e-6,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        parser = build_parser()
        args = parser.parse_args(["--model", "rough-multi", "--layers-file", str(target)])
        params = default_params_for_model("rough-multi", args)
        self.assertIn("base_layer", params)
        self.assertIn("layers", params)
        self.assertEqual(params["step_size"], 0.1e-6 / 25.0)

    def test_rough_multi_case_uses_its_own_roughness_for_default_step_size(self) -> None:
        class Args:
            case = [
                "label=rough-multi-a,model=rough-multi,sigma1=5.8e7,sigma2=4.2e7,rq01=1e-12,rq12=2e-12,t1=5e-9",
            ]
            model = "normal-skin"
            sigma = 5.8e7
            sigma_metal = 5.8e7
            sigma1 = 5.8e7
            sigma2 = 4.2e7
            mu_r = 1.0
            tau = 2.5e-14
            rq = 0.5e-6
            rq01 = 0.5e-6
            rq12 = 0.5e-6
            t1 = 2e-6
            step_size = None
            xmin_factor = -5.0
            domain_factor = 10.0
            epsr_real = 1.0
            epsr_imag = 0.0
            mur_real = 1.0
            mur_imag = 0.0
            layers_file = None

        cases = collect_cases(Args())
        self.assertEqual(cases[0].params["step_size"], 1e-12 / 25.0)

    def test_rough_multi_case_can_load_layers_file(self) -> None:
        target = self.output_dir / "rough-case-layers.json"
        target.write_text(
            json.dumps(
                {
                    "base_layer": {
                        "epsilon": [1.0, 0.0],
                        "mu": [1.0, 0.0],
                        "sigma": 1.35e6,
                        "tau": 0.0,
                        "rq": 0.08e-6,
                    },
                    "layers": [
                        {
                            "epsilon": [1.0, 0.0],
                            "mu": [1.0, 0.0],
                            "sigma": 5.8e7,
                            "tau": 0.0,
                            "thickness": 75e-6,
                            "rq": 0.4e-6,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        class Args:
            case = [f"label=rough-json,model=rough-multi,layers_file={target}"]
            model = "normal-skin"
            sigma = 5.8e7
            sigma_metal = 5.8e7
            sigma1 = 5.8e7
            sigma2 = 4.2e7
            mu_r = 1.0
            tau = 2.5e-14
            rq = 0.5e-6
            rq01 = 0.5e-6
            rq12 = 0.5e-6
            t1 = 2e-6
            step_size = None
            xmin_factor = -5.0
            domain_factor = 10.0
            epsr_real = 1.0
            epsr_imag = 0.0
            mur_real = 1.0
            mur_imag = 0.0
            layers_file = None

        cases = collect_cases(Args())
        self.assertEqual(cases[0].label, "rough-json")
        self.assertIn("base_layer", cases[0].params)
        self.assertEqual(cases[0].params["step_size"], 0.08e-6 / 25.0)

    def test_parser_accepts_layers_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--layers-file", "layers.json"])
        self.assertEqual(args.layers_file, Path("layers.json"))

    def test_load_layers_from_file_requires_base_layer_and_layers(self) -> None:
        target = self.output_dir / "layers.json"
        target.write_text(
            json.dumps(
                {
                    "base_layer": {
                        "epsilon": [1.0, 0.0],
                        "mu": [1.0, 0.0],
                        "sigma": 5.8e7,
                        "tau": 0.0,
                    },
                    "layers": [
                        {
                            "epsilon": [1.0, 0.0],
                            "mu": [1.0, 0.0],
                            "sigma": 4.2e7,
                            "tau": 0.0,
                            "thickness": 1e-6,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        from surface_impedance.cli import load_layers_from_file

        loaded = load_layers_from_file(target)
        self.assertIn("base_layer", loaded)
        self.assertIn("layers", loaded)

    def test_export_json_writes_rows(self) -> None:
        rows = build_output_rows(
            np.array([1.0]),
            np.array([1.0 + 2.0j]),
            metadata={"label": "case-a", "model": "normal-skin"},
        )
        target = self.output_dir / "result.json"
        export_data(target, rows)
        loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["label"], "case-a")
        self.assertEqual(loaded[0]["frequency_hz"], 1.0)


if __name__ == "__main__":
    unittest.main()
