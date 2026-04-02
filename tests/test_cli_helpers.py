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
    default_rough_single_step_size,
    make_frequency_grid,
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
                "power-loss",
                "--profile-plot",
                "profile.png",
            ]
        )
        self.assertEqual(args.profile_frequency, 1e9)
        self.assertEqual(args.profile_quantity, ["conductivity", "power-loss"])
        self.assertEqual(args.profile_plot, Path("profile.png"))

    def test_frequency_unit_factor_maps_units(self) -> None:
        self.assertEqual(_frequency_unit_factor("Hz"), 1.0)
        self.assertEqual(_frequency_unit_factor("MHz"), 1e6)

    def test_impedance_unit_factor_maps_units(self) -> None:
        self.assertEqual(_impedance_unit_factor("Ohm"), 1.0)
        self.assertEqual(_impedance_unit_factor("uOhm"), 1e-6)

    def test_default_rough_single_step_size_scales_with_rq(self) -> None:
        self.assertEqual(default_rough_single_step_size(1e-6), 4e-8)

    def test_rough_single_defaults_use_requested_convention(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--model", "rough-single", "--rq", "1e-6"])
        params = default_params_for_model("rough-single", args)
        self.assertEqual(params["step_size"], 4e-8)
        self.assertEqual(params["xmin_factor"], -5.0)
        self.assertEqual(params["domain_factor"], 10.0)

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
