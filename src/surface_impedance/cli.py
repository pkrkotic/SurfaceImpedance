from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import build_output_rows, export_data
from .models import MODEL_SPECS, ProfileResult, compute_profile, compute_surface_impedance


@dataclass(frozen=True)
class ComparisonCase:
    label: str
    model: str
    params: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute surface impedance versus frequency for several models."
    )
    parser.add_argument("--list-models", action="store_true", help="List models and exit.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default="normal-skin",
        help="Surface impedance model to evaluate when --case is not used.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Comparison case definition such as "
            "'label=copper,model=normal-skin,sigma=5.8e7' or "
            "'label=bulk-half-space,model=half-space,sigma=5.8e7,tau=0'. "
            "Repeat --case to compare multiple curves."
        ),
    )
    parser.add_argument("--f-min", type=float, default=1e3, help="Minimum frequency in Hz.")
    parser.add_argument("--f-max", type=float, default=1e9, help="Maximum frequency in Hz.")
    parser.add_argument("--points", type=int, default=200, help="Number of frequency points.")

    parser.add_argument("--sigma", type=float, default=5.8e7, help="Conductivity in S/m.")
    parser.add_argument("--mu-r", type=float, default=1.0, help="Relative permeability.")
    parser.add_argument("--tau", type=float, default=2.5e-14, help="Relaxation time in s.")
    parser.add_argument(
        "--sigma-metal",
        type=float,
        default=5.8e7,
        help="Bulk conductivity in S/m for the rough-single model.",
    )
    parser.add_argument(
        "--rq",
        type=float,
        default=0.5e-6,
        help="RMS roughness in m for the rough-single model.",
    )
    parser.add_argument(
        "--sigma1",
        type=float,
        default=5.8e7,
        help="Conductivity of material 1 in S/m for the rough-multi model.",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=4.2e7,
        help="Conductivity of material 2 in S/m for the rough-multi model.",
    )
    parser.add_argument(
        "--rq01",
        type=float,
        default=0.5e-6,
        help="RMS roughness in m for the vacuum/material-1 transition.",
    )
    parser.add_argument(
        "--rq12",
        type=float,
        default=0.5e-6,
        help="RMS roughness in m for the material-1/material-2 transition.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=2.0e-6,
        help="Mean thickness of material 1 in m for the rough-multi model.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Finite-difference step size in m for roughness models.",
    )
    parser.add_argument(
        "--xmin-factor",
        type=float,
        default=-5.0,
        help="Left-domain extent factor for roughness models.",
    )
    parser.add_argument(
        "--domain-factor",
        type=float,
        default=10.0,
        help="Right-domain extent in skin-depth units for roughness models.",
    )
    parser.add_argument(
        "--epsr-real",
        type=float,
        default=1.0,
        help="Real part of relative permittivity for half-space model.",
    )
    parser.add_argument(
        "--epsr-imag",
        type=float,
        default=0.0,
        help="Imaginary part of relative permittivity for half-space model.",
    )
    parser.add_argument(
        "--mur-real",
        type=float,
        default=1.0,
        help="Real part of relative permeability for half-space model.",
    )
    parser.add_argument(
        "--mur-imag",
        type=float,
        default=0.0,
        help="Imaginary part of relative permeability for half-space model.",
    )
    parser.add_argument("--export", type=Path, help="Output data path (.csv or .json).")
    parser.add_argument("--plot", type=Path, help="Save a plot to this image path.")
    parser.add_argument(
        "--profile-plot",
        type=Path,
        help="Save the requested z-profile plot to this image path.",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display the plot window.")
    parser.add_argument(
        "--layers-file",
        type=Path,
        help="JSON file describing the multilayer stack for the multi-layer model.",
    )
    parser.add_argument(
        "--plot-scale",
        choices=["loglog", "semilogx", "semilogy", "plot"],
        default="loglog",
        help="Axis scaling to use for the impedance plot.",
    )
    parser.add_argument(
        "--frequency-unit",
        choices=["Hz", "kHz", "MHz", "GHz", "THz"],
        default="Hz",
        help="Display unit for the plot x-axis.",
    )
    parser.add_argument(
        "--impedance-unit",
        choices=["uOhm", "mOhm", "Ohm", "kOhm", "MOhm"],
        default="Ohm",
        help="Display unit for the plot y-axis.",
    )
    parser.add_argument(
        "--profile-frequency",
        type=float,
        help="Single frequency in Hz used for optional z-profile diagnostics.",
    )
    parser.add_argument(
        "--profile-quantity",
        action="append",
        choices=["conductivity", "magnetic-field", "current-density", "power-loss"],
        default=[],
        help=(
            "Profile quantity to plot versus z. Repeat to show multiple quantities. "
            "If omitted while --profile-frequency is set, all quantities are plotted."
        ),
    )
    parser.add_argument(
        "--profile-x-min-um",
        type=float,
        help="Optional lower x-limit for profile plots in um.",
    )
    parser.add_argument(
        "--profile-x-max-um",
        type=float,
        help="Optional upper x-limit for profile plots in um.",
    )
    parser.add_argument(
        "--profile-overlay-cases",
        action="store_true",
        help="Overlay all profile cases on a shared axis instead of separate subplots.",
    )
    parser.add_argument(
        "--profile-normalize-to",
        help=(
            "Case label used as the normalization reference for overlaid profile comparisons. "
            "Currently supported for current-density overlays."
        ),
    )
    return parser


def list_models() -> str:
    lines = []
    for name in sorted(MODEL_SPECS):
        spec = MODEL_SPECS[name]
        lines.append(f"{name}: {spec.description}")
        for parameter, text in spec.parameter_help.items():
            lines.append(f"  - {parameter}: {text}")
    return "\n".join(lines)


def select_model_params(args: argparse.Namespace) -> dict[str, float]:
    return default_params_for_model(args.model, args)


def default_rough_single_step_size(rq: float) -> float:
    return rq / 25.0


def default_rough_multi_step_size(rq01: float, rq12: float) -> float:
    return min(rq01, rq12) / 25.0


def default_rough_stack_step_size(stack_cfg: dict[str, object]) -> float:
    rq_values = [float(layer["rq"]) for layer in stack_cfg["layers"]]
    rq_values.append(float(stack_cfg["base_layer"]["rq"]))
    return min(rq_values) / 25.0


def default_params_for_model(model_name: str, args: argparse.Namespace) -> dict[str, float]:
    shared = {"mu_r": args.mu_r}
    if model_name == "normal-skin":
        return {**shared, "sigma": args.sigma}
    if model_name == "half-space":
        return {
            "sigma": args.sigma,
            "tau": args.tau,
            "epsr_real": args.epsr_real,
            "epsr_imag": args.epsr_imag,
            "mur_real": args.mur_real,
            "mur_imag": args.mur_imag,
        }
    if model_name == "rough-single":
        step_size = args.step_size
        if step_size is None:
            step_size = default_rough_single_step_size(args.rq)
        return {
            "sigma_metal": args.sigma_metal,
            "rq": args.rq,
            "mu_r": args.mu_r,
            "step_size": step_size,
            "xmin_factor": args.xmin_factor,
            "domain_factor": args.domain_factor,
        }
    if model_name == "rough-multi":
        if args.layers_file:
            params = load_layers_from_file(args.layers_file)
            step_size = args.step_size
            if step_size is None:
                step_size = default_rough_stack_step_size(params)
            return {
                **params,
                "step_size": step_size,
                "xmin_factor": args.xmin_factor,
                "domain_factor": args.domain_factor,
            }
        step_size = args.step_size
        if step_size is None:
            step_size = default_rough_multi_step_size(args.rq01, args.rq12)
        return {
            "sigma1": args.sigma1,
            "sigma2": args.sigma2,
            "rq01": args.rq01,
            "rq12": args.rq12,
            "t1": args.t1,
            "mu_r": args.mu_r,
            "step_size": step_size,
            "xmin_factor": args.xmin_factor,
            "domain_factor": args.domain_factor,
        }
    if model_name == "multi-layer":
        if not args.layers_file:
            raise ValueError("The multi-layer model requires --layers-file.")
        return load_layers_from_file(args.layers_file)
    raise ValueError(f"Unsupported model '{model_name}'.")


def load_layers_from_file(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Layer file must contain a JSON object with base_layer and layers.")
    if "base_layer" not in data or "layers" not in data:
        raise ValueError("Layer file must define both 'base_layer' and 'layers'.")
    return data


def default_case_for_args(args: argparse.Namespace) -> ComparisonCase:
    label = args.model
    if args.model == "multi-layer" and args.layers_file:
        label = args.layers_file.stem
    return ComparisonCase(
        label=label,
        model=args.model,
        params=select_model_params(args),
    )


def profile_layer_centers_um(case: ComparisonCase) -> list[tuple[float, str]]:
    if case.model not in {"rough-multi", "multi-layer"}:
        return []

    layers = case.params.get("layers")
    if isinstance(layers, list) and layers:
        centers: list[tuple[float, str]] = []
        position_m = 0.0
        for index, layer in enumerate(reversed(layers), start=1):
            thickness_m = float(layer["thickness"])
            center_um = (position_m + 0.5 * thickness_m) * 1e6
            label = str(layer.get("material") or layer.get("name") or f"Layer {index}")
            centers.append((center_um, label))
            position_m += thickness_m
        return centers

    if {"t1"} <= case.params.keys():
        return [(0.5 * float(case.params["t1"]) * 1e6, "Layer 1")]

    return []


def parse_case_definition(definition: str, args: argparse.Namespace, index: int) -> ComparisonCase:
    raw_parts = [part.strip() for part in definition.split(",") if part.strip()]
    parsed: dict[str, str] = {}
    for part in raw_parts:
        if "=" not in part:
            raise ValueError(
                f"Invalid case entry '{part}'. Use key=value pairs separated by commas."
            )
        key, value = part.split("=", 1)
        parsed[key.strip().replace("-", "_")] = value.strip()

    model_name = parsed.get("model", args.model)
    if model_name not in MODEL_SPECS:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{model_name}' in case {index}. Available: {available}")

    label = parsed.get("label", f"{model_name}-{index}")
    case_layers_file = parsed.get("layers_file")

    if case_layers_file is not None and model_name not in {"multi-layer", "rough-multi"}:
        raise ValueError(
            f"Parameter 'layers_file' is only valid for multilayer models in case {index}."
        )

    if case_layers_file is not None:
        params = load_layers_from_file(Path(case_layers_file))
        if model_name == "rough-multi":
            step_size = args.step_size
            if step_size is None:
                step_size = default_rough_stack_step_size(params)
            params = {
                **params,
                "step_size": step_size,
                "xmin_factor": args.xmin_factor,
                "domain_factor": args.domain_factor,
            }
    else:
        params = default_params_for_model(model_name, args)

    for key, raw_value in parsed.items():
        if key in {"label", "model", "layers_file"}:
            continue
        if key not in params:
            raise ValueError(
                f"Parameter '{key}' is not valid for model '{model_name}' in case {index}."
        )
        params[key] = float(raw_value)

    if model_name == "rough-single" and "step_size" not in parsed:
        params["step_size"] = default_rough_single_step_size(float(params["rq"]))
    if model_name == "rough-multi" and "step_size" not in parsed and case_layers_file is None:
        params["step_size"] = default_rough_multi_step_size(
            float(params["rq01"]),
            float(params["rq12"]),
        )

    return ComparisonCase(label=label, model=model_name, params=params)


def collect_cases(args: argparse.Namespace) -> list[ComparisonCase]:
    if not args.case:
        return [default_case_for_args(args)]

    return [
        parse_case_definition(definition, args, index)
        for index, definition in enumerate(args.case, start=1)
    ]


def make_frequency_grid(f_min: float, f_max: float, points: int) -> np.ndarray:
    if f_min <= 0.0 or f_max <= 0.0:
        raise ValueError("Frequencies must be strictly positive.")
    if f_min >= f_max:
        raise ValueError("f-min must be smaller than f-max.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    return np.logspace(np.log10(f_min), np.log10(f_max), points)



########################################
### colour palette
########################################
palette = [
    "#104375",  # 0
    "#77C8A2",  # 1
    "#FCDC9C",  # 2
    "#F07A46",  # 3
    "#DC3977",  # 4
    "#7C1D6F",  # 5
    "#AD1017",  # 6
    "#939698",  # 7
    "#408099",  # 8
]


def _plot_with_scale(ax, scale: str, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    if scale == "loglog":
        ax.loglog(x, y, **kwargs)
        return
    if scale == "semilogx":
        ax.semilogx(x, y, **kwargs)
        return
    if scale == "semilogy":
        ax.semilogy(x, y, **kwargs)
        return
    ax.plot(x, y, **kwargs)


def _frequency_unit_factor(unit: str) -> float:
    factors = {
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
        "THz": 1e12,
    }
    return factors[unit]


def _impedance_unit_factor(unit: str) -> float:
    factors = {
        "uOhm": 1e-6,
        "mOhm": 1e-3,
        "Ohm": 1.0,
        "kOhm": 1e3,
        "MOhm": 1e6,
    }
    return factors[unit]


def maybe_plot(
    freq_hz: np.ndarray,
    cases: list[tuple[ComparisonCase, np.ndarray]],
    plot_path: Path | None,
    show_plot: bool,
    plot_scale: str,
    frequency_unit: str,
    impedance_unit: str,
) -> Path | None:
    if not plot_path and not show_plot:
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requested, but matplotlib is not installed. Install with 'pip install -e .[plot]'."
        ) from exc

    multi_layer_cases = [
        (case, impedance)
        for case, impedance in cases
        if case.model == "multi-layer"
    ]
    nrows = 1 + len(multi_layer_cases)
    height_ratios = [4] + [1] * len(multi_layer_cases)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8, 4 + 1.6 * len(multi_layer_cases)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if nrows == 1:
        ax = axes
        schematic_axes: list[object] = []
    else:
        ax = axes[0]
        schematic_axes = list(axes[1:])

    x_values = freq_hz / _frequency_unit_factor(frequency_unit)
    y_scale = _impedance_unit_factor(impedance_unit)
    for index, (case, impedance) in enumerate(cases):
        color = palette[index % len(palette)]
        _plot_with_scale(
            ax,
            plot_scale,
            x_values,
            np.real(impedance) / y_scale,
            label=f"{case.label} Re(Zs)",
            linewidth=3,
            color=color,
            linestyle="-",
        )
        _plot_with_scale(
            ax,
            plot_scale,
            x_values,
            np.abs(np.imag(impedance)) / y_scale,
            label=f"{case.label} Im(Zs)",
            linewidth=2,
            color=color,
            linestyle="--",
        )

    unit_labels = {
        "uOhm": r"$\mu\Omega$",
        "mOhm": r"$\mathrm{m}\Omega$",
        "Ohm": r"$\Omega$",
        "kOhm": r"$\mathrm{k}\Omega$",
        "MOhm": r"$\mathrm{M}\Omega$",
    }
    ax.set_ylabel(f"Surface Impedance [{unit_labels[impedance_unit]}]", fontsize=11)
    ax.set_xlabel(f"Frequency [{frequency_unit}]", fontsize=11)
    # ax.set_title(
        # f"Surface impedance comparison ({plot_scale}, {frequency_unit}, {impedance_unit})"
    # )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(False, which="minor")
    ax.minorticks_on()
    ax.legend()

    for schematic_ax, (case, _) in zip(schematic_axes, multi_layer_cases, strict=True):
        layers = case.params["layers"]
        base_layer = case.params["base_layer"]
        x_left = 0.0
        max_thickness = max(float(layer["thickness"]) for layer in layers)
        base_width = max_thickness

        base_lines = []
        if base_layer.get("name"):
            base_lines.append(str(base_layer["name"]))
        if base_layer.get("material"):
            base_lines.append(str(base_layer["material"]))
        if base_layer.get("note"):
            base_lines.append(str(base_layer["note"]))

        base_rect = Rectangle(
            (-base_width, 0.0),
            base_width,
            1.0,
            facecolor="#d9d9d9",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.95,
        )
        schematic_ax.add_patch(base_rect)
        schematic_ax.axvline(0.0, color="black", linewidth=1.2, linestyle=":")
        base_label_text = "Base layer"
        if base_lines:
            base_label_text += "\n" + "\n".join(base_lines[:3])
        schematic_ax.text(
            -base_width / 2.0,
            0.5,
            base_label_text,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

        for index, layer in enumerate(layers):
            thickness = float(layer["thickness"])
            color = palette[index % len(palette)]
            rect = Rectangle(
                (x_left, 0.0),
                thickness,
                1.0,
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.85,
            )
            schematic_ax.add_patch(rect)

            text_lines = []
            if layer.get("name"):
                text_lines.append(str(layer["name"]))
            if layer.get("material"):
                text_lines.append(str(layer["material"]))
            if layer.get("note"):
                text_lines.append(str(layer["note"]))
            label_text = "\n".join(text_lines[:3])
            schematic_ax.text(
                x_left + thickness / 2.0,
                0.5,
                label_text,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            x_left += thickness

        schematic_ax.set_xlim(-base_width, max(x_left, 1e-12))
        schematic_ax.set_ylim(0.0, 1.0)
        schematic_ax.set_yticks([])
        schematic_ax.set_ylabel("Layer", fontsize=9)
        schematic_ax.set_xlabel("Thickness [m]", fontsize=10)
        schematic_ax.set_title(f"Representative layer stack: {case.label}", fontsize=10)
        schematic_ax.grid(False)
        for spine in schematic_ax.spines.values():
            spine.set_linewidth(1.2)

    fig.tight_layout()

    saved_path = None
    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        saved_path = plot_path

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def maybe_plot_profiles(
    cases: list[tuple[ComparisonCase, ProfileResult]],
    plot_path: Path | None,
    show_plot: bool,
    requested_quantities: list[str],
    x_min_um: float | None = None,
    x_max_um: float | None = None,
    overlay_cases: bool = False,
    normalize_to_label: str | None = None,
) -> Path | None:
    if not cases:
        return None
    if not plot_path and not show_plot:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requested, but matplotlib is not installed. Install with 'pip install -e .[plot]'."
        ) from exc

    quantities = requested_quantities or [
        "conductivity",
        "magnetic-field",
        "current-density",
        "power-loss",
    ]
    quantity_specs = {
        "conductivity": ("Normalized conductivity", "normalized_conductivity", "-"),
        "magnetic-field": ("Normalized |B(z)|", "normalized_magnetic_field", "--"),
        "current-density": ("Normalized |J(z)|", "normalized_current_density", "-."),
        "power-loss": (
            "Normalized power-loss density",
            "normalized_power_loss_density",
            ":",
        ),
    }

    global_x_min_um = min(float(np.min(profile.x_m * 1e6)) for _, profile in cases)
    global_x_max_um = max(float(np.max(profile.x_m * 1e6)) for _, profile in cases)
    plot_x_min_um = global_x_min_um if x_min_um is None else x_min_um
    plot_x_max_um = global_x_max_um if x_max_um is None else x_max_um

    if plot_x_min_um >= plot_x_max_um:
        raise ValueError("profile-x-min-um must be smaller than profile-x-max-um.")

    if overlay_cases:
        if len(quantities) != 1:
            raise ValueError("profile-overlay-cases currently requires exactly one --profile-quantity.")

        quantity = quantities[0]
        if normalize_to_label is not None and quantity != "current-density":
            raise ValueError(
                "profile-normalize-to is currently supported only for current-density overlays."
            )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))
        label, attribute, linestyle = quantity_specs[quantity]

        normalization_scale = 1.0
        ylabel = label
        if normalize_to_label is not None:
            reference_matches = [
                profile
                for case, profile in cases
                if case.label == normalize_to_label
            ]
            if not reference_matches:
                available = ", ".join(case.label for case, _ in cases)
                raise ValueError(
                    f"profile-normalize-to='{normalize_to_label}' not found. Available labels: {available}"
                )
            reference_profile = reference_matches[0]
            x_ref_um = reference_profile.x_m * 1e6
            mask = (x_ref_um >= plot_x_min_um) & (x_ref_um <= plot_x_max_um)
            reference_values = reference_profile.current_density_magnitude[mask]
            if reference_values.size == 0:
                raise ValueError("Reference profile has no samples inside the requested x-range.")
            normalization_scale = float(np.max(reference_values))
            if normalization_scale <= 0.0:
                raise ValueError("Reference profile current-density maximum must be positive.")
            ylabel = f"|J(z)| / max_ref(|J(z)|) [{normalize_to_label}]"

        for index, (case, profile) in enumerate(cases):
            color = palette[index % len(palette)]
            x_um = profile.x_m * 1e6
            if normalize_to_label is not None:
                values = profile.current_density_magnitude / normalization_scale
            else:
                values = getattr(profile, attribute)
            ax.plot(
                x_um,
                values,
                linewidth=2.5,
                linestyle=linestyle,
                color=color,
                label=case.label,
            )

        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlim(plot_x_min_um, plot_x_max_um)
        ax.set_xlabel(r"z [$\mu$m]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"z-profile comparison: {label}", fontsize=10)
        ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
        ax.grid(False, which="minor")
        ax.minorticks_on()
        ax.legend()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()

        saved_path = None
        if plot_path:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=180, bbox_inches="tight")
            saved_path = plot_path

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    fig, axes = plt.subplots(
        nrows=len(cases),
        ncols=1,
        figsize=(8, 3.5 * len(cases)),
        squeeze=False,
        sharex=True,
    )

    for row_index, (case, profile) in enumerate(cases):
        ax = axes[row_index][0]
        x_um = profile.x_m * 1e6
        for quantity_index, quantity in enumerate(quantities):
            label, attribute, linestyle = quantity_specs[quantity]
            color = palette[quantity_index % len(palette)]
            values = getattr(profile, attribute)
            if quantity in {"conductivity", "power-loss"} and case.model == "multi-layer":
                ax.step(
                    x_um,
                    values,
                    where="post",
                    linewidth=2.5,
                    linestyle="-" if quantity == "conductivity" else ":",
                    color=color,
                    label=label,
                )
            else:
                ax.plot(
                    x_um,
                    values,
                    linewidth=2.5,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                )
        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
        for center_um, center_label in profile_layer_centers_um(case):
            if plot_x_min_um <= center_um <= plot_x_max_um:
                ax.axvline(center_um, color="#666666", linestyle="-.", linewidth=1.0, alpha=0.8)
                ax.text(
                    center_um,
                    0.98,
                    center_label,
                    rotation=90,
                    va="top",
                    ha="right",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    color="#555555",
                )
        ax.set_xlim(plot_x_min_um, plot_x_max_um)
        ax.set_xlabel(r"z [$\mu$m]")
        ax.set_ylabel("Normalized value")
        ax.set_title(f"z-profile diagnostics: {case.label}", fontsize=10)
        ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
        ax.grid(False, which="minor")
        ax.minorticks_on()
        ax.legend()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    fig.tight_layout()

    saved_path = None
    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        saved_path = plot_path

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return 0

    freq_hz = make_frequency_grid(args.f_min, args.f_max, args.points)
    cases = collect_cases(args)
    computed_cases: list[tuple[ComparisonCase, np.ndarray]] = []
    profile_cases: list[tuple[ComparisonCase, ProfileResult]] = []
    rows: list[dict[str, float | str]] = []

    print(f"Frequency range: {args.f_min:.3e} Hz to {args.f_max:.3e} Hz")
    print(f"Points: {args.points}")

    for case in cases:
        compute_params = dict(case.params)
        if case.model in {"rough-single", "rough-multi"}:
            compute_params["_progress"] = True
            compute_params["_progress_label"] = case.label
        impedance = compute_surface_impedance(case.model, freq_hz, compute_params)
        computed_cases.append((case, impedance))
        rows.extend(
            build_output_rows(
                freq_hz,
                impedance,
                metadata={"label": case.label, "model": case.model},
            )
        )
        print(f"Case: {case.label} ({case.model})")
        print(f"First |Zs|: {abs(impedance[0]):.6e} Ohm")
        print(f"Last  |Zs|: {abs(impedance[-1]):.6e} Ohm")
        if args.profile_frequency is not None:
            profile_cases.append(
                (
                    case,
                    compute_profile(case.model, args.profile_frequency, compute_params),
                )
            )

    if args.export:
        exported = export_data(args.export, rows)
        print(f"Exported data: {exported}")

    plot_result = maybe_plot(
        freq_hz,
        computed_cases,
        args.plot,
        args.show_plot,
        args.plot_scale,
        args.frequency_unit,
        args.impedance_unit,
    )
    if plot_result:
        print(f"Saved plot: {plot_result}")

    profile_plot_result = maybe_plot_profiles(
        profile_cases,
        args.profile_plot,
        args.show_plot,
        args.profile_quantity,
        args.profile_x_min_um,
        args.profile_x_max_um,
        args.profile_overlay_cases,
        args.profile_normalize_to,
    )
    if profile_plot_result:
        print(f"Saved profile plot: {profile_plot_result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
