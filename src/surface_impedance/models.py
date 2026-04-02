from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

MU_0 = 4e-7 * np.pi  # H/m
EPS_0 = 8.854187817e-12  # F/m


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    parameter_help: dict[str, str]
    evaluator: Callable[[np.ndarray, dict[str, object]], np.ndarray]


@dataclass(frozen=True)
class ProfileResult:
    x_m: np.ndarray
    normalized_conductivity: np.ndarray
    normalized_magnetic_field: np.ndarray
    normalized_power_loss_density: np.ndarray


def _normal_skin_effect(freq_hz: np.ndarray, params: dict[str, object]) -> np.ndarray:
    sigma = float(params["sigma"])
    mu_r = float(params["mu_r"])
    omega = 2 * np.pi * freq_hz
    mu = MU_0 * mu_r
    return (1.0 + 1.0j) * np.sqrt(omega * mu / (2.0 * sigma))


def conductivity_metal(
    frequency: np.ndarray, sigma: float, tau: float | None
) -> np.ndarray | complex:
    if tau is None or tau == 0:
        return sigma + 0j
    omega = 2 * np.pi * frequency
    return sigma / (1 - 1j * omega * tau)


def epsilon_eff(epsr_real: float, epsr_imag: float) -> complex:
    return EPS_0 * (epsr_real - 1j * epsr_imag)


def mu_eff(mur_real: float, mur_imag: float) -> complex:
    return MU_0 * (mur_real - 1j * mur_imag)


def propagation_constant(
    frequency: np.ndarray,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    sigma: float,
    tau: float | None,
) -> np.ndarray:
    omega = 2 * np.pi * frequency
    permea = mu_eff(mu[0], mu[1])
    cond = conductivity_metal(frequency, sigma, tau)
    eps = epsilon_eff(epsilon[0], epsilon[1])
    gamma = np.sqrt(1j * omega * permea * (cond + 1j * omega * eps))
    return np.where(np.real(gamma) < 0, -gamma, gamma)


def wavenumber(
    frequency: np.ndarray,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    sigma: float,
    tau: float | None,
) -> np.ndarray:
    return -1j * propagation_constant(frequency, epsilon, mu, sigma, tau)


def waveimpedance(
    frequency: np.ndarray,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    sigma: float,
    tau: float | None,
) -> np.ndarray:
    omega = 2 * np.pi * frequency
    permea = mu_eff(mu[0], mu[1])
    k = wavenumber(frequency, epsilon, mu, sigma, tau)
    return omega * permea / k


def _half_space(freq_hz: np.ndarray, params: dict[str, object]) -> np.ndarray:
    epsilon = (float(params["epsr_real"]), float(params["epsr_imag"]))
    mu = (float(params["mur_real"]), float(params["mur_imag"]))
    sigma = float(params["sigma"])
    tau = float(params["tau"]) if "tau" in params else None
    return waveimpedance(freq_hz, epsilon, mu, sigma, tau)


def phase_thickness(
    frequency: np.ndarray,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    sigma: float,
    tau: float | None,
    thickness: float,
) -> np.ndarray:
    return 1j * wavenumber(frequency, epsilon, mu, sigma, tau) * thickness


def lambda_p(
    z_prev: np.ndarray,
    frequency: np.ndarray,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    sigma: float,
    tau: float | None,
    thickness: float,
) -> np.ndarray:
    zeta = waveimpedance(frequency, epsilon, mu, sigma, tau)
    k_term = phase_thickness(frequency, epsilon, mu, sigma, tau, thickness)
    return ((z_prev - zeta) / (z_prev + zeta)) * np.exp(-2 * k_term)


def zs_next(
    z_prev: np.ndarray,
    frequency: np.ndarray,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    sigma: float,
    tau: float | None,
    thickness: float,
) -> np.ndarray:
    zeta = waveimpedance(frequency, epsilon, mu, sigma, tau)
    lam = lambda_p(z_prev, frequency, epsilon, mu, sigma, tau, thickness)
    return zeta * (1 + lam) / (1 - lam)


def _multi_layer(frequency: np.ndarray, z_start: np.ndarray, layers: list[dict[str, object]]) -> np.ndarray:
    z_val = np.array(z_start, dtype=complex) + 0j * np.asarray(frequency)
    for layer in layers:
        z_val = zs_next(
            z_prev=z_val,
            frequency=frequency,
            epsilon=tuple(layer["epsilon"]),
            mu=tuple(layer["mu"]),
            sigma=float(layer["sigma"]),
            tau=float(layer["tau"]),
            thickness=float(layer["thickness"]),
        )
    return z_val


def _multi_layer_model(freq_hz: np.ndarray, params: dict[str, object]) -> np.ndarray:
    base_layer = params["base_layer"]
    layers = params["layers"]
    z_start = waveimpedance(
        freq_hz,
        tuple(base_layer["epsilon"]),
        tuple(base_layer["mu"]),
        float(base_layer["sigma"]),
        float(base_layer["tau"]),
    )
    return _multi_layer(freq_hz, z_start, layers)


def cdf(x: np.ndarray, mean: float, rq_val: float) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)((x - mean) / (rq_val * np.sqrt(2.0))))


def sigma_single_layer(x: np.ndarray, single_cfg: dict[str, float]) -> np.ndarray:
    return single_cfg["sigma_metal"] * cdf(x, mean=0.0, rq_val=single_cfg["rq"])


def _normalize_profile(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    if max_value <= 0.0:
        return np.zeros_like(values)
    return values / max_value


def _skin_depth(omega: float, mu: float, sigma: float) -> float:
    return math.sqrt(2.0 / (omega * mu * sigma))


def _build_smooth_half_space_profile(
    f_hz: float,
    *,
    sigma: float,
    tau: float | None,
    epsilon: tuple[float, float],
    mu: tuple[float, float],
    x_left_factor: float = -5.0,
    x_right_factor: float = 10.0,
    points: int = 600,
) -> ProfileResult:
    omega = 2.0 * math.pi * f_hz
    cond = np.asarray(conductivity_metal(np.array([f_hz]), sigma, tau)).reshape(-1)[0]
    permea = mu_eff(mu[0], mu[1])
    gamma = propagation_constant(
        np.array([f_hz]),
        epsilon,
        mu,
        sigma,
        tau,
    )[0]
    delta = _skin_depth(omega, float(np.real(permea)), sigma)
    x_left = x_left_factor * delta
    x_right = x_right_factor * delta
    x_m = np.linspace(x_left, x_right, points)

    magnetic_field = np.ones_like(x_m, dtype=np.complex128)
    conductor_mask = x_m >= 0.0
    magnetic_field[conductor_mask] = np.exp(-gamma * x_m[conductor_mask])

    current_density = np.zeros_like(magnetic_field)
    current_density[conductor_mask] = (gamma / permea) * magnetic_field[conductor_mask]

    power_loss_density = np.zeros_like(x_m, dtype=float)
    if np.real(1.0 / cond) > 0.0:
        power_loss_density[conductor_mask] = (
            0.5 * np.real(1.0 / cond) * np.abs(current_density[conductor_mask]) ** 2
        )

    normalized_conductivity = np.zeros_like(x_m, dtype=float)
    normalized_conductivity[conductor_mask] = 1.0

    return ProfileResult(
        x_m=x_m,
        normalized_conductivity=normalized_conductivity,
        normalized_magnetic_field=_normalize_profile(np.abs(magnetic_field)),
        normalized_power_loss_density=_normalize_profile(power_loss_density),
    )

def solve_tridiagonal_for_frequency_single(
    f_hz: float,
    *,
    sigma_metal: float,
    rq: float,
    mu_r: float,
    xmin_factor: float,
    domain_factor: float,
    step_size: float,
) -> complex:
    omega = 2.0 * math.pi * f_hz
    mu0 = MU_0 * mu_r
    delta = math.sqrt(2.0 / (omega * mu0 * sigma_metal))
    x_left = xmin_factor * rq
    x_right = domain_factor * delta

    xp = np.arange(x_left, x_right + step_size, step_size)
    n_points = len(xp)

    sigma = sigma_single_layer(
        xp,
        {
            "sigma_metal": sigma_metal,
            "rq": rq,
        },
    )
    sigma_floor = np.maximum(sigma, 1e-16)
    ln_sigma = np.log(sigma_floor)

    b_left = 1.0 + 0.0j
    b_right = 0.0 + 0.0j

    lower = np.zeros(n_points, dtype=np.complex128)
    diag = np.zeros(n_points, dtype=np.complex128)
    upper = np.zeros(n_points, dtype=np.complex128)
    rhs = np.zeros(n_points, dtype=np.complex128)

    diag[0] = 1.0
    rhs[0] = b_left
    diag[-1] = 1.0
    rhs[-1] = b_right

    for index in range(1, n_points - 1):
        dlnsigma_dx = (ln_sigma[index + 1] - ln_sigma[index - 1]) / (2.0 * step_size)
        a_plus = 1.0 / step_size**2 - dlnsigma_dx / (2.0 * step_size)
        a_minus = 1.0 / step_size**2 + dlnsigma_dx / (2.0 * step_size)
        a_zero = -2.0 / step_size**2 - 1j * omega * mu0 * sigma[index]
        lower[index] = a_minus
        diag[index] = a_zero
        upper[index] = a_plus

    cprime = np.zeros(n_points, dtype=np.complex128)
    dprime = np.zeros(n_points, dtype=np.complex128)
    cprime[0] = upper[0] / diag[0]
    dprime[0] = rhs[0] / diag[0]

    for index in range(1, n_points):
        denom = diag[index] - lower[index] * cprime[index - 1]
        cprime[index] = upper[index] / denom if index < n_points - 1 else 0.0
        dprime[index] = (rhs[index] - lower[index] * dprime[index - 1]) / denom

    b_profile = np.zeros(n_points, dtype=np.complex128)
    b_profile[-1] = dprime[-1]
    for index in range(n_points - 2, -1, -1):
        b_profile[index] = dprime[index] - cprime[index] * b_profile[index + 1]

    j_profile = np.empty_like(b_profile)
    j_profile[0] = (b_profile[1] - b_profile[0]) / step_size / mu0
    j_profile[-1] = (b_profile[-1] - b_profile[-2]) / step_size / mu0
    j_profile[1:-1] = (b_profile[2:] - b_profile[:-2]) / (2.0 * step_size) / mu0

    mask =  sigma > 0

    integral_b = np.trapezoid(b_profile[mask], xp[mask])
    integral_j = np.trapezoid(j_profile[mask], xp[mask])
    return (-1j * omega * integral_b) / integral_j


def solve_tridiagonal_profile_single(
    f_hz: float,
    *,
    sigma_metal: float,
    rq: float,
    mu_r: float,
    xmin_factor: float,
    domain_factor: float,
    step_size: float,
) -> ProfileResult:
    omega = 2.0 * math.pi * f_hz
    mu0 = MU_0 * mu_r
    delta = math.sqrt(2.0 / (omega * mu0 * sigma_metal))
    x_left = xmin_factor * rq
    x_right = domain_factor * delta

    xp = np.arange(x_left, x_right + step_size, step_size)
    n_points = len(xp)

    sigma = sigma_single_layer(
        xp,
        {
            "sigma_metal": sigma_metal,
            "rq": rq,
        },
    )
    sigma_floor = np.maximum(sigma, 1e-16)
    ln_sigma = np.log(sigma_floor)

    b_left = 1.0 + 0.0j
    b_right = 0.0 + 0.0j

    lower = np.zeros(n_points, dtype=np.complex128)
    diag = np.zeros(n_points, dtype=np.complex128)
    upper = np.zeros(n_points, dtype=np.complex128)
    rhs = np.zeros(n_points, dtype=np.complex128)

    diag[0] = 1.0
    rhs[0] = b_left
    diag[-1] = 1.0
    rhs[-1] = b_right

    for index in range(1, n_points - 1):
        dlnsigma_dx = (ln_sigma[index + 1] - ln_sigma[index - 1]) / (2.0 * step_size)
        a_plus = 1.0 / step_size**2 - dlnsigma_dx / (2.0 * step_size)
        a_minus = 1.0 / step_size**2 + dlnsigma_dx / (2.0 * step_size)
        a_zero = -2.0 / step_size**2 - 1j * omega * mu0 * sigma[index]
        lower[index] = a_minus
        diag[index] = a_zero
        upper[index] = a_plus

    cprime = np.zeros(n_points, dtype=np.complex128)
    dprime = np.zeros(n_points, dtype=np.complex128)
    cprime[0] = upper[0] / diag[0]
    dprime[0] = rhs[0] / diag[0]

    for index in range(1, n_points):
        denom = diag[index] - lower[index] * cprime[index - 1]
        cprime[index] = upper[index] / denom if index < n_points - 1 else 0.0
        dprime[index] = (rhs[index] - lower[index] * dprime[index - 1]) / denom

    b_profile = np.zeros(n_points, dtype=np.complex128)
    b_profile[-1] = dprime[-1]
    for index in range(n_points - 2, -1, -1):
        b_profile[index] = dprime[index] - cprime[index] * b_profile[index + 1]

    j_profile = np.empty_like(b_profile)
    j_profile[0] = (b_profile[1] - b_profile[0]) / step_size / mu0
    j_profile[-1] = (b_profile[-1] - b_profile[-2]) / step_size / mu0
    j_profile[1:-1] = (b_profile[2:] - b_profile[:-2]) / (2.0 * step_size) / mu0

    power_loss_density = 0.5 * np.real(1.0 / sigma_floor) * np.abs(j_profile) ** 2

    return ProfileResult(
        x_m=xp,
        normalized_conductivity=sigma / sigma_metal,
        normalized_magnetic_field=_normalize_profile(np.abs(b_profile)),
        normalized_power_loss_density=_normalize_profile(power_loss_density),
    )


def _rough_single_model(freq_hz: np.ndarray, params: dict[str, object]) -> np.ndarray:
    show_progress = bool(params.get("_progress", False))
    progress_label = str(params.get("_progress_label", "rough-single"))
    total = len(freq_hz)
    values: list[complex] = []

    for index, freq in enumerate(freq_hz, start=1):
        values.append(
            solve_tridiagonal_for_frequency_single(
                f_hz=float(freq),
                sigma_metal=float(params["sigma_metal"]),
                rq=float(params["rq"]),
                mu_r=float(params["mu_r"]),
                xmin_factor=float(params["xmin_factor"]),
                domain_factor=float(params["domain_factor"]),
                step_size=float(params["step_size"]),
            )
        )
        if show_progress and total > 1:
            fraction = index / total
            bar_width = 24
            filled = int(bar_width * fraction)
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\r[{bar}] {fraction:6.1%}  {progress_label}  ({index}/{total})"
            )
            sys.stdout.flush()

    if show_progress and total > 1:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return np.asarray(values, dtype=np.complex128)


MODEL_SPECS: dict[str, ModelSpec] = {
    "normal-skin": ModelSpec(
        name="normal-skin",
        description="Classical good-conductor skin-effect model.",
        parameter_help={
            "sigma": "DC conductivity in S/m.",
            "mu_r": "Relative permeability.",
        },
        evaluator=_normal_skin_effect,
    ),
    "half-space": ModelSpec(
        name="half-space",
        description="Generalized half-space wave-impedance approximation.",
        parameter_help={
            "sigma": "Conductivity in S/m.",
            "tau": "Relaxation time in s. Use 0 for frequency-independent conductivity.",
            "epsr_real": "Real part of relative permittivity.",
            "epsr_imag": "Imaginary part of relative permittivity.",
            "mur_real": "Real part of relative permeability.",
            "mur_imag": "Imaginary part of relative permeability.",
        },
        evaluator=_half_space,
    ),
    "multi-layer": ModelSpec(
        name="multi-layer",
        description="Multilayer surface impedance with half-space approximation.",
        parameter_help={
            "base_layer": "Base-layer material loaded from a JSON configuration file.",
            "layers": "Finite layer stack loaded from a JSON configuration file.",
        },
        evaluator=_multi_layer_model,
    ),
    "rough-single": ModelSpec(
        name="rough-single",
        description="Surface roughness gradient model for a single rough transition.",
        parameter_help={
            "sigma_metal": "Bulk metal conductivity in S/m.",
            "rq": "RMS roughness in m.",
            "mu_r": "Relative permeability.",
            "step_size": "Finite-difference step size in m.",
            "xmin_factor": "Left-domain extent in skin-depth units.",
            "domain_factor": "Right-domain extent in skin-depth units.",
        },
        evaluator=_rough_single_model,
    ),
}


def compute_surface_impedance(
    model_name: str,
    freq_hz: np.ndarray,
    params: dict[str, object],
) -> np.ndarray:
    if model_name not in MODEL_SPECS:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    _validate_inputs(freq_hz, params)
    return MODEL_SPECS[model_name].evaluator(freq_hz, params)


def compute_profile(
    model_name: str,
    frequency_hz: float,
    params: dict[str, object],
) -> ProfileResult:
    if frequency_hz <= 0.0:
        raise ValueError("Profile frequency must be strictly positive.")

    if model_name == "normal-skin":
        return _build_smooth_half_space_profile(
            frequency_hz,
            sigma=float(params["sigma"]),
            tau=0.0,
            epsilon=(1.0, 0.0),
            mu=(float(params["mu_r"]), 0.0),
        )
    if model_name == "half-space":
        return _build_smooth_half_space_profile(
            frequency_hz,
            sigma=float(params["sigma"]),
            tau=float(params["tau"]) if "tau" in params else None,
            epsilon=(float(params["epsr_real"]), float(params["epsr_imag"])),
            mu=(float(params["mur_real"]), float(params["mur_imag"])),
        )
    if model_name == "rough-single":
        return solve_tridiagonal_profile_single(
            frequency_hz,
            sigma_metal=float(params["sigma_metal"]),
            rq=float(params["rq"]),
            mu_r=float(params["mu_r"]),
            xmin_factor=float(params["xmin_factor"]),
            domain_factor=float(params["domain_factor"]),
            step_size=float(params["step_size"]),
        )
    raise NotImplementedError(
        f"x-profile plotting is not implemented for model '{model_name}' yet."
    )


def _validate_inputs(freq_hz: np.ndarray, params: dict[str, object]) -> None:
    if np.any(freq_hz <= 0.0):
        raise ValueError("All frequencies must be strictly positive.")

    for key, value in params.items():
        if not isinstance(value, (int, float)):
            continue
        if key in {
            "sigma",
            "sigma_metal",
            "mu_r",
            "epsr_real",
            "mur_real",
            "rq",
            "step_size",
            "domain_factor",
        } and value <= 0.0:
            raise ValueError(f"Parameter '{key}' must be positive.")
        if key in {"tau", "epsr_imag", "mur_imag", "sigma0", "xmin_factor"}:
            if key == "tau" and value < 0.0:
                raise ValueError("Parameter 'tau' must be non-negative.")

    base_layer = params.get("base_layer")
    if base_layer is not None:
        if not isinstance(base_layer, dict):
            raise ValueError("Parameter 'base_layer' must be a dictionary.")
        required = {"epsilon", "mu", "sigma", "tau"}
        for required_key in required:
            if required_key not in base_layer:
                raise ValueError(f"Base layer is missing '{required_key}'.")
        if float(base_layer["sigma"]) <= 0.0:
            raise ValueError("Base layer parameter 'sigma' must be positive.")
        if "tau" in base_layer and float(base_layer["tau"]) < 0.0:
            raise ValueError("Base layer parameter 'tau' must be non-negative.")

    layers = params.get("layers")
    if layers is not None:
        if not isinstance(layers, list) or not layers:
            raise ValueError("Parameter 'layers' must be a non-empty list.")
        for index, layer in enumerate(layers, start=1):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer {index} must be a dictionary.")
            required = {"epsilon", "mu", "sigma", "tau", "thickness"}
            for required_key in required:
                if required_key not in layer:
                    raise ValueError(f"Layer {index} is missing '{required_key}'.")
            if float(layer["sigma"]) <= 0.0:
                raise ValueError(f"Layer {index} parameter 'sigma' must be positive.")
            if "tau" in layer and float(layer["tau"]) < 0.0:
                raise ValueError(f"Layer {index} parameter 'tau' must be non-negative.")
            if float(layer["thickness"]) <= 0.0:
                raise ValueError(f"Layer {index} parameter 'thickness' must be positive.")
