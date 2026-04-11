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
    for layer in reversed(layers):
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


def sigma_rough_multilayer(x: np.ndarray, multi_cfg: dict[str, float]) -> np.ndarray:
    x01 = 0.0
    x12 = multi_cfg["t1"]
    cdf0_u = cdf(x, mean=x01, rq_val=multi_cfg["rq01"])
    cdf0_d = cdf(x, mean=x12, rq_val=multi_cfg["rq12"])
    cdf1_u = cdf(x, mean=x12, rq_val=multi_cfg["rq12"])
    return (cdf0_u - cdf0_d) * multi_cfg["sigma1"] + cdf1_u * multi_cfg["sigma2"]


def _rough_stack_transitions_from_layers(
    base_layer: dict[str, object],
    layers: list[dict[str, object]],
) -> list[dict[str, float]]:
    transitions: list[dict[str, float]] = []
    position = 0.0
    previous_sigma = 0.0

    for medium in [*reversed(layers), base_layer]:
        if "rq" not in medium:
            raise ValueError("Each rough multilayer transition must define 'rq' in the JSON stack.")
        transitions.append(
            {
                "position": position,
                "rq": float(medium["rq"]),
                "sigma_prev": previous_sigma,
                "sigma_next": float(medium["sigma"]),
            }
        )
        previous_sigma = float(medium["sigma"])
        if "thickness" in medium:
            position += float(medium["thickness"])

    return transitions


def _rough_stack_transitions_from_params(params: dict[str, object]) -> list[dict[str, float]]:
    if "base_layer" in params and "layers" in params:
        return _rough_stack_transitions_from_layers(
            params["base_layer"],
            params["layers"],
        )

    return [
        {
            "position": 0.0,
            "rq": float(params["rq01"]),
            "sigma_prev": 0.0,
            "sigma_next": float(params["sigma1"]),
        },
        {
            "position": float(params["t1"]),
            "rq": float(params["rq12"]),
            "sigma_prev": float(params["sigma1"]),
            "sigma_next": float(params["sigma2"]),
        },
    ]


def sigma_rough_stack(x: np.ndarray, transitions: list[dict[str, float]]) -> np.ndarray:
    sigma = np.zeros_like(x, dtype=float)
    for transition in transitions:
        sigma += (transition["sigma_next"] - transition["sigma_prev"]) * cdf(
            x,
            mean=transition["position"],
            rq_val=transition["rq"],
        )
    return sigma


def _normalize_profile(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    if max_value <= 0.0:
        return np.zeros_like(values)
    return values / max_value


def _skin_depth(omega: float, mu: float, sigma: float) -> float:
    return math.sqrt(2.0 / (omega * mu * sigma))


def _solve_tridiagonal_system(
    xp: np.ndarray,
    sigma: np.ndarray,
    *,
    omega: float,
    mu0: float,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = len(xp)
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
    return b_profile, j_profile


def _positive_sigma_values(*values: float) -> list[float]:
    return [value for value in values if value > 0.0]


def _rough_stack_reference_values(
    transitions: list[dict[str, float]],
) -> tuple[float, float, float, float]:
    sigma_values = _positive_sigma_values(*(t["sigma_next"] for t in transitions))
    if not sigma_values:
        raise ValueError("At least one positive conductivity is required for a rough stack.")
    base_sigma = transitions[-1]["sigma_next"]
    if base_sigma <= 0.0:
        raise ValueError("Base material conductivity must be positive for a rough stack.")
    air_facing_rq = transitions[0]["rq"]
    last_position = max(t["position"] for t in transitions)
    return base_sigma, max(sigma_values), air_facing_rq, last_position


def _rough_stack_mu_r(params: dict[str, object]) -> float:
    if "mu_r" in params:
        return float(params["mu_r"])

    media = [*params["layers"], params["base_layer"]]
    mu_real = float(media[0]["mu"][0])
    mu_imag = float(media[0]["mu"][1])
    if mu_real <= 0.0:
        raise ValueError("Rough multilayer JSON requires a positive real permeability.")
    if abs(mu_imag) > 0.0:
        raise ValueError("Rough multilayer JSON currently supports only purely real permeability.")

    for index, medium in enumerate(media[1:], start=2):
        next_real = float(medium["mu"][0])
        next_imag = float(medium["mu"][1])
        if not math.isclose(next_real, mu_real, rel_tol=0.0, abs_tol=0.0) or not math.isclose(
            next_imag, mu_imag, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                "Rough multilayer JSON currently requires the same permeability in all layers."
            )

    return mu_real


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


def _layer_field_coefficients(
    h_top: complex,
    z_top: complex,
    zeta: complex,
) -> tuple[complex, complex]:
    e_over_zeta = z_top * h_top / zeta
    a_coeff = 0.5 * (h_top + e_over_zeta)
    b_coeff = 0.5 * (e_over_zeta - h_top)
    return a_coeff, b_coeff


def _build_multilayer_profile(
    f_hz: float,
    *,
    base_layer: dict[str, object],
    layers: list[dict[str, object]],
    x_left_factor: float = -5.0,
    x_right_factor: float = 10.0,
    points_air: int = 160,
    points_per_layer: int = 240,
    points_base: int = 320,
) -> ProfileResult:
    top_to_bottom_layers = list(reversed(layers))
    total_thickness = sum(float(layer["thickness"]) for layer in top_to_bottom_layers)

    top_layer = top_to_bottom_layers[0]
    top_gamma = propagation_constant(
        np.array([f_hz]),
        tuple(top_layer["epsilon"]),
        tuple(top_layer["mu"]),
        float(top_layer["sigma"]),
        float(top_layer["tau"]),
    )[0]
    top_delta = 1.0 / max(float(np.real(top_gamma)), 1e-30)

    base_gamma = propagation_constant(
        np.array([f_hz]),
        tuple(base_layer["epsilon"]),
        tuple(base_layer["mu"]),
        float(base_layer["sigma"]),
        float(base_layer["tau"]),
    )[0]
    base_delta = 1.0 / max(float(np.real(base_gamma)), 1e-30)

    x_air = np.linspace(x_left_factor * top_delta, 0.0, points_air, endpoint=False)

    stack_from_base: list[tuple[dict[str, object], complex, complex]] = []
    z_load = waveimpedance(
        np.array([f_hz]),
        tuple(base_layer["epsilon"]),
        tuple(base_layer["mu"]),
        float(base_layer["sigma"]),
        float(base_layer["tau"]),
    )[0]
    for layer in layers:
        z_top = zs_next(
            np.array([z_load]),
            np.array([f_hz]),
            tuple(layer["epsilon"]),
            tuple(layer["mu"]),
            float(layer["sigma"]),
            float(layer["tau"]),
            float(layer["thickness"]),
        )[0]
        stack_from_base.append((layer, z_load, z_top))
        z_load = z_top

    layer_state_by_id = {
        id(layer): {"z_top": z_top_local}
        for layer, _, z_top_local in stack_from_base
    }

    h_top = 1.0 + 0.0j
    x_segments: list[np.ndarray] = [x_air]
    sigma_segments: list[np.ndarray] = [np.zeros_like(x_air, dtype=float)]
    magnetic_segments: list[np.ndarray] = [np.full_like(x_air, h_top, dtype=np.complex128)]
    power_segments: list[np.ndarray] = [np.zeros_like(x_air, dtype=float)]
    position = 0.0
    sigma_reference = max(
        [float(base_layer["sigma"]), *(float(layer["sigma"]) for layer in layers)]
    )

    for layer in top_to_bottom_layers:
        state = layer_state_by_id[id(layer)]
        zeta = waveimpedance(
            np.array([f_hz]),
            tuple(layer["epsilon"]),
            tuple(layer["mu"]),
            float(layer["sigma"]),
            float(layer["tau"]),
        )[0]
        gamma = propagation_constant(
            np.array([f_hz]),
            tuple(layer["epsilon"]),
            tuple(layer["mu"]),
            float(layer["sigma"]),
            float(layer["tau"]),
        )[0]
        local_mu = mu_eff(float(layer["mu"][0]), float(layer["mu"][1]))
        cond = np.asarray(
            conductivity_metal(np.array([f_hz]), float(layer["sigma"]), float(layer["tau"]))
        ).reshape(-1)[0]
        thickness = float(layer["thickness"])
        x_local = np.linspace(position, position + thickness, points_per_layer, endpoint=False)
        u = x_local - position

        a_coeff, b_coeff = _layer_field_coefficients(h_top, state["z_top"], zeta)
        magnetic_field = a_coeff * np.exp(-gamma * u) - b_coeff * np.exp(gamma * u)
        current_density = (gamma / local_mu) * (
            a_coeff * np.exp(-gamma * u) + b_coeff * np.exp(gamma * u)
        )

        power_loss_density = np.zeros_like(u, dtype=float)
        if np.real(1.0 / cond) > 0.0:
            power_loss_density = 0.5 * np.real(1.0 / cond) * np.abs(current_density) ** 2

        x_segments.append(x_local)
        sigma_segments.append(
            np.full_like(u, float(layer["sigma"]) / sigma_reference, dtype=float)
        )
        magnetic_segments.append(magnetic_field)
        power_segments.append(power_loss_density)

        h_top = a_coeff * np.exp(-gamma * thickness) - b_coeff * np.exp(gamma * thickness)
        position += thickness

    x_base = np.linspace(total_thickness, total_thickness + x_right_factor * base_delta, points_base)
    u_base = x_base - total_thickness
    base_mu = mu_eff(float(base_layer["mu"][0]), float(base_layer["mu"][1]))
    base_field = h_top * np.exp(-base_gamma * u_base)
    base_current_density = (base_gamma / base_mu) * base_field
    base_cond = np.asarray(
        conductivity_metal(
            np.array([f_hz]),
            float(base_layer["sigma"]),
            float(base_layer["tau"]),
        )
    ).reshape(-1)[0]
    base_power_loss_density = np.zeros_like(u_base, dtype=float)
    if np.real(1.0 / base_cond) > 0.0:
        base_power_loss_density = 0.5 * np.real(1.0 / base_cond) * np.abs(base_current_density) ** 2

    x_segments.append(x_base)
    sigma_segments.append(
        np.full_like(u_base, float(base_layer["sigma"]) / sigma_reference, dtype=float)
    )
    magnetic_segments.append(base_field)
    power_segments.append(base_power_loss_density)

    x_m = np.concatenate(x_segments)
    normalized_conductivity = np.concatenate(sigma_segments)
    magnetic_field = np.concatenate(magnetic_segments)
    power_loss_density = np.concatenate(power_segments)

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
    b_profile, j_profile = _solve_tridiagonal_system(
        xp,
        sigma,
        omega=omega,
        mu0=mu0,
        step_size=step_size,
    )

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
    b_profile, j_profile = _solve_tridiagonal_system(
        xp,
        sigma,
        omega=omega,
        mu0=mu0,
        step_size=step_size,
    )

    power_loss_density = 0.5 * np.real(1.0 / sigma_floor) * np.abs(j_profile) ** 2

    return ProfileResult(
        x_m=xp,
        normalized_conductivity=sigma / sigma_metal,
        normalized_magnetic_field=_normalize_profile(np.abs(b_profile)),
        normalized_power_loss_density=_normalize_profile(power_loss_density),
    )


def solve_tridiagonal_for_frequency_multilayer(
    f_hz: float,
    *,
    params: dict[str, object],
    xmin_factor: float,
    domain_factor: float,
    step_size: float,
) -> complex:
    omega = 2.0 * math.pi * f_hz
    mu0 = MU_0 * _rough_stack_mu_r(params)
    transitions = _rough_stack_transitions_from_params(params)
    reference_sigma, _, roughness_ref, last_position = _rough_stack_reference_values(transitions)
    delta_ref = _skin_depth(omega, mu0, reference_sigma)
    x_left = xmin_factor * roughness_ref
    x_right = last_position + domain_factor * delta_ref

    xp = np.arange(x_left, x_right + step_size, step_size)
    sigma = sigma_rough_stack(xp, transitions)
    b_profile, j_profile = _solve_tridiagonal_system(
        xp,
        sigma,
        omega=omega,
        mu0=mu0,
        step_size=step_size,
    )

    mask = sigma > 0
    integral_b = np.trapezoid(b_profile[mask], xp[mask])
    integral_j = np.trapezoid(j_profile[mask], xp[mask])
    return (-1j * omega * integral_b) / integral_j


def solve_tridiagonal_profile_multilayer(
    f_hz: float,
    *,
    params: dict[str, object],
    xmin_factor: float,
    domain_factor: float,
    step_size: float,
) -> ProfileResult:
    omega = 2.0 * math.pi * f_hz
    mu0 = MU_0 * _rough_stack_mu_r(params)
    transitions = _rough_stack_transitions_from_params(params)
    reference_sigma, sigma_ref, roughness_ref, last_position = _rough_stack_reference_values(
        transitions
    )
    delta_ref = _skin_depth(omega, mu0, reference_sigma)
    x_left = xmin_factor * roughness_ref
    x_right = last_position + domain_factor * delta_ref

    xp = np.arange(x_left, x_right + step_size, step_size)
    sigma = sigma_rough_stack(xp, transitions)
    sigma_floor = np.maximum(sigma, 1e-16)
    b_profile, j_profile = _solve_tridiagonal_system(
        xp,
        sigma,
        omega=omega,
        mu0=mu0,
        step_size=step_size,
    )
    power_loss_density = 0.5 * np.real(1.0 / sigma_floor) * np.abs(j_profile) ** 2

    return ProfileResult(
        x_m=xp,
        normalized_conductivity=sigma / sigma_ref,
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


def _rough_multi_model(freq_hz: np.ndarray, params: dict[str, object]) -> np.ndarray:
    show_progress = bool(params.get("_progress", False))
    progress_label = str(params.get("_progress_label", "rough-multi"))
    total = len(freq_hz)
    values: list[complex] = []

    for index, freq in enumerate(freq_hz, start=1):
        values.append(
            solve_tridiagonal_for_frequency_multilayer(
                f_hz=float(freq),
                params=params,
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
    "rough-multi": ModelSpec(
        name="rough-multi",
        description="Surface roughness gradient model for multiple rough transitions, optionally loaded from JSON.",
        parameter_help={
            "layers_file": "JSON file with base_layer + layers, each transition medium carrying an 'rq' value.",
            "sigma1": "Legacy two-layer mode: conductivity of material 1 in S/m.",
            "sigma2": "Legacy two-layer mode: conductivity of material 2 in S/m.",
            "rq01": "Legacy two-layer mode: RMS roughness of the vacuum/material-1 transition in m.",
            "rq12": "Legacy two-layer mode: RMS roughness of the material-1/material-2 transition in m.",
            "t1": "Legacy two-layer mode: mean thickness of material 1 in m.",
            "mu_r": "Relative permeability used in legacy mode, or inferred from JSON if omitted there.",
            "step_size": "Finite-difference step size in m.",
            "xmin_factor": "Left-domain extent factor relative to the largest roughness.",
            "domain_factor": "Right-domain extent in skin-depth units of the more penetrable conductor.",
        },
        evaluator=_rough_multi_model,
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
    if model_name == "multi-layer":
        return _build_multilayer_profile(
            frequency_hz,
            base_layer=params["base_layer"],
            layers=params["layers"],
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
    if model_name == "rough-multi":
        return solve_tridiagonal_profile_multilayer(
            frequency_hz,
            params=params,
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
            "sigma1",
            "sigma2",
            "mu_r",
            "epsr_real",
            "mur_real",
            "rq",
            "rq01",
            "rq12",
            "t1",
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
        if "rq" in base_layer and float(base_layer["rq"]) <= 0.0:
            raise ValueError("Base layer parameter 'rq' must be positive.")

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
            if "rq" in layer and float(layer["rq"]) <= 0.0:
                raise ValueError(f"Layer {index} parameter 'rq' must be positive.")
