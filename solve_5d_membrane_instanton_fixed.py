#!/usr/bin/env python3
"""
Static 5D black-hole-seeded thin-wall membrane instanton solver.

Inputs use natural units. The five-dimensional Einstein-Hilbert normalization is
    S = ∫ d^5x sqrt(-g) (M5^3/2) R
so G5 = 1/(8*pi*M5^3).

The solver searches for an ordinary-orientation static saddle satisfying
    sqrt(f_minus) - sqrt(f_plus) = kappa * R
and the radial force-balance equation.

It then computes
    B = [A_h(plus) - A_h(minus)]/(4 G5).

This is a local 5D solver. It is not valid when the wall or horizons approach
the compactification radius or an orbifold endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class InstantonInput:
    M5: float
    sigma: float
    rho_plus: float
    rho_minus: float
    M_plus: float
    R_DD: Optional[float] = None
    wall_thickness: Optional[float] = None


@dataclass(frozen=True)
class InstantonResult:
    R_s: float
    M_minus: float
    mu_plus: float
    mu_minus: float
    r_h_plus: float
    r_h_minus: float
    B_static: float
    F_second: float
    local_5d_valid: Optional[bool]


def black_hole_horizon(mu: float, Lambda: float) -> float:
    """Return the smallest positive black-hole horizon of f=1-mu/r^2-Lambda*r^2/6."""
    if mu < 0:
        raise ValueError("mu must be nonnegative for the ordinary black-hole branch.")

    # Let x=r^2. Solve (Lambda/6)x^2 - x + mu = 0.
    if abs(Lambda) < 1e-30:
        if mu <= 0:
            return 0.0
        return sqrt(mu)

    roots = np.roots([Lambda / 6.0, -1.0, mu])
    positive = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-9 and root.real > 0
    )
    if not positive:
        raise ValueError("No positive horizon root exists.")

    # For de Sitter there can be a black-hole and cosmological horizon;
    # the smaller positive root is the black-hole horizon.
    return sqrt(positive[0])


def solve_static_instanton(
    p: InstantonInput,
    radius_guesses: int = 30,
    mass_guesses: int = 25,
) -> InstantonResult:
    if p.M5 <= 0 or p.sigma <= 0 or p.M_plus <= 0:
        raise ValueError("M5, sigma, and M_plus must be positive.")

    G5 = 1.0 / (8.0 * pi * p.M5**3)
    Lambda_plus = p.rho_plus / p.M5**3
    Lambda_minus = p.rho_minus / p.M5**3
    mu_plus = 8.0 * G5 * p.M_plus / (3.0 * pi)
    kappa = p.sigma / (3.0 * p.M5**3)

    r_h_plus = black_hole_horizon(mu_plus, Lambda_plus)

    def f(r: float, mu: float, Lambda: float) -> float:
        return 1.0 - mu / r**2 - Lambda * r**2 / 6.0

    def fp(r: float, mu: float, Lambda: float) -> float:
        return 2.0 * mu / r**3 - Lambda * r / 3.0

    def residual(log_vars: np.ndarray) -> np.ndarray:
        R = np.exp(log_vars[0])
        mu_minus = np.exp(log_vars[1])
        f_plus = f(R, mu_plus, Lambda_plus)
        f_minus = f(R, mu_minus, Lambda_minus)

        # Penalize guesses inside a horizon or outside a static region.
        if f_plus <= 0 or f_minus <= 0:
            penalty = 10.0 + abs(min(f_plus, 0.0)) + abs(min(f_minus, 0.0))
            return np.array([penalty, penalty])

        root_plus = sqrt(f_plus)
        root_minus = sqrt(f_minus)
        eq1 = root_minus - root_plus - kappa * R
        eq2 = (
            fp(R, mu_minus, Lambda_minus) / (2.0 * root_minus)
            - fp(R, mu_plus, Lambda_plus) / (2.0 * root_plus)
            - kappa
        )
        return np.array([eq1, R * eq2])

    R_min = r_h_plus * 1.001
    scale = max(r_h_plus, 1.0 / max(kappa, 1e-300))
    R_max = scale * 1e4
    R_grid = np.geomspace(R_min, R_max, radius_guesses)
    mu_grid = np.geomspace(max(mu_plus * 1e-8, 1e-300), mu_plus * 10.0, mass_guesses)

    candidates = []
    for R0 in R_grid:
        for mu0 in mu_grid:
            fit = least_squares(
                residual,
                x0=np.log([R0, mu0]),
                max_nfev=3000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            norm = float(np.linalg.norm(fit.fun))
            if fit.success and norm < 1e-7:
                R_s, mu_minus = np.exp(fit.x)
                try:
                    r_h_minus = black_hole_horizon(mu_minus, Lambda_minus)
                except ValueError:
                    continue
                if R_s <= max(r_h_plus, r_h_minus):
                    continue
                candidates.append((norm, R_s, mu_minus, r_h_minus))

    if not candidates:
        raise RuntimeError(
            "No ordinary-orientation static saddle was found. "
            "Try a wider scan, a different orientation, or a periodic bounce."
        )

    # Remove near-duplicates and choose the candidate with the smallest residual.
    candidates.sort(key=lambda item: item[0])
    _, R_s, mu_minus, r_h_minus = candidates[0]

    def F(R: float) -> float:
        f_plus = f(R, mu_plus, Lambda_plus)
        f_minus = f(R, mu_minus, Lambda_minus)
        fbar = 0.5 * (f_plus + f_minus)
        delta_f = f_plus - f_minus
        return (
            (kappa**2 * R**2) / 4.0
            - fbar
            + delta_f**2 / (4.0 * kappa**2 * R**2)
        )

    # Five-point finite-difference second derivative.
    h = max(1e-5 * R_s, 1e-12)
    F_second = (
        -F(R_s + 2*h)
        + 16*F(R_s + h)
        - 30*F(R_s)
        + 16*F(R_s - h)
        - F(R_s - 2*h)
    ) / (12*h**2)

    M_minus = 3.0 * pi * mu_minus / (8.0 * G5)
    B_static = (pi**2 / (2.0 * G5)) * (r_h_plus**3 - r_h_minus**3)

    local_valid: Optional[bool] = None
    if p.R_DD is not None:
        scales = [R_s, r_h_plus, r_h_minus]
        if p.wall_thickness is not None:
            scales.append(p.wall_thickness)
        local_valid = max(scales) < 0.1 * p.R_DD

    if B_static <= 0:
        raise RuntimeError(
            "The selected solution has nonpositive entropy difference. "
            "Check the horizon choice, branch direction, and normal orientation."
        )

    return InstantonResult(
        R_s=R_s,
        M_minus=M_minus,
        mu_plus=mu_plus,
        mu_minus=mu_minus,
        r_h_plus=r_h_plus,
        r_h_minus=r_h_minus,
        B_static=B_static,
        F_second=F_second,
        local_5d_valid=local_valid,
    )


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Solve the ordinary-orientation static 5D membrane instanton. "
            "Run with no model arguments to execute a verified dimensionless self-test."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--M5", type=float, default=None, help="Reduced 5D Planck scale.")
    parser.add_argument("--sigma", type=float, default=None, help="Effective 3-brane tension.")
    parser.add_argument(
        "--rho-plus", type=float, default=None,
        help="Relaxed 5D parent-branch energy density."
    )
    parser.add_argument(
        "--rho-minus", type=float, default=None,
        help="Relaxed 5D daughter-branch energy density."
    )
    parser.add_argument("--M-plus", type=float, default=None, help="Parent ADM mass.")
    parser.add_argument("--R-DD", type=float, default=None, help="Compactification radius.")
    parser.add_argument(
        "--wall-thickness", type=float, default=None,
        help="Physical wall thickness for the local-5D validity check."
    )
    parser.add_argument(
        "--radius-guesses", type=int, default=30,
        help="Number of logarithmic initial guesses for the wall radius."
    )
    parser.add_argument(
        "--mass-guesses", type=int, default=25,
        help="Number of logarithmic initial guesses for the remnant mass parameter."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the result as JSON instead of a formatted report."
    )
    args = parser.parse_args()

    required_names = ("M5", "sigma", "rho_plus", "rho_minus", "M_plus")
    supplied = {name: getattr(args, name) for name in required_names}
    supplied_count = sum(value is not None for value in supplied.values())

    if supplied_count == 0:
        # Verified dimensionless self-test. These values are chosen so that an
        # ordinary static saddle exists near R_s = 1 and B_static > 0.
        params = InstantonInput(
            M5=1.0,
            sigma=0.3,
            rho_plus=0.8,
            rho_minus=0.0,
            M_plus=18.257,
            R_DD=args.R_DD,
            wall_thickness=args.wall_thickness,
        )
        mode = "verified dimensionless self-test"
    elif supplied_count != len(required_names):
        missing = [name.replace("_", "-") for name, value in supplied.items() if value is None]
        parser.error(
            "either provide all five model parameters or none of them. "
            "Missing: " + ", ".join("--" + name for name in missing)
        )
    else:
        params = InstantonInput(
            M5=args.M5,
            sigma=args.sigma,
            rho_plus=args.rho_plus,
            rho_minus=args.rho_minus,
            M_plus=args.M_plus,
            R_DD=args.R_DD,
            wall_thickness=args.wall_thickness,
        )
        mode = "user-supplied model"

    try:
        result = solve_static_instanton(
            params,
            radius_guesses=args.radius_guesses,
            mass_guesses=args.mass_guesses,
        )
    except Exception as exc:
        raise SystemExit(
            "No valid static saddle was found.\n"
            f"Reason: {exc}\n\n"
            "This does not necessarily mean that the transition is forbidden. "
            "The relevant solution may use a different wall orientation, a wider "
            "initial-guess scan, or a nonstatic periodic bounce."
        ) from exc

    payload = {
        "mode": mode,
        "inputs": {
            "M5": params.M5,
            "sigma": params.sigma,
            "rho_plus": params.rho_plus,
            "rho_minus": params.rho_minus,
            "M_plus": params.M_plus,
            "R_DD": params.R_DD,
            "wall_thickness": params.wall_thickness,
        },
        "result": {
            "R_s": float(result.R_s),
            "M_minus": float(result.M_minus),
            "mu_plus": float(result.mu_plus),
            "mu_minus": float(result.mu_minus),
            "r_h_plus": float(result.r_h_plus),
            "r_h_minus": float(result.r_h_minus),
            "B_static": float(result.B_static),
            "F_second": float(result.F_second),
            "negative_mode_proxy": bool(result.F_second > 0),
            "local_5d_valid": result.local_5d_valid,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Mode: {mode}")
        if mode.startswith("verified"):
            print(
                "Note: the self-test values are dimensionless validation inputs, "
                "not predictions of the physical model."
            )
        print("\nInputs")
        for key, value in payload["inputs"].items():
            print(f"  {key:16s} = {value}")
        print("\nStatic instanton")
        for key in (
            "R_s", "M_minus", "mu_plus", "mu_minus",
            "r_h_plus", "r_h_minus", "B_static", "F_second"
        ):
            print(f"  {key:16s} = {payload['result'][key]:.12g}")
        print(
            "  negative mode    = "
            + str(payload["result"]["negative_mode_proxy"])
        )
        print(
            "  local 5D valid   = "
            + str(payload["result"]["local_5d_valid"])
        )
