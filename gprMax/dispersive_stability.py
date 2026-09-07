# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Pre-run timestep checks for the current-density dispersive recurrence.

The test is a sufficient material/Yee-curl certificate, not a proof of the
stability of PML, subgrid, or source-device coupling. It evaluates all poles
together, including Drude's effective conductivity. No update coefficients,
material parameters, or timesteps are changed by this module.
"""

from dataclasses import dataclass

import numpy as np

import gprMax.config as config


@dataclass(frozen=True)
class StabilityCheck:
    """Dimensionless margins for one material and a spatial upper bound."""

    reason: str | None
    nyquist_permittivity: float = float("nan")
    spatial_bound: float = float("nan")
    passivity_margin: float = float("nan")

    @property
    def passed(self):
        return self.reason is None


def _conductance_deficit(q, residue):
    """Conductance sufficient to make s*Re_coeff[C/(s-q)] positive real.

    Re_coeff conjugates coefficients, not the complex argument. For a damped
    conjugate pair, minimise its boundary real part over x=(omega/abs(q))**2.
    Scaling both the pole and residue avoids unnecessarily large polynomial
    coefficients; the denominator is evaluated as a sum of non-negative terms.
    """
    if q.imag == 0:
        return max(0.0, -float(residue.real))
    scale = abs(residue)
    if scale == 0:
        return 0.0
    if q.real == 0:
        # Imaginary-axis poles must have real, non-negative admittance
        # residues. A finite added conductance cannot repair a wrong residue.
        if residue.real == 0 and -residue.imag * q.imag >= 0:
            return 0.0
        return float("inf")

    eta = -q.real / abs(q)
    beta = q.imag / abs(q)
    c, d = residue.real / scale, residue.imag / scale
    b = 2 * eta * (c * eta - d * beta) - c
    if c >= 0 and b >= 0:
        return 0.0
    denominator_linear = 4 * eta * eta - 2
    aa, bb, cc = c * denominator_linear - b, 2 * c, b
    if aa == 0:
        roots = [] if bb == 0 else [-cc / bb]
    else:
        discriminant = bb * bb - 4 * aa * cc
        roots = []
        if discriminant >= 0:
            # Avoid cancellation in the root closest to zero.
            numerator = -0.5 * (bb + np.copysign(np.sqrt(discriminant), bb))
            if numerator != 0:
                roots = [numerator / aa, cc / numerator]
            else:
                roots = [0.0]
    minimum = min(0.0, c)  # Values at zero and infinity.
    for root in roots:
        if not np.isfinite(root) or root <= 0:
            continue
        x = float(root)
        if x > 1:
            inverse = 1 / x
            numerator = c + b * inverse
            denominator = (1 - inverse) ** 2 + 4 * eta * eta * inverse
        else:
            numerator = c * x * x + b * x
            denominator = (1 - x) ** 2 + 4 * eta * eta * x
        if denominator == 0:
            return float("inf")
        minimum = min(minimum, numerator / denominator)
    return -minimum * scale


def check_dispersive_timestep(er, conductivity, terms, dt, spatial_rate, *, e0=config.e0):
    """Check one SI material law at a given, unchanged timestep.

    ``terms`` contains (W, Q) in s^-1. ``conductivity`` includes any Drude
    equivalent conductivity in S/m. ``spatial_rate * dt**2`` bounds the Yee
    curl coupling and uses a common lower bound on magnetic permeability,
    not this electric material's permeability alone.

    With a=exp(Q*dt), q=(a-1)/(a+1), the numerical permittivity at z=-1 is
    er-sum(Re[(W/Q)*(1-sech(Q*dt/2))]). The residual admittance is positive
    real when its available conductance covers the sum of individual pole
    deficits. That last test is conservative for arbitrary fitted mixtures.
    """
    parameters = (er, conductivity, dt, spatial_rate, e0)
    if not all(np.isfinite(value) for value in parameters):
        return StabilityCheck("non-finite material or timestep parameters")
    if er <= 0 or conductivity < 0 or dt <= 0 or spatial_rate < 0 or e0 <= 0:
        return StabilityCheck("invalid permittivity, conductivity, or timestep parameters")

    nyquist = float(er)
    instantaneous = float(er)
    conductance = conductivity * dt / (2 * e0)
    if not np.isfinite(conductance):
        return StabilityCheck("non-finite dimensionless conductivity")
    deficits = []
    residues = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for w, q_physical in terms:
            w, q_physical = complex(w), complex(q_physical)
            if not np.isfinite(w) or not np.isfinite(q_physical):
                return StabilityCheck("non-finite exponential material pole or residue")
            if q_physical.real > 0:
                return StabilityCheck("a material pole has a positive growth rate")
            if w == 0:
                continue
            if q_physical == 0:
                return StabilityCheck("the Q=0 exponential-pole representation is undefined")
            # For a real pole, an imaginary residue has no physical real part.
            if q_physical.imag == 0:
                w = complex(w.real)
            increment = q_physical * dt
            a, half = np.exp(increment), np.exp(increment / 2)
            denominator = 1 + a
            if abs(denominator) < 1e-10:
                return StabilityCheck("a pole is too close to the temporal Nyquist singularity")
            ratio = w / q_physical
            difference = -np.expm1(increment / 2)
            # (1-exp(x/2))**2/(1+exp(x)) = 1-sech(x/2), without subtracting
            # nearly equal numbers for slow poles.
            nyquist -= float(np.real(ratio * difference**2 / denominator))
            instantaneous -= float(np.real(ratio * difference))
            if q_physical.real == 0:
                q = 1j * np.tan(increment.imag / 2)
                residue = ratio * q / np.cos(increment.imag / 2)
            else:
                # Evaluate the small negative real part without cancellation
                # against a possibly much larger imaginary part.
                norm = abs(denominator) ** 2
                q = np.expm1(2 * increment.real) / norm + 2j * a.imag / norm
                residue = ratio * q * (2 * half / denominator)
            if not np.isfinite(q) or not np.isfinite(residue) or not np.isfinite(abs(residue)):
                return StabilityCheck("non-finite transformed material coefficients")
            if q.real > 0:
                return StabilityCheck("a transformed material pole has a positive growth rate")
            deficits.append(_conductance_deficit(q, residue))
            residues.append(abs(residue))

    bound = spatial_rate * dt**2
    margin = conductance - sum(deficits)
    if not np.isfinite(bound) or not np.isfinite(sum(residues)):
        return StabilityCheck(
            "non-finite spatial bound or accumulated residues", nyquist, bound, margin
        )
    if not np.isfinite(nyquist) or not np.isfinite(instantaneous + conductance):
        return StabilityCheck("non-finite electric update denominator", nyquist, bound, margin)
    if instantaneous + conductance <= 0:
        return StabilityCheck("non-positive electric update denominator", nyquist, bound, margin)
    # This tolerance only allows roundoff in a positive-real identity. A
    # strictly positive storage margin is still required; no CFL violation
    # is accepted using this tolerance.
    tolerance = 64 * np.finfo(float).eps * max(np.finfo(float).tiny, conductance, sum(residues))
    if not np.isfinite(margin) or margin < -tolerance:
        return StabilityCheck(
            "the discrete material passivity check cannot certify this timestep",
            nyquist,
            bound,
            margin,
        )
    if nyquist < 0:
        return StabilityCheck(
            "negative numerical Nyquist permittivity: the homogeneous material update "
            "has a growing zero-curl mode",
            nyquist,
            bound,
            margin,
        )
    if nyquist <= bound:
        return StabilityCheck(
            "the conservative dispersive-material CFL bound is not satisfied",
            nyquist,
            bound,
            margin,
        )
    return StabilityCheck(None, nyquist, bound, margin)


def _material_inputs(material, e0):
    from gprMax.materials import _inclusive_material_terms

    terms, extra_conductivity = _inclusive_material_terms(material)
    if not material.inclusive_w and "drude" in material.type:
        # The shared conversion helper expresses this term with config.e0;
        # native coefficient generation uses the simulation's constant.
        extra_conductivity *= e0 / config.e0
    return float(material.er), float(material.se + extra_conductivity), terms


def _failure_message(grid, material, inputs, check, spatial_rate, e0):
    message = (
        f"Dispersive timestep check failed on grid {getattr(grid, 'name', 'grid')!r}, "
        f"material {material.ID!r}: {check.reason}. "
        f"dt={grid.dt:.9g} s; numerical Nyquist permittivity="
        f"{check.nyquist_permittivity:.9g}; spatial bound={check.spatial_bound:.9g}. "
        "The simulation has been stopped; dt and material parameters were not changed. "
    )
    # Give a checked smaller step, not an unproved 'maximum stable dt'.
    # Halving also avoids assuming monotonicity for arbitrary fitted residues.
    for power in range(1, 25):
        ratio = 2.0**-power
        candidate = float(grid.dt) * ratio
        if check_dispersive_timestep(*inputs, candidate, spatial_rate, e0=e0).passed:
            message += (
                f"For this material, a checked smaller candidate is dt={candidate:.9g} s. "
                f"Multiply your current #time_step_stability_factor (API: "
                f"TimeStepStabilityFactor(f=...)) by {ratio:.9g}, then rerun the checks "
                "for all materials. "
            )
            break
    else:
        message += (
            "No smaller timestep was certified in the 24 halvings checked. "
            "Review the pole representation/passivity as well as the timestep. "
        )
    return message + (
        "Check material units and parameters. Changing physical parameters changes "
        "the model; do not change them solely to bypass this check. Failure of a "
        "conservative certificate is not by itself proof that a finite model is unstable."
    )


def validate_grid_dispersive_timestep(grid):
    """Stop before solving if a defined dispersive material is not certified.

    Called after component materials are final, on every grid. MPI ranks
    exchange only scalar bounds/diagnostics during setup, including ranks
    with no local dispersive terms; no collectives or checks enter the loop.
    Geometry-fixed runs retain the validated coefficients and timestep.
    """
    comm = getattr(grid, "comm", None)
    materials = [
        material
        for material in grid.materials
        if getattr(material, "poles", 0) > 0
        and material.se != float("inf")
        and material.sm != float("inf")
    ]
    if comm is None and not materials:
        return

    error = None
    e0 = config.sim_config.em_consts["e0"]
    m0 = config.sim_config.em_consts["m0"]
    minimum_mr = float("inf")
    try:
        minimum_mr = min(
            float(material.mr) for material in grid.materials if material.sm != float("inf")
        )
        if not np.isfinite(minimum_mr) or minimum_mr <= 0:
            raise ValueError("finite, positive magnetic permeability is required")
    except (ValueError, TypeError) as exc:
        error = str(exc)

    if comm is not None:
        setup = comm.allgather((bool(materials), minimum_mr, error))
        if not any(item[0] for item in setup):
            return
        minimum_mr = min(item[1] for item in setup)
        error = next((item[2] for item in setup if item[2] is not None), None)
    if error is not None:
        raise ValueError(f"Cannot check dispersive timestep: {error}")

    try:
        mode = config.get_model_config().mode
        invariant = mode[-1] if mode.startswith("2D ") else None
        steps = (grid.dx, grid.dy, grid.dz)
        spatial_rate = sum(
            1.0 / step**2 for axis, step in zip("xyz", steps) if axis != invariant
        ) / (e0 * m0 * minimum_mr)
        for material in materials:
            inputs = _material_inputs(material, e0)
            check = check_dispersive_timestep(*inputs, grid.dt, spatial_rate, e0=e0)
            if not check.passed:
                error = _failure_message(grid, material, inputs, check, spatial_rate, e0)
                break
    except (ValueError, TypeError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        error = f"Cannot check dispersive timestep: {exc}"
    if comm is not None:
        errors = comm.allgather(error)
        error = next((message for message in errors if message is not None), None)
    if error is not None:
        raise ValueError(error)
