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

"""Continuum material spectra and pre-solve spatial-resolution diagnostics.

The lossless grid-axis estimate follows Schneider, Understanding the FDTD
Method, chapter 7, equations 7.39--7.43. It is not a stability certificate or
a dispersion model for the recursive dispersive updates, PML, or interfaces.
The convention is exp(+j*omega*t); a forward passive wave is exp(-j*k*x),
with k = beta - j*alpha and alpha >= 0.
"""

import numpy as np

from gprMax import config
from gprMax.mode2d import mode2d_geometry


def complex_relative_permittivity(material, frequencies):
    """Return relative permittivity including electric conductivity once."""
    values = np.empty(frequencies.shape, dtype=np.complex128)
    zero = frequencies == 0
    values[zero] = complex(material.er)
    positive = ~zero
    if not np.any(positive):
        return values
    if hasattr(material, "poles"):
        values[positive] = np.asarray(
            material.calculate_er(frequencies[positive]), dtype=np.complex128
        )
    else:
        omega = 2 * np.pi * frequencies[positive]
        values[positive] = material.er + material.se / (
            1j * omega * config.sim_config.em_consts["e0"]
        )
    return values


def complex_relative_permeability(material, frequencies):
    """Return relative permeability including magnetic conductivity once."""
    values = np.empty(frequencies.shape, dtype=np.complex128)
    zero = frequencies == 0
    values[zero] = complex(material.mr)
    positive = ~zero
    if np.any(positive):
        omega = 2 * np.pi * frequencies[positive]
        values[positive] = material.mr + material.sm / (
            1j * omega * config.sim_config.em_consts["m0"]
        )
    return values


def active_spatial_steps(grid, mode):
    """Return (axis name, spacing in m) pairs, excluding a 2-D invariant axis."""
    geometry = mode2d_geometry(mode)
    return tuple(
        ("xyz"[axis], float(step))
        for axis, step in enumerate((grid.dx, grid.dy, grid.dz))
        if geometry is None or axis != geometry.invariant_axis
    )


def diagnostic_frequencies(materials, maximum_frequency):
    """Sample the positive source band and known relaxation/resonance scales.

    This is a sampled-band diagnostic, not an optimisation proving a global
    extremum for an arbitrary rational fit. Pole-centred samples avoid testing
    only the upper band edge, which can miss a Lorentz resonance entirely.
    """
    high = float(maximum_frequency)
    if not np.isfinite(high) or high <= 0:
        raise ValueError("a finite positive maximum frequency is required")
    samples = [*np.linspace(high / 128, high, 128), *np.geomspace(high * 1e-6, high, 65)]
    for material in materials:
        if not getattr(material, "poles", 0):
            continue
        if getattr(material, "inclusive_q", None):
            centres = [(abs(complex(q).imag), abs(complex(q).real)) for q in material.inclusive_q]
        elif "lorentz" in material.type:
            centres = [
                (2 * np.pi * f, damping) for f, damping in zip(material.tau, material.alpha)
            ]
        elif "debye" in material.type:
            centres = [(1 / tau, 0) for tau in material.tau]
        else:
            centres = [(2 * np.pi * f, 0) for f in material.tau]
        for centre, width in centres:
            for offset in (-2, -1, -0.5, 0, 0.5, 1, 2):
                frequency = (centre + offset * width) / (2 * np.pi)
                if 0 < frequency <= high:
                    samples.append(frequency)
    return np.unique(samples)


def material_propagation(material, frequencies):
    """Return isotropic continuum k in rad/m, including loss and dispersion.

    Directional material records must not be interpreted as their scalar mean.
    Zero frequency is excluded: neither a phase velocity nor a wavelength at
    DC is needed by the pre-solve diagnostic.
    """
    if getattr(material, "directional_materials", None) is not None:
        raise ValueError("a directional material requires a tensor propagation calculation")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if np.any(frequencies <= 0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("propagation frequencies must be finite and positive")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        epsilon = complex_relative_permittivity(material, frequencies)
        mu = complex_relative_permeability(material, frequencies)
        index = np.sqrt(epsilon) * np.sqrt(mu)
        # Choose the attenuating branch, including purely evanescent media.
        index = np.where(index.imag > 0, -index, index)
        k = (2 * np.pi * frequencies / config.sim_config.em_consts["c"]) * index
    if not np.all(np.isfinite(k)):
        raise ValueError(
            f"material {material.ID!r} has a singular/non-finite response in the sampled band; "
            "check pole frequencies, damping and units"
        )
    return k


def spatial_resolution(grid, maximum_frequency, mode):
    """Return sampled continuum resolution and a scoped lossless phase error.

    N measures phase-wavelength sampling. Attenuation uses the separate 1/e
    amplitude-decay length. For diagonal anisotropy only a conservative index
    magnitude bound is returned, not an invented scalar phase velocity.
    """
    steps = active_spatial_steps(grid, mode)
    axis, delta = max(steps, key=lambda item: item[1])
    materials = [
        m
        for m in grid.materials
        if np.isfinite(m.se)
        and np.isfinite(m.sm)
        and not getattr(m, "is_pec", False)
        and not getattr(m, "is_pmc", False)
        and "voltage-source" not in str(getattr(m, "type", "")).lower()
    ]
    if not materials:
        raise ValueError("no finite bulk material is available for spatial-resolution analysis")
    frequencies = diagnostic_frequencies(materials, maximum_frequency)
    result = dict(
        N=float("inf"),
        material=materials[0],
        sampling_frequency=float(maximum_frequency),
        wavelength=float("inf"),
        phase_velocity=None,
        sampling_kind="phase wavelength",
        attenuation_cells=float("inf"),
        attenuation_material=None,
        attenuation_frequency=None,
        deltavp=None,
        phase_error_material=None,
        phase_error_frequency=None,
        phase_error_axis=axis,
        phase_error_notes=[],
    )
    c = config.sim_config.em_consts["c"]
    for material in materials:
        directional = getattr(material, "directional_materials", None)
        if directional is not None:
            if any(
                m.er <= 0 or m.mr <= 0 or m.se != 0 or m.sm != 0 or getattr(m, "poles", 0)
                for m in directional
            ):
                result["phase_error_notes"].append(
                    f"{material.ID}: full tensor wavelength/attenuation analysis is unavailable; "
                    "scalar constituent checks do not certify this anisotropic medium"
                )
                continue
            # ||epsilon||*||mu|| bounds the diagonal constitutive operator.
            # This bound requires positive, lossless tensors. Hyperbolic or
            # complex tensors cannot be certified using their largest entries.
            epsilon = np.asarray(
                [complex_relative_permittivity(m, frequencies) for m in directional]
            )
            mu = np.asarray([complex_relative_permeability(m, frequencies) for m in directional])
            bound = np.sqrt(np.max(np.abs(epsilon), axis=0) * np.max(np.abs(mu), axis=0))
            beta = 2 * np.pi * frequencies / c * bound
            if not np.all(np.isfinite(beta)):
                raise ValueError(f"material {material.ID!r} has a non-finite directional response")
            alpha = np.zeros_like(beta)
            kind = "anisotropic index-magnitude bound"
        else:
            k = material_propagation(material, frequencies)
            beta, alpha = np.abs(k.real), np.maximum(0, -k.imag)
            kind = "phase wavelength"
        wavelength = np.full(frequencies.shape, np.inf)
        np.divide(2 * np.pi, beta, out=wavelength, where=beta > 0)
        cells = wavelength / delta
        index = int(np.argmin(cells))
        if cells[index] < result["N"]:
            result.update(
                N=float(cells[index]),
                material=material,
                sampling_frequency=float(frequencies[index]),
                wavelength=float(wavelength[index]),
                phase_velocity=None
                if directional is not None
                else float(2 * np.pi * frequencies[index] / k.real[index]),
                sampling_kind=kind,
            )
        attenuation = np.full(frequencies.shape, np.inf)
        np.divide(1, alpha * delta, out=attenuation, where=alpha > 0)
        index = int(np.argmin(attenuation))
        if attenuation[index] < result["attenuation_cells"]:
            result.update(
                attenuation_cells=float(attenuation[index]),
                attenuation_material=material,
                attenuation_frequency=float(frequencies[index]),
            )
        if (
            directional is not None
            or material.se != 0
            or material.sm != 0
            or getattr(material, "poles", 0)
        ):
            note = "lossy, dispersive or anisotropic materials: numerical phase-error estimate unavailable"
            if note not in result["phase_error_notes"]:
                result["phase_error_notes"].append(note)
            continue
        if material.er <= 0 or material.mr <= 0:
            result["phase_error_notes"].append(
                f"{material.ID}: no positive lossless bulk wave speed"
            )
            continue
        velocity = c / np.sqrt(material.er * material.mr)
        argument = delta / (velocity * grid.dt) * np.sin(np.pi * frequencies * grid.dt)
        resolved = (frequencies < 0.5 / grid.dt) & (argument > 0) & (argument <= 1)
        if np.any(~resolved):
            result["phase_error_notes"].append(
                f"{material.ID}: sampled frequencies reach temporal/spatial cutoff"
            )
        if np.any(resolved):
            # beta / beta_num = v_num / v; do not mix N in the material with
            # a vacuum Courant factor. This is an axial homogeneous estimate.
            ratio = (np.pi * frequencies[resolved] * delta / velocity) / np.arcsin(
                argument[resolved]
            )
            errors = (ratio - 1) * 100
            index = int(np.argmax(np.abs(errors)))
            if result["deltavp"] is None or abs(errors[index]) > abs(result["deltavp"]):
                result.update(
                    deltavp=float(errors[index]),
                    phase_error_material=material,
                    phase_error_frequency=float(frequencies[resolved][index]),
                )
    return result
