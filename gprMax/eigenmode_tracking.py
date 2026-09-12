"""Degenerate modal groups with a shared physical E/H basis."""

import logging
import numbers
from copy import copy

import numpy as np

_CONDITION_LIMIT = 1e8
_DEGENERACY_TOLERANCE = 1e-8
_RESIDUAL_TOLERANCE = 1e-9
logger = logging.getLogger(__name__)


def _index(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError("Degenerate mode labels must be positive integers.")
    return int(value)


def normalize_groups(value, modes):
    if value is None:
        return ()
    try:
        groups = tuple(value)
    except TypeError as exc:
        raise ValueError("degenerate must contain groups of mode indices.") from exc
    if groups and all(isinstance(item, numbers.Integral) for item in groups):
        groups = (groups,)
    result, seen = [], set()
    for group in groups:
        try:
            group = tuple(_index(item) for item in group)
        except TypeError as exc:
            raise ValueError("degenerate must contain groups of mode indices.") from exc
        if len(group) < 2 or len(set(group)) != len(group):
            raise ValueError("Each degenerate group requires at least two distinct modes.")
        if seen.intersection(group) or not set(group).issubset(modes):
            raise ValueError("Degenerate groups must be disjoint and fully included in modes.")
        seen.update(group)
        result.append(group)
    return tuple(result)


def normalize_polarizations(value, groups, normal_axis, invariant_axis=None):
    if value is None:
        return {}
    if not hasattr(value, "items"):
        raise ValueError("mode_polarizations must map mode labels to axes or real vectors.")
    if value and invariant_axis is not None:
        raise ValueError("Physical mode_polarizations require a 3D port cross-section.")
    result = {}
    for mode, direction in value.items():
        mode = _index(mode)
        if isinstance(direction, str):
            if direction.lower() not in ("x", "y", "z"):
                raise ValueError("Mode polarization axes must be x, y, or z.")
            direction = np.eye(3)["xyz".index(direction.lower())]
        raw = np.asarray(direction)
        if np.iscomplexobj(raw):
            raise ValueError("Mode polarization directions must be real vectors.")
        vector = np.asarray(raw, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not np.any(vector):
            raise ValueError("Mode polarization directions must be finite nonzero three-vectors.")
        vector = vector / np.max(np.abs(vector))
        vector /= np.linalg.norm(vector)
        if abs(vector[normal_axis]) > 1e-12:
            raise ValueError("Mode polarization must be transverse to the port normal.")
        vector[normal_axis] = 0
        result[mode] = tuple(float(value) for value in vector / np.linalg.norm(vector))
    if not set(result).issubset({item for group in groups for item in group}):
        raise ValueError("mode_polarizations must belong to a declared degenerate group.")
    for group in groups:
        if set(group).intersection(result):
            if len(group) != 2 or not set(group).issubset(result):
                raise ValueError("Physical polarization selection requires both members of a pair.")
            matrix = np.asarray([result[item] for item in group]).T
            if np.linalg.cond(matrix) > _CONDITION_LIMIT:
                raise ValueError(
                    "Requested polarization directions are linearly dependent or ill-conditioned."
                )
    return result


def parse_port_options(tokens):
    """Remove explicit key=value options before parsing the legacy anchor tail."""
    positional, options = [], {}
    for token in tokens:
        if "=" not in token:
            positional.append(token)
            continue
        key, value = token.split("=", 1)
        if key not in ("degenerate", "mode_polarizations") or key in options:
            raise ValueError(f"Unknown or repeated eigenmode port option {key!r}.")
        try:
            if key == "degenerate":
                options[key] = tuple(
                    tuple(int(v) for v in group.split(",")) for group in value.split(";")
                )
            else:
                directions = {}
                for entry in value.split(";"):
                    label, vector = entry.split(":", 1)
                    label = int(label)
                    if label in directions:
                        raise ValueError("Repeated polarization label")
                    directions[label] = (
                        tuple(float(v) for v in vector.split(",")) if "," in vector else vector
                    )
                options[key] = directions
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid eigenmode port {key}: {value!r}.") from exc
    return positional, options


def _mix(fields, transform):
    return [
        [sum(fields[i][axis] * transform[i, j] for i in range(len(fields))) for axis in range(3)]
        for j in range(transform.shape[1])
    ]


def _frame(electric, magnetic, impedance):
    return np.column_stack(
        [
            np.concatenate([*(v.ravel() for v in e), *(impedance * v.ravel() for v in h)])
            for e, h in zip(electric, magnetic)
        ]
    )


def _orthonormal(frame, *, transformation=False):
    u, s, vh = np.linalg.svd(frame, full_matrices=False)
    if not np.all(np.isfinite(s)) or s[-1] <= s[0] / _CONDITION_LIMIT:
        raise ValueError("Degenerate modal group has rank loss or ill-conditioned fields.")
    return (u, vh.conj().T / s) if transformation else u


def _power(owner, grid, electric, magnetic):
    # Coefficient convention: power(c) = c.conj().T @ matrix @ c.
    matrix = np.asarray(
        [[owner._modal_cross_power(ej, hi, grid) for ej in electric] for hi in magnetic]
    )
    return (matrix + matrix.conj().T) / 2


def _moments(owner, grid, electric):
    u, v = owner.transverse_axes
    area = grid.dl[u] * grid.dl[v]
    return np.asarray(
        [
            [
                np.sum(owner._average_to_transverse_cells(e[u], "eu")) * area,
                np.sum(owner._average_to_transverse_cells(e[v], "ev")) * area,
            ]
            for e in electric
        ]
    ).T


def balanced_power(owner, grid, electric, magnetic):
    """Positive field norm using the solver's transverse quadrature."""
    impedance = owner._tracking_impedance
    if owner.invariant_axis is None:
        u, v = owner.transverse_axes
        fields = [
            owner._average_to_transverse_cells(field, component)
            for field, component in (
                (electric[u], "eu"),
                (electric[v], "ev"),
                (impedance * magnetic[u], "hu"),
                (impedance * magnetic[v], "hv"),
            )
        ]
        measure = grid.dl[u] * grid.dl[v]
    else:
        t, a = owner.physical_transverse_axis, owner.invariant_axis
        axes = (t, a) if owner.domain_polarization == "TE" else (a, t)
        fields = [
            owner._sample_1d_component(electric[axes[0]]),
            impedance * owner._sample_1d_component(magnetic[axes[1]]),
        ]
        if owner.domain_polarization == "TM":
            fields = [0.5 * (field[:-1] + field[1:]) for field in fields]
        measure = grid.dl[t]
    return float(sum(np.sum(abs(field) ** 2) for field in fields) * measure / (4 * impedance))


def _residuals(solver, group, transform):
    indices = np.asarray(group) - 1
    operator = getattr(solver, "mode_tracking_operator", None)
    if operator is not None:
        columns = np.column_stack(
            [
                np.concatenate(
                    (solver.Eu[..., i].ravel(order="F"), solver.Ev[..., i].ravel(order="F"))
                )
                for i in indices
            ]
        )[solver.free_euv_mask]
    else:
        operator = solver.operator
        free = solver.free_scalar_mask
        operator = operator[free, :][:, free]
        field = solver.Ea if solver.polarization == "TM" else solver.Ha
        columns = field[free][:, indices]
    mixed = columns @ transform
    applied = operator @ mixed
    target = mixed * np.asarray(solver.eigenvalues)[indices]
    denominator = np.linalg.norm(applied, axis=0) + np.linalg.norm(target, axis=0)
    return np.linalg.norm(applied - target, axis=0) / np.maximum(denominator, 1e-300)


def align_groups(owner, grid, frequencies, solvers, mode_indices, anchor_e, anchor_h, propagating):
    """Align declared groups in place; never change solved propagation constants."""
    owner.degenerate_diagnostics = []
    owner._degenerate_masks = {}
    owner._degenerate_overlaps = {}
    owner._degenerate_guard_trimmed = {}
    for group in owner.degenerate:
        positions = [mode_indices.index(item) for item in group]
        context = f"Eigenmode port {owner.port_index} degenerate group {group}"
        active = propagating[:, positions[0]].copy()
        if not np.all(propagating[:, positions] == active[:, None]):
            raise ValueError(f"{context} has inconsistent propagation/cutoff among its partners.")
        used = np.flatnonzero(active)
        if not len(used) or np.any(np.diff(used) != 1):
            raise ValueError(f"{context} requires one contiguous propagating anchor range.")
        physical = group[0] in owner.mode_polarizations
        frames, transforms, raw, diagnostics = {}, {}, {}, []
        for k, solver in enumerate(solvers):
            eigenvalues = np.asarray(solver.eigenvalues)[np.asarray(group) - 1]
            if not np.all(np.isfinite(eigenvalues)):
                raise ValueError(f"{context} has nonfinite eigenvalues.")
            spread = float(np.max(abs(eigenvalues[:, None] - eigenvalues)))
            if spread > _DEGENERACY_TOLERANCE * max(1.0, float(np.max(abs(eigenvalues)))):
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has resolved eigenvalue splitting "
                    f"({spread:g}). Use independent modes or correct unintended geometry asymmetry."
                )
            # The second-order eigenvalue is -n_eff**2 and cannot distinguish
            # opposite phase branches. Such E/H pairs cannot be mixed even
            # when both carry forward real power (e.g. a backward-wave mode).
            indices = np.asarray(group) - 1
            propagation = np.asarray(solver.operator_neff)[indices]
            if active[k] and np.any(np.real(propagation[:, None] * propagation.conj()) < 0):
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has opposite propagation branches. "
                    "Use independent modes; equal squared eigenvalues do not permit mixing their E/H fields."
                )
            diagnostic = dict(
                frequency=frequencies[k],
                active=bool(active[k]),
                eigenvalue_spread=spread,
                eigenvalues=eigenvalues.copy(),
            )
            outside = [i for i in range(len(solver.eigenvalues)) if i + 1 not in group]
            if outside and np.any(
                abs(np.asarray(solver.eigenvalues)[outside, None] - eigenvalues)
                <= _DEGENERACY_TOLERANCE * max(1.0, float(np.max(abs(eigenvalues))))
            ):
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has a degenerate partner "
                    "outside the declared group. Include every partner in modes and degenerate."
                )
            diagnostics.append(diagnostic)
            if not active[k]:
                continue
            e = [anchor_e[k][p] for p in positions]
            h = [anchor_h[k][p] for p in positions]
            raw[k] = e, h
            frame = _frame(e, h, float(owner._tracking_impedance))
            # Test power in a well-conditioned representation of the subspace.
            # A raw Gram matrix squares the solver basis's condition number,
            # spuriously rejecting valid invertible changes of basis.
            _, preconditioner = _orthonormal(frame, transformation=True)
            power = _power(owner, grid, _mix(e, preconditioner), _mix(h, preconditioner))
            values, vectors = np.linalg.eigh(power)
            if values[0] <= values[-1] / _CONDITION_LIMIT or not np.all(np.isfinite(values)):
                raise ValueError(f"{context} has no independent positive-power basis.")
            diagnostic["power_condition"] = float(values[-1] / values[0])
            if physical:
                moment = _moments(owner, grid, e)
                condition = float(np.linalg.cond(moment))
                # Reject a numerically zero moment even if its two noise columns
                # happen to have a moderate relative condition number.
                scale = np.asarray(
                    [
                        sum(
                            np.sum(abs(field))
                            for field in (
                                item[owner.transverse_axes[0]],
                                item[owner.transverse_axes[1]],
                            )
                        )
                        for item in e
                    ]
                ) * np.prod(grid.dl[list(owner.transverse_axes)])
                if (
                    not np.isfinite(condition)
                    or condition > _CONDITION_LIMIT
                    or np.any(np.linalg.norm(moment, axis=0) <= 1e-12 * scale)
                ):
                    raise ValueError(
                        f"{context} at {frequencies[k]:g} Hz has zero or ill-conditioned "
                        "integrated transverse E. Axis/vector references cannot orient this group; "
                        "omit mode_polarizations to use generic subspace tracking."
                    )
                directions = np.asarray([owner.mode_polarizations[item] for item in group]).T
                diagnostic["direction_condition"] = float(np.linalg.cond(directions))
                transform = np.linalg.solve(moment, directions[list(owner.transverse_axes)])
                mixed_e, mixed_h = _mix(e, transform), _mix(h, transform)
                norms = np.real(np.diag(_power(owner, grid, mixed_e, mixed_h)))
                if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
                    raise ValueError(f"{context} has invalid polarized modal power.")
                transform /= np.sqrt(norms)[None, :]
                diagnostic["moment_condition"] = condition
            else:
                transform = preconditioner @ (vectors / np.sqrt(values)) @ vectors.conj().T
            transforms[k] = transform
            frames[k] = frame @ transform
            residual = _residuals(solver, group, transform)
            if not np.all(np.isfinite(residual)) or np.max(residual) > _RESIDUAL_TOLERANCE:
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has mixed-mode eigen-residual "
                    f"{np.max(residual):g}; use independent modes or improve the eigensolve."
                )
        overlaps = np.full(max(0, len(frequencies) - 1), np.nan)
        guard_trimmed = False
        while len(used) > 1:
            retry = False
            for left, right in zip(used[:-1], used[1:]):
                singular = np.linalg.svd(
                    _orthonormal(frames[left]).conj().T @ _orthonormal(frames[right]),
                    compute_uv=False,
                )
                overlap = float(np.min(singular))
                overlaps[left] = overlap
                if not np.isfinite(overlap) or overlap < owner.ANCHOR_OVERLAP_ERROR_THRESHOLD:
                    trim = None
                    if owner._automatic_anchor_policy() and len(used) > 2:
                        tolerance = 1e-12 * max(1.0, max(frequencies))
                        if (
                            left == used[0]
                            and owner.dft_start is not None
                            and frequencies[right] <= owner.dft_start + tolerance
                        ):
                            trim = left
                        elif (
                            right == used[-1]
                            and owner.dft_stop is not None
                            and frequencies[left] >= owner.dft_stop - tolerance
                        ):
                            trim = right
                    if trim is not None:
                        active[trim] = False
                        diagnostics[trim]["active"] = False
                        diagnostics[trim]["guard_trimmed"] = True
                        used = np.flatnonzero(active)
                        guard_trimmed = retry = True
                        if owner.mpi_coordinator:
                            logger.warning(
                                f"{context}: trimming the whole group at guard anchor "
                                f"{frequencies[trim]:g} Hz (subspace overlap {overlap:g})."
                            )
                        break
                    raise ValueError(
                        f"{context} cannot be tracked between {frequencies[left]:g} and "
                        f"{frequencies[right]:g} Hz: subspace overlap {overlap:g}. "
                        "Add anchors or inspect missing partners and nearby modes."
                    )
                owner._check_anchor_overlap(
                    overlap,
                    frequencies[left],
                    frequencies[right],
                    group,
                    context,
                    coordinator=owner.mpi_coordinator,
                )
            if not retry:
                break
        if not np.all(propagating[:, positions[0]]) and owner.mpi_coordinator:
            logger.warning(
                f"{context}: excluding non-propagating anchors as a whole group "
                "from excitation and physical modal references; inspect validity masks near cutoff."
            )
        centre = owner.fallback_frequency
        if centre is None:
            centre = (frequencies[0] + frequencies[-1]) / 2
        reference = int(min(used, key=lambda k: (abs(frequencies[k] - centre), frequencies[k])))
        if not physical:
            for path in (
                list(range(reference + 1, used[-1] + 1)),
                list(range(reference - 1, used[0] - 1, -1)),
            ):
                previous = reference
                for k in path:
                    u, _, vh = np.linalg.svd(frames[k].conj().T @ frames[previous])
                    rotation = u @ vh
                    transforms[k] = transforms[k] @ rotation
                    frames[k] = frames[k] @ rotation
                    previous = k
        for k in used:
            e, h = raw[k]
            transform = transforms[k]
            residual = _residuals(solvers[k], group, transform)
            if not np.all(np.isfinite(residual)) or np.max(residual) > _RESIDUAL_TOLERANCE:
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has mixed-mode eigen-residual "
                    f"{np.max(residual):g}; use independent modes or improve the eigensolve."
                )
            e, h = _mix(e, transform), _mix(h, transform)
            for j, p in enumerate(positions):
                anchor_e[k][p], anchor_h[k][p] = e[j], h[j]
            diagnostics[k].update(
                transform=transform, residual=residual, power_gram=_power(owner, grid, e, h)
            )
            if physical:
                diagnostics[k]["electric_moments"] = _moments(owner, grid, e)
        for p in positions:
            owner._degenerate_masks[p] = active
            owner._degenerate_overlaps[p] = overlaps
            owner._degenerate_guard_trimmed[p] = guard_trimmed
        owner.degenerate_diagnostics.append(
            dict(
                modes=group,
                physical=physical,
                reference_anchor=reference,
                overlaps=overlaps,
                anchors=diagnostics,
            )
        )


def write_diagnostics(group, owner):
    if not getattr(owner, "degenerate_diagnostics", None):
        return
    root = group.create_group("degenerate_groups")
    for number, record in enumerate(owner.degenerate_diagnostics):
        output = root.create_group(str(number + 1))
        output.attrs["ModeIndices"] = record["modes"]
        output.attrs["PhysicalPolarization"] = record["physical"]
        output.attrs["ReferenceAnchorIndex"] = record["reference_anchor"]
        output.attrs["TransverseAxes"] = owner.transverse_axes
        output.attrs["TransformationConvention"] = "aligned fields = raw fields @ transform"
        output["subspace_overlaps"] = record["overlaps"]
        if record["physical"]:
            output["requested_directions"] = [
                owner.mode_polarizations[item] for item in record["modes"]
            ]
        for k, anchor in enumerate(record["anchors"]):
            row = output.create_group(f"anchor{k}")
            for key, value in anchor.items():
                row[key] = value


def aligned_plot_solvers(owner, solvers):
    """Plot copies in the authoritative bank basis without altering raw solves."""
    result = [copy(solver) for solver in solvers]
    for record in owner.degenerate_diagnostics:
        indices = np.asarray(record["modes"]) - 1
        for k, diagnostic in enumerate(record["anchors"]):
            if not diagnostic["active"]:
                continue
            for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw", "Ea", "Ha", "Et", "Ht"):
                field = getattr(solvers[k], name, None)
                if field is None:
                    continue
                destination = np.array(getattr(result[k], name), copy=True)
                destination[..., indices] = field[..., indices] @ diagnostic["transform"]
                setattr(result[k], name, destination)
            result[k].mode_polarizations = owner.mode_polarizations
    return tuple(result)
