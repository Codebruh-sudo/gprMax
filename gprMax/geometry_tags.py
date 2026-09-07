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

"""Cell-centred semantic labels, separate from electromagnetic material IDs.

Scene preparation discovers tag names before building geometry, freezes one
model-wide registry, and allocates maps only on grids that declare tags. A map
has the same cell indexing as ``solid``, not the component indexing of the Yee
material arrays. Geometry writers overwrite tags alongside cell materials;
an untagged volume writes zero, including when it carves out another volume.
Material smoothing does not average these discrete labels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Optional

import numpy as np
import numpy.typing as npt

UNTAGGED_NAME = "untagged"
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def validate_geometry_tag(tag: Optional[str]) -> Optional[str]:
    """Validate a tag and return it unchanged; ``None`` denotes untagged cells.

    Names are case-sensitive and are neither stripped nor otherwise rewritten.
    The reserved display name ``untagged`` cannot be an explicit user tag.
    """

    if tag is None:
        return None
    if not isinstance(tag, str):
        raise TypeError("Geometry tag must be a string or None")
    if tag == UNTAGGED_NAME:
        raise ValueError(f"Geometry tag '{UNTAGGED_NAME}' is reserved for tag ID 0")
    if not _TAG_PATTERN.fullmatch(tag):
        raise ValueError(
            "Geometry tag must start with a letter or digit and contain only letters, "
            "digits, '_', '-', '.', or ':'"
        )
    return tag


def validate_geometry_tag_ids(values: npt.ArrayLike, catalogue_size: int) -> npt.NDArray:
    """Check file-local IDs before indexing, including an all-untagged file."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu":
        raise ValueError("Geometry tag IDs must be integers")
    if values.size and int(values.min()) < 0:
        raise ValueError("Geometry tag IDs must be non-negative")
    if values.size and int(values.max()) >= catalogue_size:
        raise ValueError("Geometry object contains a tag ID absent from its tag-name table")
    return values


class GeometryTagRegistry:
    """Assign compact IDs in registration order, with zero reserved for untagged.

    Repeated names share an ID. IDs are local to this registry, so persistence
    must include ``names`` and imports must remap by name, not copy numeric IDs
    between independently constructed models. Freezing prevents new names from
    exceeding the integer width selected when a grid allocates its tag map.
    """

    def __init__(self) -> None:
        self._names = [UNTAGGED_NAME]
        self._ids = {UNTAGGED_NAME: 0}
        self._frozen = False

    def register(self, tag: Optional[str]) -> int:
        tag = validate_geometry_tag(tag)
        if tag is None:
            return 0
        existing = self._ids.get(tag)
        if existing is not None:
            return existing
        if self._frozen:
            raise RuntimeError(f"Cannot register geometry tag '{tag}' after registry is frozen")
        tag_id = len(self._names)
        if tag_id > np.iinfo(np.uint32).max:
            raise ValueError("Geometry tag count exceeds the uint32 ID range")
        self._ids[tag] = tag_id
        self._names.append(tag)
        return tag_id

    def register_many(self, tags: Iterable[Optional[str]]) -> None:
        for tag in tags:
            self.register(tag)

    def freeze(self) -> None:
        """Prevent new names while allowing lookup or registration of existing ones."""

        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def has_tags(self) -> bool:
        return len(self._names) > 1

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def dtype(self) -> np.dtype:
        """Smallest supported unsigned width that can hold the highest assigned ID."""

        max_id = len(self._names) - 1
        if max_id <= np.iinfo(np.uint8).max:
            return np.dtype(np.uint8)
        if max_id <= np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    def id_for(self, tag: Optional[str]) -> int:
        tag = validate_geometry_tag(tag)
        if tag is None:
            return 0
        try:
            return self._ids[tag]
        except KeyError as exc:
            raise KeyError(f"Geometry tag '{tag}' is not registered") from exc


class GeometryTagMap:
    """Host-side, C-contiguous ``(nx, ny, nz)`` cell labels for one grid.

    ``shape`` is in cells, with no extra Yee-component endpoint planes. The
    frozen registry chooses uint8, uint16 or uint32 storage; all cells initially
    have ID zero. Scene preparation leaves the map absent for untagged grids.
    This is persistent geometry, not field history: geometry-fixed runs retain
    it along with ``solid`` rather than clearing it during the field reset.
    """

    def __init__(self, shape: tuple[int, int, int], registry: GeometryTagRegistry) -> None:
        if not registry.frozen:
            raise RuntimeError("Geometry tag registry must be frozen before allocating a map")
        if not registry.has_tags:
            raise ValueError("A geometry tag map is not required for an empty registry")
        self.registry = registry
        self.data = np.zeros(shape, dtype=registry.dtype, order="C")

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def id_for(self, tag: Optional[str]) -> int:
        return self.registry.id_for(tag)

    def remap_file_ids(
        self, values: npt.NDArray[np.integer], file_names: Iterable[str]
    ) -> npt.NDArray[np.integer]:
        """Return file-local labels remapped to this registry, preserving shape.

        ``file_names[index]`` identifies each nonzero input ID; index zero is
        always untagged. Input IDs must be non-negative integers within the
        file's catalogue; validate before indexing or narrowing their dtype.
        All file names must be registered before freezing the registry. The
        returned array uses this map's dtype without writing into the map.
        """

        names = tuple(file_names)
        values = validate_geometry_tag_ids(values, len(names))
        lookup = np.asarray(
            [self.registry.id_for(name if index else None) for index, name in enumerate(names)]
        )
        return np.asarray(lookup[values], dtype=self.data.dtype, order="C")
