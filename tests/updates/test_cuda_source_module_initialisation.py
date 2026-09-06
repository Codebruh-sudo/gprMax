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

"""Each independently compiled CUDA source module owns its coefficient tables."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.cuda_opencl import knl_source_updates
from gprMax.updates import cuda_updates


pytestmark = pytest.mark.unit
FAMILIES = (
    ("hertzian", "hertziandipoles", "update_hertzian_dipole"),
    ("magnetic", "magneticdipoles", "update_magnetic_dipole"),
    ("voltage", "voltagesources", "update_voltage_source"),
)
SUBSETS = [tuple(index for index in range(3) if mask & (1 << index)) for mask in range(1, 8)]


class _DeviceGlobal:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.payload = None


class _SourceModule:
    """Keep constant addresses genuinely distinct across module generations."""

    def __init__(self, build, table_bytes):
        self.build = build
        self.table_bytes = table_bytes
        self.globals = {name: _DeviceGlobal(self, name) for name in ("updatecoeffsE", "updatecoeffsH")}
        self.lookups = []
        self.function = SimpleNamespace(module=self, name=build.function_name)

    def get_function(self, name):
        assert name == self.build.function_name
        self.lookups.append(("function", name))
        return self.function

    def get_global(self, name):
        self.lookups.append(("global", name))
        return self.globals[name], self.table_bytes[name]


@pytest.fixture
def source_setup_transport(monkeypatch, updates_config):
    """Exercise real module setup/copies without importing or initialising CUDA."""

    updates_config.sim_config.devices = {"nvcc_opts": ["--test-compiler-option"]}
    updates_config.model_config.device = {"dev": SimpleNamespace(total_constant_memory=64 * 1024)}

    def make(subset, dtype):
        updates = cuda_updates.CUDAUpdates.__new__(cuda_updates.CUDAUpdates)
        grid = SimpleNamespace(iterations=11)
        for index, (_, attribute, _) in enumerate(FAMILIES):
            # Different source counts make accidental cross-family array
            # reuse visible independently of the coefficient assertions.
            setattr(grid, attribute, [object() for _ in range(index + 1)] if index in subset else [])
        grid.updatecoeffsE = np.arange(15, dtype=dtype).reshape(3, 5) + 0.25
        grid.updatecoeffsH = -np.arange(15, dtype=dtype).reshape(3, 5) - 1.5
        updates.grid = grid
        updates.subs_name_args = {"REAL": "float" if dtype is np.float32 else "double"}
        updates.subs_func = {"preserved_macro": 321, "NY_SRCWAVES": -99}
        modules, transfers, uploads, builds = [], [], [], []

        def upload(sources, owner):
            assert owner is grid
            family = next(name for name, attribute, _ in FAMILIES if sources is getattr(grid, attribute))
            arrays = tuple(SimpleNamespace(family=family, slot=slot) for slot in range(3))
            uploads.append((family, sources, arrays))
            return arrays

        def build_kernel(template, name_args, substitutions):
            assert name_args is updates.subs_name_args
            family, _, function_name = next(
                record for record in FAMILIES if template is getattr(knl_source_updates, record[2])
            )
            build = SimpleNamespace(
                family=family,
                function_name=function_name,
                template=template,
                name_args=dict(name_args),
                substitutions=dict(substitutions),
            )
            builds.append(build)
            return build

        def compile_module(build, *, options):
            assert options is updates_config.sim_config.devices["nvcc_opts"]
            module = _SourceModule(
                build, {name: getattr(grid, name).nbytes for name in ("updatecoeffsE", "updatecoeffsH")}
            )
            modules.append(module)
            return module

        def memcpy_htod(destination, source):
            transfers.append((destination, source))
            destination.payload = source.copy()

        monkeypatch.setattr(cuda_updates, "htod_src_arrays", upload)
        updates._build_knl = build_kernel
        updates.source_module = compile_module
        updates.drv = SimpleNamespace(memcpy_htod=memcpy_htod)
        return SimpleNamespace(
            updates=updates,
            grid=grid,
            modules=modules,
            transfers=transfers,
            uploads=uploads,
            builds=builds,
        )

    return make


@pytest.mark.parametrize(
    "subset", SUBSETS, ids=["-".join(FAMILIES[index][0] for index in subset) for subset in SUBSETS]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["single", "double"])
def test_every_cuda_source_module_receives_fresh_material_tables(source_setup_transport, subset, dtype):
    transport = source_setup_transport(subset, dtype)
    updates, grid = transport.updates, transport.grid
    expected_families = [FAMILIES[index][0] for index in subset]
    previous_modules = []
    previous_arrays = []
    previous_tables = None

    for generation in range(2):
        if generation:
            # A new setup may change waveform length and material tables;
            # freshly compiled modules must not rely on prior module state.
            grid.iterations = 23
            grid.updatecoeffsE = np.full((4, 5), 7.25, dtype=dtype)
            grid.updatecoeffsH = np.full((4, 5), -3.5, dtype=dtype)
        module_start = len(transport.modules)
        transfer_start = len(transport.transfers)
        upload_start = len(transport.uploads)
        build_start = len(transport.builds)

        # Keep _copy_mat_coeffs real: assertions below inspect what its
        # actual driver copies wrote into each module's independent globals.
        updates._set_src_knls()

        modules = transport.modules[module_start:]
        uploads = transport.uploads[upload_start:]
        builds = transport.builds[build_start:]
        transfers = transport.transfers[transfer_start:]
        assert [module.build.family for module in modules] == expected_families
        assert [family for family, _, _ in uploads] == expected_families
        assert len(builds) == len(subset)
        assert len(transfers) == 2 * len(subset)
        assert len({id(module) for module in transport.modules}) == len(transport.modules)
        assert updates.subs_func == {
            "preserved_macro": 321,
            "NY_SRCINFO": 4,
            "NY_SRCWAVES": grid.iterations + 1,
        }

        for family_index, module, (_, sources, arrays), build in zip(subset, modules, uploads, builds):
            family, attribute, function_name = FAMILIES[family_index]
            assert sources is getattr(grid, attribute)
            assert build is module.build
            assert build.template is getattr(knl_source_updates, function_name)
            assert build.substitutions == updates.subs_func
            assert build.name_args == updates.subs_name_args
            assert getattr(updates, f"{function_name}_dev") is module.function
            assert module.function.module is module
            for slot, prefix in enumerate(("srcinfo1", "srcinfo2", "srcwaves")):
                assert getattr(updates, f"{prefix}_{family}_dev") is arrays[slot]
                assert all(arrays[slot] is not previous for previous in previous_arrays)
            assert module.lookups.count(("function", function_name)) == 1

            for name in ("updatecoeffsE", "updatecoeffsH"):
                destination = module.globals[name]
                expected = getattr(grid, name)
                assert destination.module is module
                assert module.lookups.count(("global", name)) == 1
                matching = [(target, source) for target, source in transfers if target is destination]
                assert len(matching) == 1, (family, name)
                assert matching[0][1] is expected
                assert destination.payload.dtype == np.dtype(dtype)
                np.testing.assert_array_equal(destination.payload, expected)

        absent = set(range(3)) - set(subset)
        for index in absent:
            family, _, function_name = FAMILIES[index]
            assert not hasattr(updates, f"{function_name}_dev")
            assert not hasattr(updates, f"srcwaves_{family}_dev")

        # No newly initialised module may overwrite or alias old constants.
        for module in previous_modules:
            for name, expected in previous_tables.items():
                np.testing.assert_array_equal(module.globals[name].payload, expected)
        pointers = [constant for module in transport.modules for constant in module.globals.values()]
        assert len({id(pointer) for pointer in pointers}) == 2 * len(transport.modules)
        previous_modules = list(modules)
        previous_arrays = [array for _, _, arrays in uploads for array in arrays]
        previous_tables = {name: getattr(grid, name).copy() for name in ("updatecoeffsE", "updatecoeffsH")}
