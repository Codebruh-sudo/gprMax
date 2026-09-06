"""Real CUDA/OpenCL native Yee snapshot interpolation; no Metal hardware claim."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.cuda_opencl.knl_snapshots import store_snapshot
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.mode2d import mode2d_geometry
from gprMax.snapshots import Snapshot, YEE_OFFSETS, _snapshot_axis_strides


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["3D", "2D TMx", "2D TMy", "2D TMz", "2D TEx", "2D TEy", "2D TEz"])
def test_device_snapshot_native_interpolation_matches_cpu_and_coordinates(
    monkeypatch, request, backend, dtype, mode
):
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(dtypes={"float_or_double": dtype}))
    monkeypatch.setattr(
        config, "get_model_config", lambda: SimpleNamespace(mode=mode, ompthreads=1)
    )
    geometry = mode2d_geometry(mode)
    grid = FDTDGrid()
    grid.size, grid.dl, grid.dt = np.asarray((14, 13, 12)), np.asarray((0.001, 0.002, 0.003)), 1e-12
    indices = np.indices(tuple(grid.size + 1))
    active = (
        tuple(YEE_OFFSETS)
        if geometry is None
        else geometry.active_electric + geometry.active_magnetic
    )
    for name, offsets in YEE_OFFSETS.items():
        values = sum((indices[a] + offsets[a] / 2) * grid.dl[a] * (a + 1) for a in range(3))
        if geometry is not None:
            values = np.where(indices[geometry.invariant_axis] == geometry.live_index, values, 0)
        setattr(grid, name, np.ascontiguousarray(values, dtype=dtype))
    context = None
    if backend == "cuda":
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        from pycuda.compiler import SourceModule

        index = request.getfixturevalue("gpu_device")
        context = cuda.Device(index).make_context()
        fields = [gpuarray.to_gpu(getattr(grid, name)) for name in YEE_OFFSETS]
    else:
        import pyopencl as cl
        import pyopencl.array as clarray

        index = request.getfixturevalue("opencl_device")
        devices = [device for platform in cl.get_platforms() for device in platform.get_devices()]
        device = devices[index]
        if dtype == np.float64 and "cl_khr_fp64" not in device.extensions:
            pytest.skip("Selected OpenCL device does not support double precision")
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        fields = [clarray.to_device(queue, getattr(grid, name)) for name in YEE_OFFSETS]
    try:
        real = "float" if dtype == np.float32 else "double"
        for spacing in ((1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4)):
            start, stop, step = np.asarray((1, 2, 1)), np.asarray((10, 11, 10)), np.asarray(spacing)
            if geometry is not None:
                axis = geometry.invariant_axis
                start[axis], stop[axis], step[axis] = (
                    geometry.live_index,
                    geometry.live_index + 1,
                    1,
                )
            snap = Snapshot(
                *start, *stop, *step, 4, "unused", ".h5", dict.fromkeys(YEE_OFFSETS, True), grid
            )
            snap.initialise_snapfields()
            snap.store()
            shape = tuple(snap.grid_view.size)
            header = "#define IDX3D_FIELDS(x,y,z) ((x)*14*13+(y)*13+(z))\n"
            header += f"#define IDX4D_SNAPS(p,x,y,z) ((p)*{np.prod(shape)}+(x)*{shape[1]*shape[2]}+(y)*{shape[2]}+(z))\n"
            substitutions = dict(REAL=real, NX_SNAPS=shape[0], NY_SNAPS=shape[1], NZ_SNAPS=shape[2])
            arguments = [np.int32(v) for v in (0, *start, *shape, *step, *_snapshot_axis_strides())]
            if backend == "cuda":
                code = header + store_snapshot["args_cuda"].substitute(REAL=real) + "{\n"
                code += (
                    store_snapshot["func"].substitute(
                        **substitutions, CUDA_IDX="int i = blockIdx.x * blockDim.x + threadIdx.x;"
                    )
                    + "\n}"
                )
                module = SourceModule(code)
                outputs = [gpuarray.zeros(shape, dtype) for _ in YEE_OFFSETS]
                module.get_function("store_snapshot")(
                    *arguments, *fields, *outputs, block=(1, 1, 1), grid=(int(np.prod(shape)), 1, 1)
                )
            else:
                code = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" + header
                code += (
                    "__kernel void store_snapshot("
                    + store_snapshot["args_opencl"].substitute(REAL=real)
                    + ") {\n"
                )
                # Force one work-item to process many indices, as a real
                # ElementwiseKernel does when its launch is smaller than n.
                # A return in the shared coarse branch would truncate output.
                code += (
                    "for (int i = get_global_id(0); i < "
                    + str(int(np.prod(shape)))
                    + "; i += get_global_size(0)) {\n"
                )
                code += store_snapshot["func"].substitute(**substitutions, CUDA_IDX="") + "\n}}"
                program = cl.Program(context, code).build()
                outputs = [clarray.zeros(queue, shape, dtype) for _ in YEE_OFFSETS]
                cl.Kernel(program, "store_snapshot")(
                    queue,
                    (1,),
                    (1,),
                    *arguments,
                    *(field.data for field in fields),
                    *(array.data for array in outputs),
                )
            expected = sum(
                (snap._physical_origin()[a] + (np.indices(shape)[a] + 0.5) * step[a] * grid.dl[a])
                * (a + 1)
                for a in range(3)
            )
            for name, output in zip(YEE_OFFSETS, outputs):
                values = output.get()
                assert values.tobytes() == snap.snapfields[name].tobytes(), name
                if name in active:
                    np.testing.assert_allclose(values, expected, rtol=3e-7, atol=1e-9)
    finally:
        if backend == "cuda" and context is not None:
            context.pop()


@pytest.mark.unit
def test_metal_template_uses_shared_native_interpolation_without_abi_change():
    args = store_snapshot["args_metal"].substitute(REAL="float")
    body = store_snapshot["func"].substitute(
        REAL="float", NX_SNAPS=2, NY_SNAPS=3, NZ_SNAPS=4, CUDA_IDX=""
    )
    assert "device const int& dx" in args
    assert "device const int& sx" in args
    assert "sx * (dx - 1)" in body
    assert "xx+qx/2+a,yy+qy/2+b,zz+qz/2+c" in body
