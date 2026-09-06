"""Count actual device writes under padded and looped snapshot launches."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.cuda_opencl.knl_snapshots import store_snapshot
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.snapshots import Snapshot, YEE_OFFSETS


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_snapshot_overlaunch_has_exactly_one_writer_and_preserves_padding(
    monkeypatch, request, backend, precision
):
    dtype = np.float32 if precision == "single" else np.float64
    real = "float" if precision == "single" else "double"
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(dtypes={"float_or_double": dtype}))
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D", ompthreads=1))
    grid = FDTDGrid()
    grid.size = np.asarray((22, 30, 20), dtype=np.int32)
    grid.dl, grid.dt = np.asarray((0.001, 0.002, 0.003)), 1e-12
    rng = np.random.default_rng(20260906)
    for name in YEE_OFFSETS:
        setattr(grid, name, np.ascontiguousarray(rng.normal(size=tuple(grid.size + 1)), dtype=dtype))
    allocation = (7, 9, 5)
    volume = int(np.prod(allocation))  # 315: not a multiple of 32.
    shape = (3, *allocation)
    sentinel = dtype(-123.25)
    header = "#define IDX3D_FIELDS(x,y,z) ((x)*31*21+(y)*21+(z))\n"
    header += "#define IDX4D_SNAPS(p,x,y,z) ((p)*315+(x)*45+(y)*5+(z))\n"
    context = None
    if backend == "cuda":
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        from pycuda.compiler import SourceModule
        context = cuda.Device(request.getfixturevalue("gpu_device")).make_context()
        fields = [gpuarray.to_gpu(getattr(grid, name)) for name in YEE_OFFSETS]
        counter_arg = ", unsigned int *writes"
        count_write = "atomicAdd(&writes[IDX4D_SNAPS(p,x,y,z)], 1u);"
    else:
        import pyopencl as cl
        import pyopencl.array as clarray
        devices = [device for platform in cl.get_platforms() for device in platform.get_devices()]
        device = devices[request.getfixturevalue("opencl_device")]
        if precision == "double" and "cl_khr_fp64" not in device.extensions:
            pytest.skip("Selected OpenCL device lacks double precision")
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        fields = [clarray.to_device(queue, getattr(grid, name)) for name in YEE_OFFSETS]
        counter_arg = ", __global unsigned int *writes"
        count_write = "atomic_inc(&writes[IDX4D_SNAPS(p,x,y,z)]);"
    try:
        body = store_snapshot["func"].substitute(
            REAL=real, NX_SNAPS=7, NY_SNAPS=9, NZ_SNAPS=5, CUDA_IDX=""
        )
        anchor = "// Increment subscripts for field array to account for spatial sampling of snapshot"
        assert body.count(anchor) == 1
        # Instrument entry to the common six-component write block. An atomic
        # count exposes duplicate writers even when numerical values coincide.
        body = body.replace(anchor, count_write + "\n" + anchor)
        launch_count = 32 * ((volume + 37 + 31) // 32)
        if backend == "cuda":
            args = store_snapshot["args_cuda"].substitute(REAL=real).rstrip()
            code = header + args[:-1] + counter_arg + ") {\n"
            code += "int i = blockIdx.x * blockDim.x + threadIdx.x;\n" + body + "\n}"
            module = SourceModule(code)
            kernels = [module.get_function("store_snapshot")]
        else:
            args = store_snapshot["args_opencl"].substitute(REAL=real)
            kernels = []
            for looped in (False, True):
                code = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" + header
                code += "__kernel void store_snapshot(" + args + counter_arg + ") {\n"
                if looped:
                    code += f"for (int i = get_global_id(0); i < {launch_count}; i += get_global_size(0)) {{\n"
                    code += body + "\n}\n}"
                else:
                    code += "int i = get_global_id(0);\n" + body + "\n}"
                program = cl.Program(context, code).build()
                kernels.append(cl.Kernel(program, "store_snapshot"))
        # Rotated smaller windows share max-shaped output pitches; p isolates
        # stored snapshots and all unselected padding must remain untouched.
        for p, current_shape in enumerate(((7, 3, 2), (2, 9, 5), (3, 4, 2))):
            for step in ((1, 1, 1), (2, 3, 2)):
                start = np.asarray((1, 2, 1))
                stop = start + np.asarray(current_shape) * np.asarray(step)
                snap = Snapshot(*start, *stop, *step, 0, "unused", ".h5",
                                dict.fromkeys(YEE_OFFSETS, True), grid)
                snap.initialise_snapfields()
                snap.store()
                scalars = [np.int32(v) for v in (p, *start, *current_shape, *step, 1, 1, 1)]
                selection = (p, *(slice(0, n) for n in current_shape))
                expected_count = np.zeros(shape, dtype=np.uint32)
                expected_count[selection] = 1
                for index, kernel in enumerate(kernels):
                    initial = np.full(shape, sentinel, dtype=dtype)
                    if backend == "cuda":
                        outputs = [gpuarray.to_gpu(initial) for _ in YEE_OFFSETS]
                        counts = gpuarray.zeros(shape, np.uint32)
                        kernel(*scalars, *fields, *outputs, counts,
                               block=(32, 1, 1), grid=(launch_count // 32, 1, 1))
                    else:
                        outputs = [clarray.to_device(queue, initial) for _ in YEE_OFFSETS]
                        counts = clarray.zeros(queue, shape, np.uint32)
                        kernel(queue, (1,) if index else (launch_count,), (1,) if index else (32,),
                               *scalars, *(field.data for field in fields),
                               *(array.data for array in outputs), counts.data)
                    np.testing.assert_array_equal(counts.get(), expected_count)
                    for name, output in zip(YEE_OFFSETS, outputs):
                        expected = initial.copy()
                        expected[selection] = snap.snapfields[name]
                        assert output.get().tobytes() == expected.tobytes(), (name, current_shape, step)
    finally:
        if backend == "cuda" and context is not None:
            context.pop()
