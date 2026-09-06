pytest test suite
=================

The ``tests`` directory is the automated pytest test suite. It contains
focused unit tests as well as compact integration and hardware tests. The
larger, manually run scientific comparisons in ``testing/validation`` are
kept separate from this suite.

Installation
------------

Install the development requirements, including pytest, in the active
environment::

    python -m pip install -r requirements.txt
    python -m pip install -e ".[mpi]"

The complete developer suite requires an MPI runtime and ``mpi4py``, even
when running the non-GPU selection; ordinary serial installations do not.
Real distributed-output tests additionally need MPI-enabled HDF5/h5py.
For distributed fractals use the supported Python 3.12 environment with
FFTW and ``mpi4py-fft`` (the ``mpi-fractals`` extra). The FFT smoke tests
exercise real and complex 2-D and 3-D plans, not just module imports.

Running tests
-------------

Run the complete suite::

    python -m pytest

Run the usual CPU development selection, excluding real-GPU and unusually
slow tests::

    python -m pytest -m "not gpu and not slow"

CI splits this coverage: ``tests.yml`` runs ``-m unit`` on several platforms,
and ``pytest.yml`` runs ``-m "not unit and not gpu and not slow"``. Slow and
real-device tests need a separate local or hardware-enabled run.

Run only compact integration tests::

    python -m pytest -m "integration and not slow and not gpu"

Run tests that require a real GPU, selecting device 1 in this example::

    python -m pytest -m gpu --gpu-device 1

The CUDA index can instead be set with ``GPRMAX_TEST_GPU``. Tests skip
themselves when the selected CUDA device is unavailable. Tests that inspect
generated GPU source or use mocks are not marked ``gpu`` because they do
not require real hardware.

OpenCL tests use ``--opencl-device`` (or ``GPRMAX_TEST_OPENCL``). For example::

    python -m pytest -m gpu -k opencl --opencl-device 0

Use pytest's duration report when deciding whether a test needs the
``slow`` marker::

    python -m pytest --durations=25

Markers
-------

``unit``
    Focused tests with no full production solve. This is the cross-platform
    CI selection; it can still require installed compiled dependencies.

``integration``
    Exercises several gprMax components together or executes a complete,
    compact model. This includes the automated FDTD comparisons with a
    Hertzian-dipole closed form and PEC-sphere Mie theory.

``gpu``
    Executes on a real GPU. This marker may overlap with ``integration``.

``slow``
    Normally takes more than 10 seconds on a development machine. It may be
    combined with either of the other markers.

Analytical helper functions tested without running an FDTD model are normal
unit tests; there is deliberately no separate ``analytical`` marker.
