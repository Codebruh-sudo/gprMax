"""Physical Yee support, not array padding, bounds conventional point sources."""

from contextlib import nullcontext

import numpy as np
import pytest

import gprMax
from gprMax import sources
from gprMax.grid.fdtd_grid import FDTDGrid

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
# Explicit upper indices for a 12 x 14 x 16 cell model. These are independent
# expectations for each component, not the production offset calculation.
UPPER = {
    "Ex": (11, 14, 16),
    "Ey": (12, 13, 16),
    "Ez": (12, 14, 15),
    "Hx": (12, 13, 15),
    "Hy": (11, 14, 15),
    "Hz": (11, 13, 16),
}
DL = np.array((0.001, 0.002, 0.003))
SIZE = np.array((12, 14, 16))


def _scene(component, point, family=None, iterations=1):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=tuple(SIZE * DL)),
        gprMax.Discretisation(p1=tuple(DL)),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.TimeWindow(iterations=iterations),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="pulse"),
    ):
        scene.add(obj)
    constructor = family or (
        gprMax.HertzianDipole if component[0] == "E" else gprMax.MagneticDipole
    )
    extra = (
        {"resistance": 50} if constructor in (gprMax.VoltageSource, gprMax.TransmissionLine) else {}
    )
    source = constructor(
        p1=tuple(point * DL), polarisation=component[1], waveform_id="pulse", **extra
    )
    scene.add(source)
    return scene, source


@pytest.mark.integration
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("location", ("lower", "last", "beyond"))
@pytest.mark.parametrize("interface", ("api", "hash"))
def test_initial_dipole_uses_component_bounds(tmp_path, component, axis, location, interface):
    point = np.array((6, 7, 8))
    point[axis] = {
        "lower": 0,
        "last": UPPER[component][axis],
        "beyond": UPPER[component][axis] + 1,
    }[location]
    options = {}
    if interface == "api":
        scene, _ = _scene(component, point)
        options["scenes"] = [scene]
    else:
        filename = tmp_path / "bounds.in"
        command = "hertzian_dipole" if component[0] == "E" else "magnetic_dipole"
        filename.write_text(
            "#domain: " + " ".join(map(str, SIZE * DL)) + "\n"
            "#dx_dy_dz: " + " ".join(map(str, DL)) + "\n"
            "#pml_cells: 0\n#omp_threads: 1\n#time_window: 1\n"
            "#waveform: gaussian 1 1e10 pulse\n"
            f"#{command}: {component[1]} " + " ".join(map(str, point * DL)) + " pulse\n",
            encoding="utf-8",
        )
        options["inputfile"] = filename
    with pytest.raises(ValueError) if location == "beyond" else nullcontext():
        gprMax.run(
            **options, geometry_only=True, outputfile=tmp_path / "bounds", hide_progress_bars=True
        )


@pytest.mark.integration
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("family", (gprMax.VoltageSource, gprMax.TransmissionLine))
def test_other_single_electric_edge_sources_reject_own_axis_padding(tmp_path, axis, family):
    point = np.array((6, 7, 8))
    point[axis] = SIZE[axis]
    scene, _ = _scene("E" + "xyz"[axis], point, family, iterations=80)
    with pytest.raises(ValueError, match="physical Yee component"):
        gprMax.run(
            scenes=[scene],
            geometry_only=True,
            outputfile=tmp_path / "invalid",
            hide_progress_bars=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("offset", ((0, 0, 0), (5, 6, 7)))
def test_component_bounds_use_global_extent_not_rank_size(grid_config, component, offset):
    """Exercise coordinate conversion only; real MPI solves are tested separately."""
    grid = FDTDGrid()
    grid.size = np.array((7, 8, 9))
    grid.global_size = SIZE
    grid.is_distributed = True
    grid.local_to_global_coordinate = lambda p: p + offset
    source = sources.HertzianDipole() if component[0] == "E" else sources.MagneticDipole()
    source.polarisation = component[1]
    grid.validate_point_source_position(source, np.array(UPPER[component]) - offset)
    for axis in range(3):
        point = np.array(UPPER[component]) - offset
        point[axis] += 1
        with pytest.raises(ValueError, match="physical Yee component"):
            grid.validate_point_source_position(source, point)


@pytest.mark.integration
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("restart", (False, True))
@pytest.mark.parametrize("geometry_fixed", (False, True))
def test_stepped_dipole_rejects_padded_final_position(tmp_path, component, restart, geometry_fixed):
    axis = "xyz".index(component[1]) if component[0] == "E" else ("xyz".index(component[1]) + 1) % 3
    point = np.array((6, 7, 8))
    point[axis] = SIZE[axis] - 2
    scene, _ = _scene(component, point)
    step = np.zeros(3)
    step[axis] = DL[axis]
    scene.add(gprMax.SrcSteps(p1=tuple(step)))
    options = {"n": 2, "i": 2} if restart else {"n": 3}
    with pytest.raises(ValueError) as error:
        gprMax.run(
            scenes=[scene],
            geometry_fixed=geometry_fixed,
            geometry_only=True,
            outputfile=tmp_path / "scan",
            hide_progress_bars=True,
            **options,
        )
    assert "physical Yee component" in str(error.value.__cause__)


@pytest.mark.integration
@pytest.mark.parametrize("component", COMPONENTS)
def test_study_cannot_move_dipole_into_padding(tmp_path, component):
    point = np.array((6, 7, 8))
    scene, source = _scene(component, point, iterations=2)
    axis = "xyz".index(component[1]) if component[0] == "E" else ("xyz".index(component[1]) + 1) % 3
    point[axis] = SIZE[axis]
    study = gprMax.GPRStudy(
        [
            gprMax.StudyCase("initial", []),
            gprMax.StudyCase("invalid", [gprMax.ObjectState(source, position=tuple(point * DL))]),
        ]
    )
    with pytest.raises(ValueError, match="physical Yee component"):
        gprMax.run(
            scenes=[scene], study=study, outputfile=tmp_path / "study", hide_progress_bars=True
        )


@pytest.mark.integration
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("mode", ("TM", "TE"))
@pytest.mark.parametrize("kind", ("E", "H"))
def test_2d_dipoles_cannot_be_stepped_off_active_layer(tmp_path, axis, mode, kind):
    domain = np.full(3, 0.012)
    domain[axis] = np.inf
    point = np.full(3, 0.006)
    point[axis] = np.inf
    polarisation = "xyz"[axis if (kind == "E") == (mode == "TM") else (axis + 1) % 3]
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=tuple(domain)),
        gprMax.DomainMode(mode=mode),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.TimeWindow(iterations=1),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="pulse"),
    ):
        scene.add(obj)
    constructor = gprMax.HertzianDipole if kind == "E" else gprMax.MagneticDipole
    scene.add(constructor(p1=tuple(point), polarisation=polarisation, waveform_id="pulse"))
    # The initial active-layer source is valid in all six reductions.
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "valid_2d",
        hide_progress_bars=True,
    )
    step = np.zeros(3)
    step[axis] = 0.001
    scene.add(gprMax.SrcSteps(p1=tuple(step)))
    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene],
            n=2,
            geometry_fixed=True,
            geometry_only=True,
            outputfile=tmp_path / "invalid_2d",
            hide_progress_bars=True,
        )


@pytest.mark.integration
@pytest.mark.parametrize("component", COMPONENTS)
def test_off_origin_subgrid_dipoles_use_local_fine_coordinates(tmp_path, monkeypatch, component):
    from gprMax.model import Model

    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.09,) * 3),
        gprMax.Discretisation(p1=(0.003,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.TimeWindow(iterations=1),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="pulse"),
    ):
        scene.add(obj)
    subgrid = gprMax.SubGridHSG(p1=(0.03,) * 3, p2=(0.06,) * 3, ratio=3, id="fine")
    subgrid.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="pulse"))
    constructor = gprMax.HertzianDipole if component[0] == "E" else gprMax.MagneticDipole
    subgrid.add(constructor(p1=(0.045,) * 3, polarisation=component[1], waveform_id="pulse"))
    scene.add(subgrid)
    captured = []
    original = Model.build

    def capture(model):
        result = original(model)
        captured.extend(model.subgrids)
        return result

    monkeypatch.setattr(Model, "build", capture)
    gprMax.run(
        scenes=[scene],
        subgrid=True,
        autotranslate=True,
        geometry_only=True,
        outputfile=tmp_path / "fine",
        hide_progress_bars=True,
    )
    grid = captured[0]
    source = (grid.hertziandipoles if component[0] == "E" else grid.magneticdipoles)[0]
    expected = 15 + np.array(
        (grid.n_boundary_cells_x, grid.n_boundary_cells_y, grid.n_boundary_cells_z)
    )
    np.testing.assert_array_equal(source.coord, expected)
