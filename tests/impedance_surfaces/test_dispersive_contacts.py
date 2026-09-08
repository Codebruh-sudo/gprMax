"""Dispersive clipped rows, complex histories, and complete CPU solves."""

from copy import deepcopy

import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.impedance_surfaces as implementation
from gprMax.grid.fdtd_grid import FDTDGrid


DL = 0.001
EDGE = np.array((4, 4, 4))
OFFSETS = (
    ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)),
    ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)),
    ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)),
)


def add_dielectric(scene, kind, *, material_id=None):
    material_id = kind if material_id is None else material_id
    scene.add(gprMax.Material(er=3, se=0.02, mr=1, sm=0, id=material_id))
    common = dict(poles=2, material_ids=[material_id])
    if kind == "debye":
        scene.add(gprMax.AddDebyeDispersion(er_delta=[2., 1.], tau=[3e-11, 8e-11], **common))
    elif kind == "lorentz":
        scene.add(gprMax.AddLorentzDispersion(
            er_delta=[2., 1.], omega=[8e9, 12e9], delta=[2e9, 3e9], **common,
        ))
    elif kind == "drude":
        scene.add(gprMax.AddDrudeDispersion(omega=[5e9, 9e9], alpha=[4e9, 6e9], **common))


def scene_base(iterations=2):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(8*DL,)*3), gprMax.Discretisation(p1=(DL,)*3),
        gprMax.TimeWindow(iterations=iterations), gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1), gprMax.SurfaceImpedance(id="wall", resistance=50),
    ):
        scene.add(obj)
    return scene


def build(scene, path, monkeypatch, *, precision="double", solve=False):
    captured = {}
    original = FDTDGrid.build
    original_compile = implementation.compile_impedance_surfaces

    def compile_dynamic(grid):
        grid.surface_impedance_models["wall"] = implementation.SurfaceImpedanceModel(
            "wall", A=((-2e9, 0), (0, -5e9)), B=(1e9, 2e9), C=(20, 10), D=50,
        )
        return original_compile(grid)

    def capture(grid):
        original(grid)
        captured["grid"] = grid

    with monkeypatch.context() as patch:
        patch.setattr(FDTDGrid, "build", capture)
        patch.setattr(implementation, "compile_impedance_surfaces", compile_dynamic)
        gprMax.run(scenes=[scene], outputfile=path, geometry_only=not solve,
                   cpu_precision=precision, hide_progress_bars=True, log_level=40)
    return captured["grid"]


def corner_scene(axis, retained_count, kind):
    scene = scene_base()
    kinds = ("debye", "lorentz", "drude") if kind == "mixed" else (kind,)
    for name in kinds:
        add_dielectric(scene, name)
    retained_ids = []
    for quadrant in range(4):
        name = "wall" if quadrant < 4-retained_count else kinds[quadrant % len(kinds)]
        coord = EDGE + OFFSETS[axis][quadrant]
        scene.add(gprMax.Box(p1=tuple(coord*DL), p2=tuple((coord+1)*DL), material_id=name))
        if name != "wall":
            retained_ids.append(name)
    return scene, retained_ids


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("retained_count", (1, 2, 3))
@pytest.mark.parametrize("kind", ("debye", "lorentz", "drude", "mixed"))
def test_compiled_dispersive_corner_matches_coupled_dense_row(
    precision, axis, retained_count, kind, tmp_path, monkeypatch,
):
    scene, retained_ids = corner_scene(axis, retained_count, kind)
    grid = build(scene, tmp_path/"corner", monkeypatch, precision=precision)
    system = grid.impedance_surfaces
    index = np.flatnonzero(np.all(system.edge_info[:, :4] == (axis, *EDGE), axis=1))[0]
    edge = system.edge_info[index]
    material_map = {m.ID: m for m in grid.materials}
    materials = [material_map[name] for name in retained_ids]
    area = DL**2/4
    plus = sum(area/m.srce for m in materials)
    minus = sum(area*m.CA/m.srce for m in materials)
    tolerance = 3e-6 if precision == "single" else 3e-14
    np.testing.assert_allclose(system.edge_params[index]+system.edge_dispersion[index],
                               (plus, minus), rtol=tolerance)
    start, stop = system.pole_offsets[index:index+2]
    # Repeated quadrants of the same material share a history with summed area.
    assert stop-start == sum(material_map[name].poles for name in set(retained_ids))
    if axis == 0 and retained_count == 3 and kind == "mixed":
        estimated = grid.mem_est_basic()
        bulk_poles = [getattr(m, "poles", None) for m in grid.materials]
        try:
            for m, poles in zip(grid.materials, bulk_poles):
                if poles is not None:
                    m.poles = 0
            no_polarization_estimate = grid.mem_est_basic()
        finally:
            for m, poles in zip(grid.materials, bulk_poles):
                if poles is not None:
                    m.poles = poles
        actual = sum(getattr(system, name).nbytes for name in
                     ("edge_dispersion", "pole_offsets", "pole_coeffs", "state_p"))
        assert estimated-no_polarization_estimate >= actual
    expected = []
    for m in sorted({m.numID: m for m in materials}.values(), key=lambda m: m.numID):
        weight = area*retained_ids.count(m.ID)*config.sim_config.em_consts["e0"]
        for f, b, c in zip(m.eqt[:m.poles], weight*m.zt[:m.poles], m.eqt2[:m.poles]):
            expected.append((f.real, f.imag, b.real, b.imag, c.real, c.imag))
    np.testing.assert_allclose(system.pole_coeffs[start:stop], expected, rtol=tolerance, atol=1e-25)

    rng = np.random.default_rng(31)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(grid, name)[:] = rng.normal(size=grid.Ex.shape)
    system.state_p[:] = rng.normal(size=system.state_p.shape)*1e-5
    system.state_y[:] = rng.normal(size=system.state_y.shape)*0.1
    e_old = float((grid.Ex, grid.Ey, grid.Ez)[axis][tuple(EDGE)])
    r_h = sum(float(system.h_weight[h])*float((grid.Hx, grid.Hy, grid.Hz)[int(system.h_info[h, 0])]
              [tuple(system.h_info[h, 1:])]) for h in range(edge[4], edge[4]+edge[5]))
    states = system.state_p[start:stop].astype(float)
    coefficients = system.pole_coeffs[start:stop].astype(float)
    phi = np.sum(coefficients[:, 4]*states[:, 0]-coefficients[:, 5]*states[:, 1])
    ports = range(edge[6], edge[6]+edge[7])
    matrix = np.zeros((1+edge[7], 1+edge[7]))
    rhs = np.zeros(1+edge[7])
    matrix[0, 0], rhs[0] = plus, minus*e_old+r_h-phi
    for row, port in enumerate(ports, 1):
        model, offset = system.port_info[port]
        count = system.model_info[model, 0]
        history = float(system.state_y[offset:offset+count].astype(float).sum())
        matrix[0, row], matrix[row, 0], matrix[row, row] = -system.port_g[port], -0.5, system.model_Z0[model]
        rhs[row] = 0.5*e_old-history
    dense = np.linalg.solve(matrix, rhs)
    for backend in ("python", "cython"):
        copy_grid = deepcopy(grid)
        copy_system = copy_grid.impedance_surfaces
        (copy_system._update_python if backend == "python" else copy_system.update)(copy_grid)
        e_new = (copy_grid.Ex, copy_grid.Ey, copy_grid.Ez)[axis][tuple(EDGE)]
        np.testing.assert_allclose(e_new, dense[0], rtol=tolerance, atol=tolerance)
        f = coefficients[:, 0]+1j*coefficients[:, 1]
        b = coefficients[:, 2]+1j*coefficients[:, 3]
        expected_state = f*(states[:, 0]+1j*states[:, 1])+b*(e_old-dense[0])
        actual = copy_system.state_p[start:stop]
        np.testing.assert_allclose(actual[:, 0]+1j*actual[:, 1], expected_state,
                                   rtol=tolerance*4, atol=tolerance*1e-5)
        for row, port in enumerate(ports, 1):
            model, offset = system.port_info[port]
            count, cf = system.model_info[model]
            expected_y = (system.model_f[cf:cf+count]*system.state_y[offset:offset+count]
                          +system.model_q[cf:cf+count]*dense[row])
            np.testing.assert_allclose(copy_system.state_y[offset:offset+count], expected_y,
                                       rtol=tolerance*4, atol=tolerance)


@pytest.mark.parametrize("kind", ("debye", "lorentz", "drude", "mixed"))
def test_modal_polarization_matches_real_state_space_resolvent(kind, tmp_path, monkeypatch):
    scene, _ = corner_scene(2, 3, kind)
    grid = build(scene, tmp_path/"modal", monkeypatch)
    system = grid.impedance_surfaces
    for theta in (0.001, 0.1, 0.8, 2.7):
        z = np.exp(1j*theta)
        for index in range(system.edge_count):
            dp, dm = system.edge_dispersion[index]
            expected = dp*np.exp(0.5j*theta)-dm*np.exp(-0.5j*theta)
            start, stop = system.pole_offsets[index:index+2]
            for fr, fi, br, bi, cr, ci in system.pole_coeffs[start:stop]:
                transition = np.array(((fr, -fi), (fi, fr)))
                response = np.linalg.solve(z*np.eye(2)-transition, [br, bi])
                expected += np.dot([cr, -ci], response)*(1-z)*np.exp(-0.5j*theta)
            np.testing.assert_allclose(system.polarization_admittance(index, theta), expected,
                                       rtol=3e-12, atol=1e-20)


def test_inclusive_contact_converges_to_physical_complex_permittivity(tmp_path, monkeypatch):
    from gprMax.materials import create_electric_average_material

    scene, _ = corner_scene(2, 3, "mixed")
    grid = build(scene, tmp_path/"inclusive", monkeypatch)
    materials = {m.ID: m for m in grid.materials}
    averaged = create_electric_average_material(
        materials["debye"].numID, "inclusive_exterior",
        [materials[name] for name in ("debye", "lorentz", "drude", "free_space")],
    )
    assert averaged.inclusive_conductivity > 0
    assert averaged.poles == 6
    for quadrant in (1, 2, 3):
        grid.solid[tuple(EDGE+OFFSETS[2][quadrant])] = averaged.numID
    grid.materials[averaged.numID] = averaged
    grid.maxpoles = averaged.poles
    frequency = 7e9
    physical = averaged.calculate_er(frequency)
    errors = []
    for _ in range(3):
        system = implementation.compile_impedance_surfaces(grid)
        index = np.flatnonzero(np.all(system.edge_info[:, :4] == (2, *EDGE), axis=1))[0]
        start, stop = system.pole_offsets[index:index+2]
        assert stop-start == 6
        theta = 2*np.pi*frequency*grid.dt
        omega = 2*np.sin(theta/2)/grid.dt
        plus, minus = system.edge_params[index]
        load = (plus*np.exp(0.5j*theta)-minus*np.exp(-0.5j*theta)
                +system.polarization_admittance(index, theta))
        represented = load/(1j*omega*config.sim_config.em_consts["e0"]*3*DL**2/4)
        errors.append(abs(represented-physical))
        grid.dt /= 2
    assert errors[1] < 0.3*errors[0]
    assert errors[2] < 0.3*errors[1]


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("kind", ("debye", "lorentz", "drude"))
def test_driven_dispersive_exterior_decays_and_resets(kind, precision, tmp_path, monkeypatch):
    scene = scene_base(iterations=2400)
    add_dielectric(scene, kind)
    scene.add(gprMax.Box(p1=(0,)*3, p2=(8*DL,)*3, material_id=kind))
    scene.add(gprMax.Box(p1=(3*DL,)*3, p2=(5*DL,)*3, material_id="wall"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="pulse"))
    scene.add(gprMax.HertzianDipole((2*DL, 4*DL, 4*DL), "z", "pulse"))
    scene.add(gprMax.Rx(p1=(2*DL, 4*DL, 4*DL)))
    grid = build(scene, tmp_path/"driven", monkeypatch, precision=precision, solve=True)
    trace = grid.rxs[0].outputs["Ez"]
    assert np.isfinite(trace).all()
    peak = np.max(np.abs(trace))
    assert peak > 1e-6
    assert np.max(np.abs(trace[-200:])) < 0.02*peak
    assert np.any(grid.impedance_surfaces.state_p != 0)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Tx", "Ty", "Tz"):
        assert np.isfinite(getattr(grid, name)).all()
    # Held boundary IDs prevent bulk A/B history updates on these same edges.
    for axis, i, j, k, *_ in grid.impedance_surfaces.edge_info:
        assert np.all((grid.Tx, grid.Ty, grid.Tz)[axis][:, i, j, k] == 0)
    grid.reset_fields()
    assert not np.any(grid.impedance_surfaces.state_p)
    assert not np.any(grid.impedance_surfaces.state_y)
