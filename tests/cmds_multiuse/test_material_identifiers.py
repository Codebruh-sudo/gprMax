"""User material IDs must not enter the generated-material namespace."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects
from gprMax.materials import Material as InternalMaterial

COLLISION_ID = "dielectric+dielectric+free_space+free_space"


def _declaration(interface, kind, material_id):
    if kind == "native":
        command = f"#material: 99 0 1 0 {material_id}"
        obj = gprMax.Material(er=99, se=0, mr=1, sm=0, id=material_id)
    else:
        command = f"#material_from_database: fundamental vacuum {material_id}"
        obj = gprMax.MaterialFromDatabase(database="fundamental", material="vacuum", id=material_id)
    return obj if interface == "api" else get_user_objects([command], checkessential=False)[0]


@pytest.mark.unit
@pytest.mark.parametrize("interface", ("api", "hash"))
@pytest.mark.parametrize("kind", ("native", "database"))
@pytest.mark.parametrize(
    "material_id", ("soil+air", "+soil", "soil+", "Hmag_a+a+b+b", COLLISION_ID)
)
def test_user_material_rejects_generated_id_before_grid_mutation(interface, kind, material_id):
    grid = SimpleNamespace(materials=[])
    with pytest.raises(ValueError, match="reserved for automatically averaged material"):
        _declaration(interface, kind, material_id).build(grid)
    assert grid.materials == []


@pytest.mark.unit
@pytest.mark.parametrize("interface", ("api", "hash"))
def test_native_material_accepts_underscore_alternative(interface):
    grid = SimpleNamespace(materials=[])
    _declaration(interface, "native", "soil_air").build(grid)
    assert grid.materials[0].ID == "soil_air"
    assert grid.materials[0].er == 99


@pytest.mark.unit
def test_internal_material_still_accepts_generated_ids():
    assert InternalMaterial(3, COLLISION_ID).ID == COLLISION_ID


def _scene():
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.02,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
        gprMax.TimeWindow(iterations=1),
        gprMax.Material(er=4, se=0, mr=1, sm=0, id="dielectric"),
        gprMax.Box(p1=(0.01, 0, 0), p2=(0.02,) * 3, material_id="dielectric"),
    ):
        scene.add(obj)
    return scene


@pytest.mark.integration
@pytest.mark.parametrize("interface", ("api", "hash"))
@pytest.mark.parametrize("kind", ("native", "database"))
def test_unused_collision_declaration_cannot_change_an_interface(tmp_path, interface, kind):
    scene = _scene()
    scene.add(_declaration(interface, kind, COLLISION_ID))
    with pytest.raises(ValueError, match="reserved for automatically averaged material"):
        gprMax.run(
            scenes=[scene],
            geometry_only=True,
            outputfile=tmp_path / "invalid",
            hide_progress_bars=True,
        )

    # The safe alternative may be unused, but it must not alter the four-edge
    # interface average between air (1) and dielectric (4).
    scene = _scene()
    scene.add(_declaration(interface, kind, "unused_dielectric_air"))
    filename = tmp_path / "valid"
    scene.add(gprMax.GeometryObjectsWrite(p1=(0, 0, 0), p2=(0.02,) * 3, filename=str(filename)))
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=filename,
        hide_progress_bars=True,
    )
    catalogue = json.loads((tmp_path / "valid_materials.json").read_text())
    with h5py.File(filename.with_suffix(".h5")) as geometry:
        key = geometry["material_keys"][int(geometry["ID"][2, 10, 10, 10])].decode()
    assert catalogue["materials"][key]["base"]["relative_permittivity"] == 2.5


@pytest.mark.integration
@pytest.mark.parametrize("interface", ("api", "hash"))
def test_database_display_name_with_plus_is_not_a_local_id(tmp_path, monkeypatch, interface):
    monkeypatch.chdir(tmp_path)
    document = {
        "schema": "gprMax-material-database",
        "schema_version": 1,
        "database": {"id": "labels", "name": "Labels", "version": "1"},
        "materials": {
            "mixture": {
                "name": "Soil + air",
                "model": "constant",
                "base": {
                    "relative_permittivity": 4,
                    "electric_conductivity_s_per_m": 0,
                    "relative_permeability": 1,
                    "magnetic_conductivity_s_per_m": 0,
                },
            },
        },
    }
    (tmp_path / "labels.json").write_text(json.dumps(document))
    if interface == "api":
        material = gprMax.MaterialFromDatabase(database="labels", material="mixture")
    else:
        material = get_user_objects(
            ["#material_from_database: labels mixture"],
            checkessential=False,
        )[0]
    scene = _scene()
    scene.add(material)
    filename = tmp_path / "labelled"
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.02,) * 3, material_id="mixture"))
    scene.add(gprMax.GeometryObjectsWrite(p1=(0, 0, 0), p2=(0.02,) * 3, filename=str(filename)))
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=filename,
        hide_progress_bars=True,
    )
    catalogue = json.loads((tmp_path / "labelled_materials.json").read_text())
    with h5py.File(filename.with_suffix(".h5")) as geometry:
        keys = geometry["material_keys"].asstr()[:]
        for index in np.unique(geometry["data"][:]):
            assert catalogue["materials"][keys[index]]["metadata"]["original_id"] == "mixture"
