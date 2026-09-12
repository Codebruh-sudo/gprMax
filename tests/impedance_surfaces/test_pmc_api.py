"""Public exact-PMC limit and invalid non-finite model inputs."""

import numpy as np
import pytest

import gprMax
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import evaluate_surface_ade
from gprMax.hash_cmds_file import get_user_objects
from gprMax.impedance_surfaces import SurfaceImpedanceModel


def test_native_pmc_api_and_hash_have_exact_zero_admittance():
    command = gprMax.SurfaceImpedance(id="wall", resistance=float("inf"))
    (parsed,) = get_user_objects([str(command) + "\n"], checkessential=False)
    assert np.isposinf(parsed.D)
    model = SurfaceImpedanceModel("wall", D=parsed.D)
    assert model.is_pmc
    discrete = model.discretise(1e-12)
    assert np.isposinf(discrete.Z0) and discrete.F.size == 0
    response = evaluate_surface_ade(
        frequency_hz=1e9, dt=1e-12, F=discrete.F, G=discrete.G, L=discrete.L, Z0=discrete.Z0
    )
    assert response.admittance == 0j


@pytest.mark.parametrize("value", (float("nan"), -float("inf")))
def test_pmc_does_not_allow_other_nonfinite_resistances(value):
    with pytest.raises(ValueError):
        gprMax.SurfaceImpedance(id="wall", resistance=value)
    with pytest.raises(ValueError):
        SurfaceImpedanceModel("wall", D=value)


def test_infinite_feedthrough_is_rejected_for_dynamic_models():
    with pytest.raises(ValueError, match="finite"):
        SurfaceImpedanceModel("wall", A=((-1.0,),), B=(1.0,), C=(1.0,), D=np.inf)
