"""MPI regression defaults must support old and new Open MPI launchers."""

import pytest

from tests.conftest import _configure_mpi_test_environment

pytestmark = pytest.mark.unit


def test_mpi_test_defaults_support_openmpi_and_prrte():
    environment = {"UNRELATED_SETTING": "keep"}
    _configure_mpi_test_environment(environment)
    assert environment == {
        "UNRELATED_SETTING": "keep",
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("OMPI_MCA_rmaps_base_oversubscribe", "0"),
        ("PRTE_MCA_rmaps_default_mapping_policy", "slot:NOOVERSUBSCRIBE"),
        ("PRTE_MCA_rmaps_default_mapping_policy", "node:OVERSUBSCRIBE"),
    ],
)
def test_explicit_mpi_mapping_settings_are_preserved(key, value):
    environment = {key: value}
    _configure_mpi_test_environment(environment)
    assert environment[key] == value
    configured = environment.copy()
    _configure_mpi_test_environment(environment)
    assert environment == configured
