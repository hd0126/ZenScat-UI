from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from zenscat.core import Device1D, Grid1D, launch_rcwa_s

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_golden.mat"
FIELD_MAP = {
    "TRN": ("minus_1", "plus_1", "TRN0", "sum"),
    "REF": ("minus_1", "plus_1", "REF0", "sum"),
}


def _fixtures_by_name() -> dict[str, object]:
    data = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    fixtures = np.atleast_1d(data["fixtures"]).ravel()
    return {str(fixture.name): fixture for fixture in fixtures}


@pytest.mark.parametrize(
    "fixture_name",
    (
        "casual_default_s_e_41x21",
        "small_s_e_3x3",
        "small_s_h_3x3",
    ),
)
def test_python_s_matrix_matches_matlab_public_results(fixture_name: str) -> None:
    fixture = _fixtures_by_name()[fixture_name]
    saved_grid = fixture.grid_solver
    grid = Grid1D(
        Lam0=np.asarray(saved_grid.Lam0, dtype=np.float64),
        Theta=np.asarray(saved_grid.Theta, dtype=np.float64),
        Lx=float(saved_grid.Lx),
        erR=complex(saved_grid.erR),
        urR=complex(saved_grid.urR),
        erT=complex(saved_grid.erT),
        urT=complex(saved_grid.urT),
        erSub=complex(saved_grid.erSub),
        layer_num=int(saved_grid.layer_num),
    )
    device = Device1D(
        ERC=np.asarray(fixture.device.ERC, dtype=np.complex128),
        sub_L=np.asarray(fixture.device.sub_L, dtype=np.float64),
    )

    trn, ref = launch_rcwa_s(
        harmonic_count=int(fixture.config.NH),
        grid=grid,
        device=device,
        modes=str(fixture.config.polarization),
    )

    for result_name, python_result, matlab_result in (
        ("TRN", trn, fixture.TRN),
        ("REF", ref, fixture.REF),
    ):
        for field_name in FIELD_MAP[result_name]:
            np.testing.assert_allclose(
                getattr(python_result, field_name),
                np.asarray(getattr(matlab_result, field_name), dtype=np.float64),
                rtol=1e-11,
                atol=2e-12,
                err_msg=f"{fixture_name} {result_name}.{field_name}",
            )
