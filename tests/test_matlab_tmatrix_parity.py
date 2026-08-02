from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from zenscat.core import Device1D, Grid1D, launch_rcwa_t

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_golden.mat"
T_FIXTURES = ("small_t_e_3x3", "small_t_h_3x3")
NUMERIC_FIELDS = {
    "TRN": ("minus_1", "plus_1", "TRN0", "sum"),
    "REF": ("minus_1", "plus_1", "REF0", "sum"),
}


def _as_list(value):
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    return [value]


def _mat_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.ravel().astype(str)).strip()
        if value.size == 1:
            return _mat_text(value.item())
    return str(value)


def _mat_number(value):
    array = np.asarray(value)
    assert array.size == 1
    return array.item()


def _load_t_fixtures():
    data = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    fixtures = {_mat_text(fixture.name): fixture for fixture in _as_list(data["fixtures"])}
    return [fixtures[name] for name in T_FIXTURES]


def test_launch_rcwa_t_matches_matlab_public_trn_ref_fields() -> None:
    max_abs_diff = 0.0

    for fixture in _load_t_fixtures():
        cfg = fixture.config
        assert _mat_text(cfg.method) == "T"
        grid_solver = fixture.grid_solver
        grid = Grid1D(
            Lam0=np.asarray(grid_solver.Lam0, dtype=np.float64),
            Theta=np.asarray(grid_solver.Theta, dtype=np.float64),
            Lx=float(_mat_number(grid_solver.Lx)),
            erR=complex(_mat_number(grid_solver.erR)),
            urR=complex(_mat_number(grid_solver.urR)),
            erT=complex(_mat_number(grid_solver.erT)),
            urT=complex(_mat_number(grid_solver.urT)),
            erSub=complex(_mat_number(grid_solver.erSub)),
            layer_num=int(_mat_number(grid_solver.layer_num)),
        )
        device = Device1D(
            ERC=np.asarray(fixture.device.ERC, dtype=np.complex128),
            sub_L=np.asarray(fixture.device.sub_L, dtype=np.float64),
        )

        trn, ref = launch_rcwa_t(
            int(_mat_number(cfg.NH)),
            grid,
            device,
            _mat_text(cfg.polarization),
            bool(_mat_number(cfg.calc_fresnel)),
        )

        actual_structs = {"TRN": trn, "REF": ref}
        for struct_name, fields in NUMERIC_FIELDS.items():
            expected_struct = getattr(fixture, struct_name)
            actual_struct = actual_structs[struct_name]
            for field in fields:
                actual = np.asarray(getattr(actual_struct, field), dtype=np.float64)
                expected = np.asarray(getattr(expected_struct, field), dtype=np.float64)
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)

    assert max_abs_diff < 2e-10
