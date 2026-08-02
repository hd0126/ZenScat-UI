from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from zenscat.core import Device1D, Grid1D, convmat1d, launch_rcwa_s

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_golden.mat"


def _fixtures() -> dict[str, object]:
    payload = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    return {str(fixture.name): fixture for fixture in np.atleast_1d(payload["fixtures"]).ravel()}


def test_lossless_oracles_are_passive_and_conserve_energy() -> None:
    for fixture in _fixtures().values():
        transmission = np.asarray(fixture.TRN.sum, dtype=np.float64)
        reflection = np.asarray(fixture.REF.sum, dtype=np.float64)
        assert np.min(transmission) >= -1e-12
        assert np.min(reflection) >= -1e-12
        assert np.max(transmission) <= 1 + 1e-9
        assert np.max(reflection) <= 1 + 1e-9
        np.testing.assert_allclose(transmission + reflection, 1.0, rtol=0, atol=1e-9)


def test_public_propagating_order_efficiencies_are_non_negative() -> None:
    for fixture in _fixtures().values():
        for struct_name, zero_name in (("TRN", "TRN0"), ("REF", "REF0")):
            result = getattr(fixture, struct_name)
            for field_name in ("minus_1", "plus_1", zero_name, "sum"):
                assert np.min(np.asarray(getattr(result, field_name), dtype=np.float64)) >= -1e-12


def test_symmetric_profile_has_equal_plus_minus_orders_at_normal_incidence() -> None:
    for fixture in _fixtures().values():
        angles = np.asarray(fixture.Theta, dtype=np.float64).ravel()
        normal_indices = np.flatnonzero(np.isclose(angles, 0.0, rtol=0, atol=1e-15))
        assert normal_indices.size >= 1
        for result in (fixture.TRN, fixture.REF):
            minus = np.asarray(result.minus_1, dtype=np.float64)
            plus = np.asarray(result.plus_1, dtype=np.float64)
            if minus.ndim == 1:
                minus = minus.reshape((-1, angles.size))
                plus = plus.reshape((-1, angles.size))
            np.testing.assert_allclose(
                minus[:, normal_indices],
                plus[:, normal_indices],
                rtol=0,
                atol=2e-12,
            )


def test_harmonic_refinement_stabilizes_the_default_zero_order() -> None:
    fixture = _fixtures()["casual_default_s_e_41x21"]
    saved = fixture.grid_solver
    grid = Grid1D(
        Lam0=[510e-9],
        Theta=[0.0],
        Lx=float(saved.Lx),
        erR=complex(saved.erR),
        urR=complex(saved.urR),
        erT=complex(saved.erT),
        urT=complex(saved.urT),
        erSub=complex(saved.erSub),
        layer_num=int(saved.layer_num),
    )
    er = np.asarray(fixture.device.ER, dtype=np.complex128)
    sub_l = np.asarray(fixture.device.sub_L, dtype=np.float64)
    zero_orders = []
    for harmonic_count in range(2, 7):
        device = Device1D(convmat1d(er, harmonic_count), sub_l)
        transmission, _ = launch_rcwa_s(harmonic_count, grid, device, "E")
        zero_orders.append(float(transmission.TRN0.item()))

    deltas = np.abs(np.diff(zero_orders))
    assert np.all(np.diff(deltas) < 0)
    assert deltas[-1] < 3e-4
