from __future__ import annotations

from threading import Event

import numpy as np
import pytest

from zenscat.core import (
    Device1D,
    Grid1D,
    SimulationCancelled,
    build_fdfd_device,
    build_fdfd_grid,
    fdfd_2d,
    launch_rcwa_s,
    launch_rcwa_t,
)


def _rcwa_inputs() -> tuple[Grid1D, Device1D]:
    grid = Grid1D(Lam0=[500e-9], Theta=[0.0], Lx=0.4)
    device = Device1D(ERC=np.eye(3, dtype=np.complex128)[:, :, np.newaxis].repeat(2, axis=2), sub_L=[50e-9, 50e-9])
    return grid, device


@pytest.mark.parametrize("launcher", (launch_rcwa_s, launch_rcwa_t))
def test_rcwa_launchers_honor_preflight_cancellation(launcher) -> None:
    grid, device = _rcwa_inputs()
    cancelled = Event()
    cancelled.set()

    with pytest.raises(SimulationCancelled, match="cancelled"):
        if launcher is launch_rcwa_s:
            launcher(1, grid, device, "E", cancel_token=cancelled)
        else:
            launcher(1, grid, device, "E", cancel_token=cancelled)


def test_s_matrix_reports_completed_sweep_points() -> None:
    grid, device = _rcwa_inputs()
    updates: list[tuple[int, int]] = []

    launch_rcwa_s(1, grid, device, ("E", "H"), progress=lambda done, total: updates.append((done, total)))

    assert updates == [(1, 2), (2, 2)]


def test_fdfd_honors_preflight_cancellation() -> None:
    grid = build_fdfd_grid(
        [0.1, 1.5],
        layer_num=1,
        Lam0=[0.5],
        Theta=[0.0],
        Lx=0.32,
        h=0.1,
        nres=4,
        spacer=[0.2, 0.2],
        npml=[2, 2],
    )
    device = build_fdfd_device([0.1, 1.5], grid)
    cancelled = Event()
    cancelled.set()

    with pytest.raises(SimulationCancelled, match="cancelled"):
        fdfd_2d(grid, device, "E", cancel_token=cancelled)
