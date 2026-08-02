from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

os.environ.setdefault(
    "QT_QPA_PLATFORM", "minimal" if sys.platform == "darwin" else "offscreen"
)
os.environ.setdefault("ZENSCAT_QT_BINDING", "PySide6")

from zenscat.core import DiffractionResult
from zenscat.gui import MainWindow
from zenscat.gui.main_window import (
    COMBO_POPUP_STYLE,
    PROFILE_INTERFACE_DESCRIPTIONS,
    ValidationResult,
)
from zenscat.gui.qt_compat import (
    QT_BINDING,
    QApplication,
    QCheckBox,
    QComboBox,
    QCoreApplication,
    QDoubleSpinBox,
    QLabel,
    QListView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    Qt,
    QTableWidget,
    QWidget,
)
from zenscat.legacy_io import LegacyResultBundle
from zenscat.optimization import OptimizationProgress
from zenscat.project import ProjectDocument


@pytest.fixture()
def app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


def _process_events() -> None:
    for _ in range(3):
        QCoreApplication.processEvents()


def _wait_until(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _set_small_run(window: MainWindow) -> None:
    window.sweep_points.setValue(1)
    window.nx_input.setValue(32)
    window.nz_input.setValue(2)
    window.harmonics.setValue(1)
    window.wavelength_start_nm.setValue(510.0)
    window.wavelength_stop_nm.setValue(510.0)
    window.angle_start_deg.setValue(0.0)
    window.angle_stop_deg.setValue(0.0)


def _set_tiny_optimization(window: MainWindow) -> None:
    window.optimization_generations.setValue(1)
    window.optimization_population.setValue(1)
    values = (
        ("period_um", "0.31", "0.33"),
        ("height_um", "0.14", "0.16"),
        ("first_thickness_um", "0.16", "0.20"),
    )
    for row, row_values in enumerate(values):
        for column, value in enumerate(row_values):
            window.optimization_bounds_table.item(row, column).setText(value)


def _set_tiny_fdfd(window: MainWindow) -> None:
    window.fdfd_nres.setValue(2.0)
    window.fdfd_period_num.setValue(3)
    window.fdfd_npml_x.setValue(1)
    window.fdfd_npml_y.setValue(1)
    window.fdfd_spacer_top.setValue(0.1)
    window.fdfd_spacer_bottom.setValue(0.1)


def _nav_label(window: MainWindow) -> str:
    return str(window.nav_list.currentItem().data(Qt.ItemDataRole.UserRole))


def _select_nav(window: MainWindow, label: str) -> None:
    for index in range(window.nav_list.count()):
        if window.nav_list.item(index).data(Qt.ItemDataRole.UserRole) == label:
            window.nav_list.setCurrentRow(index)
            return
    raise AssertionError(f"missing nav page: {label}")


def _empty_legacy_bundle() -> LegacyResultBundle:
    values = np.zeros((1, 1), dtype=np.float64)
    return LegacyResultBundle(
        wavelengths_m=np.array([500e-9]),
        angles_rad=np.array([0.0]),
        transmission=DiffractionResult(values, values, TRN0=values, sum=values),
        reflection=DiffractionResult(values, values, REF0=values, sum=values),
    )


class _LegacyExportRun:
    metadata: ClassVar[dict[str, str]] = {"workflow": "casual_rcwa"}

    def __init__(self, bundle: LegacyResultBundle | object | None = None) -> None:
        self._bundle = bundle or _empty_legacy_bundle()

    def legacy_bundle(
        self, params: object | None = None
    ) -> LegacyResultBundle | object:
        return self._bundle


def test_qt_compat_prefers_supported_binding(app: QApplication) -> None:
    assert QT_BINDING in {"PySide6", "PyQt6"}
    assert QApplication.instance() is app


def test_main_window_exposes_precision_lab_shell(app: QApplication) -> None:
    window = MainWindow()
    window.show()
    _process_events()

    assert window.objectName() == "zenscatMainWindow"
    assert window.nav_list.count() == 7
    assert [
        window.nav_list.item(index).data(Qt.ItemDataRole.UserRole)
        for index in range(window.nav_list.count())
    ] == [
        "Project",
        "Device",
        "RCWA Sweep",
        "Optimize",
        "FDFD Fields",
        "Results",
        "Jobs",
    ]
    assert window.workspace.count() == 7
    assert set(window._pages) == {
        "Project",
        "Device",
        "RCWA Sweep",
        "Optimize",
        "FDFD Fields",
        "Results",
        "Jobs",
    }
    assert window.workspace.widget(0).objectName() == "projectScrollArea"
    assert window.findChild(QPushButton, "validateButton") is not None
    assert window.findChild(QPushButton, "runButton").isEnabled() is False
    assert window.findChild(QPushButton, "cancelButton").isEnabled() is False
    assert window.findChild(QPushButton, "exportButton").isEnabled() is False
    assert window.findChild(QPushButton, "openProjectButton") is not None
    assert window.findChild(QPushButton, "saveProjectButton") is not None
    assert window.findChild(QPushButton, "loadImportButton") is not None

    window.close()


def test_superset_controls_are_exposed_and_collected(app: QApplication) -> None:
    window = MainWindow()
    window.show()
    _process_events()

    assert window.compatibility_mode_combo.currentText() == "modern"
    assert window.external_command_combo.isEditable()
    assert isinstance(window.interface_smooth, QCheckBox)
    assert window.rcwa_run_mode_combo.itemText(1) == "Harmonic convergence"
    assert "R(+2)" in [
        window.optimization_objective.itemText(index)
        for index in range(window.optimization_objective.count())
    ]
    assert "PhC Honeycomb" in [
        window.phc_shape_combo.itemText(index)
        for index in range(window.phc_shape_combo.count())
    ]
    assert window.fdfd_polarization_combo.itemText(1) == "Both E + H"
    assert "phase(f)" in [
        window.fdfd_field_combo.itemText(index)
        for index in range(window.fdfd_field_combo.count())
    ]

    window.interface_smooth.setChecked(True)
    window.trapz_bottom.setValue(0.12)
    window.periodic_stack.setChecked(True)
    window.periodic_count.setValue(7)
    window.flat_substrate.setChecked(True)
    window.calc_fresnel.setChecked(True)
    inputs = window.collect_inputs()

    assert inputs.interface_smooth is True
    assert inputs.trapz_w_bot == pytest.approx(0.12)
    assert inputs.is_periodic is True
    assert inputs.period_num == 7
    assert inputs.flat_substrate is True
    assert inputs.calc_fresnel is True
    window.close()


def test_analytic_request_forwards_superset_geometry_controls(app: QApplication) -> None:
    window = MainWindow()
    window.interface_combo.setCurrentText("DE4")
    window.supergauss_sigma.setValue(0.08)
    window.supergauss_m.setValue(4.0)
    window.interface_smooth.setChecked(True)
    window.periodic_stack.setChecked(True)
    window.periodic_count.setValue(9)
    window.flat_substrate.setChecked(True)
    window.calc_fresnel.setChecked(True)

    request = window._service.build_request(window.collect_inputs())

    assert request.interface_params.smooth is True
    assert request.interface_params.supergauss_sigma == pytest.approx(0.08)
    assert request.interface_params.supergauss_m == pytest.approx(4.0)
    assert request.is_periodic is True
    assert request.period_num == 9
    assert request.flat_substrate is True
    assert request.calc_fresnel is True
    window.close()


def test_result_view_uses_heatmap_for_two_dimensional_sweep_and_slice_on_request(
    app: QApplication,
) -> None:
    window = MainWindow()
    values = np.arange(6, dtype=np.float64).reshape(2, 3) / 10
    run = SimpleNamespace(
        transmission=DiffractionResult(values, values, TRN0=values, sum=values),
        reflection=DiffractionResult(values, values, REF0=values, sum=values),
        wavelengths_m=np.array([500e-9, 510e-9]),
        angles_rad=np.deg2rad([0.0, 2.0, 4.0]),
        metadata={"workflow": "casual_rcwa"},
        elapsed_s=0.01,
    )
    window._last_run = run
    window.result_family_combo.setCurrentText("Transmission")
    window.result_order_combo.setCurrentText("0")
    window.result_view_combo.setCurrentText("Auto")
    window._update_result_plot()

    assert window.results_plot._heatmap is not None
    assert window.results_plot._series is None

    window.result_view_combo.setCurrentText("Wavelength middle")
    window._update_result_plot()
    assert window.results_plot._series is not None
    assert window.results_plot._series.shape == (3,)
    assert window.results_plot._heatmap is None

    window.result_order_combo.setCurrentText("+2")
    window._update_result_plot()
    assert "Unavailable: Transmission order +2" in window.results_plot.accessibleName()
    assert window.results_plot._series is None
    window.close()


def test_external_backend_validation_is_explicit_when_unconfigured(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZENSCAT_EXTERNAL_SOLVER", raising=False)
    window = MainWindow()
    window.backend_combo.setCurrentText("External CLI")
    window.external_command_combo.setCurrentText("")

    result = window._service.validate(window.collect_inputs())

    assert result.ok is False
    assert any("external solver command" in message.lower() for message in result.messages)
    window.close()


@pytest.mark.parametrize("window_size", [(1282, 1075), (1280, 780)])
def test_workspace_combos_keep_readable_widths(
    app: QApplication, window_size: tuple[int, int]
) -> None:
    window = MainWindow()
    window.resize(*window_size)
    window.show()
    _process_events()

    for combo in window.findChildren(QComboBox):
        if combo.objectName() not in {
            "backendCombo",
            "projectNameCombo",
            "sourceCombo",
            "methodCombo",
            "interfaceCombo",
            "distributionCombo",
            "modeCombo",
            "phcShapeCombo",
            "phcRepeatCombo",
            "optimizationObjectiveCombo",
            "fdfdInterfaceCombo",
            "fdfdPaletteCombo",
            "fdfdFieldCombo",
            "resultFamilyCombo",
            "resultOrderCombo",
        }:
            continue
        text_width = max(
            (
                combo.fontMetrics().horizontalAdvance(combo.itemText(index))
                for index in range(combo.count())
            ),
            default=combo.fontMetrics().horizontalAdvance(combo.currentText()),
        )
        popup_width = max(combo.view().sizeHintForColumn(0), text_width)
        assert combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
        assert combo.minimumWidth() >= text_width
        assert combo.view().minimumWidth() >= popup_width

    window.close()


def test_combo_popup_selection_colors_are_explicit() -> None:
    def style_block(selector: str) -> str:
        match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", COMBO_POPUP_STYLE)
        assert match is not None
        return match.group(1)

    popup = style_block("QListView")
    assert "selection-background-color: #b9dce9;" in popup
    assert "selection-color: #17212b;" in popup

    hover = style_block("QListView::item:hover")
    assert "background-color: #b9dce9;" in hover
    assert "color: #17212b;" in hover
    assert "font-weight: 600;" in hover

    selected = style_block("QListView::item:selected")
    assert "background-color: #b9dce9;" in selected
    assert "color: #17212b;" in selected
    assert "font-weight: 600;" in selected


def test_workspace_combos_use_hoverable_non_native_popup_views(
    app: QApplication,
) -> None:
    window = MainWindow()
    window.show()
    _process_events()

    for combo in window.findChildren(QComboBox):
        popup_view = combo.view()
        assert isinstance(popup_view, QListView)
        assert popup_view.metaObject().className() == "QListView"
        assert popup_view.objectName() == f"{combo.objectName()}PopupView"
        assert popup_view.hasMouseTracking()
        assert popup_view.viewport().hasMouseTracking()
        assert popup_view.styleSheet() == COMBO_POPUP_STYLE

        selected_index = combo.currentIndex()
        if combo.count() > 1:
            highlighted_index = combo.model().index(
                (selected_index + 1) % combo.count(), 0
            )
            popup_view.setCurrentIndex(highlighted_index)
            assert combo.currentIndex() == selected_index

    window.close()


def test_editable_project_combo_resizes_for_loaded_name(app: QApplication) -> None:
    window = MainWindow()
    window.resize(1280, 812)
    window.show()
    _process_events()

    loaded_name = "Loaded multilayer RCWA characterization study"
    window.project_name.setCurrentText(loaded_name)
    window._refresh_combo_popup_widths()
    required_width = window.project_name.fontMetrics().horizontalAdvance(loaded_name)
    setup_card = window.findChild(QWidget, "projectSetupCard")
    combo_rect = window.project_name.rect().translated(
        window.project_name.mapTo(setup_card, window.project_name.rect().topLeft())
    )

    assert window.project_name.view().minimumWidth() >= required_width
    assert setup_card.rect().contains(combo_rect)
    assert window.project_name.toolTip() == loaded_name

    window.close()


def test_project_stack_header_fits_thickness_label_without_horizontal_overflow(
    app: QApplication,
) -> None:
    window = MainWindow()
    window.resize(1280, 812)
    window.show()
    _process_events()

    table = window.findChild(QTableWidget, "projectStackSummaryTable")
    header = table.horizontalHeader()
    thickness_column = 1
    thickness_label = table.horizontalHeaderItem(thickness_column).text()
    required_width = table.fontMetrics().horizontalAdvance(thickness_label) + 18

    assert header.sectionSize(thickness_column) >= required_width
    assert table.horizontalScrollBar().maximum() == 0
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    window.close()


def test_project_page_scroll_reaches_result_canvas_at_packaged_size(
    app: QApplication,
) -> None:
    window = MainWindow()
    window.resize(1280, 812)
    window.show()
    _process_events()

    assert window.workspace.count() == 7
    assert window._pages["Project"].objectName() == "projectPage"
    scroll_area = window.workspace.widget(0)
    assert isinstance(scroll_area, QScrollArea)
    assert scroll_area.objectName() == "projectScrollArea"
    assert scroll_area.widget() is window._pages["Project"]
    assert scroll_area.widgetResizable() is True
    assert (
        scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )

    dashboard = window.findChild(QWidget, "dashboardPlotPreview")
    ancestor = dashboard.parentWidget()
    while ancestor is not None and not isinstance(ancestor, QScrollArea):
        ancestor = ancestor.parentWidget()
    assert ancestor is scroll_area
    assert (
        scroll_area.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )

    vertical_bar = scroll_area.verticalScrollBar()
    if vertical_bar.maximum() > 0:
        vertical_bar.setValue(vertical_bar.maximum())
        _process_events()
        assert vertical_bar.value() == vertical_bar.maximum()
        content_bottom = scroll_area.widget().mapTo(
            scroll_area.viewport(), scroll_area.widget().rect().bottomLeft()
        )
        assert scroll_area.viewport().rect().contains(content_bottom)

    scroll_area.ensureWidgetVisible(dashboard)
    _process_events()

    viewport_rect = scroll_area.viewport().rect()
    top_left = dashboard.mapTo(scroll_area.viewport(), dashboard.rect().topLeft())
    dashboard_rect = dashboard.rect().translated(top_left)
    assert viewport_rect.intersects(dashboard_rect)

    window.close()


def test_shell_contains_editable_legacy_stack_and_scientific_controls(
    app: QApplication,
) -> None:
    window = MainWindow()
    window.show()
    _process_events()

    layer_table = window.findChild(QTableWidget, "layerTable")
    assert layer_table.rowCount() == 2
    assert [layer_table.item(0, 1).text(), layer_table.item(1, 1).text()] == [
        "0.182",
        "0.120",
    ]
    assert [layer_table.item(0, 2).text(), layer_table.item(1, 2).text()] == [
        "1.781",
        "1.650",
    ]
    assert window.findChild(QTableWidget, "projectStackSummaryTable").rowCount() == 2
    assert window.findChild(QComboBox, "methodCombo").currentText() == "S"
    assert window.findChild(QComboBox, "interfaceCombo").currentText() == "sin"
    assert window.findChild(QComboBox, "sourceCombo").currentText() == "Analytic"
    assert window.findChild(QComboBox, "modeCombo").currentText() == "E"
    assert window.findChild(QSpinBox, "nxInput").value() == 128
    assert window.findChild(QSpinBox, "nzInput").value() == 5
    assert "Fourier orders" in window.findChild(QLabel, "gridSummaryLabel").text()
    assert window.findChild(QLabel, "metricDeviceValue").text() == "2 layers"
    assert "302 nm total" in window.findChild(QLabel, "metricDeviceDetail").text()
    assert window.findChild(QLabel, "metricSweepValue").text() == "500-520 nm"
    assert "0-5 deg, 3 pts, S/E" in window.findChild(QLabel, "metricSweepDetail").text()
    science_summary = window.findChild(QLabel, "projectScienceSummaryLabel").text()
    assert "period 0.32 um" in science_summary
    assert "height 0.154 um" in science_summary
    assert "substrate n 1.516" in science_summary
    assert "harmonics 2" in science_summary
    assert "Nx 128, Nz 5" in science_summary
    assert (
        "sin profile"
        in window.findChild(QWidget, "projectDevicePreview").accessibleName()
    )
    assert "configured" in window.nav_list.item(2).text()
    assert window.optimization_objective.currentText() == "R(+1)"
    assert window.optimization_generations.value() == 100
    assert window.optimization_population.value() == 15
    assert window.optimization_bounds_table.rowCount() == 3
    assert [
        window.optimization_bounds_table.item(row, 1).text() for row in range(3)
    ] == ["0.6", "0.01", "0.2"]
    assert [
        window.optimization_bounds_table.item(row, 2).text() for row in range(3)
    ] == ["5", "0.5", "1"]
    assert window.findChild(QDoubleSpinBox, "fdfdNresInput").value() == 20.0
    assert window.findChild(QSpinBox, "fdfdPeriodNumInput").value() == 11
    assert window.findChild(QSpinBox, "fdfdNpmlXInput").value() == 20
    assert window.findChild(QSpinBox, "fdfdNpmlYInput").value() == 20
    assert "Export unlocks after a completed run" in window.results_text.toPlainText()

    window.add_layer_button.click()
    _process_events()
    assert layer_table.rowCount() == 3
    window.remove_layer_button.click()
    _process_events()
    assert layer_table.rowCount() == 2

    window.close()


def test_interface_selection_refreshes_distinct_live_profile_geometry(
    app: QApplication,
) -> None:
    window = MainWindow()
    window.resize(1280, 812)
    window.show()
    _process_events()

    guide = window.findChild(QLabel, "interfaceGuideLabel")
    assert guide is not None
    for interface, description in PROFILE_INTERFACE_DESCRIPTIONS.items():
        assert interface in guide.text()
        assert description in guide.text()

    signatures: set[bytes] = set()
    for interface, description in PROFILE_INTERFACE_DESCRIPTIONS.items():
        window.interface_combo.setCurrentText(interface)
        _process_events()

        project_geometry = window.project_device_plot.profile_geometry()
        device_geometry = window.structure_plot.profile_geometry()
        assert project_geometry is not None
        assert device_geometry is not None
        np.testing.assert_array_equal(project_geometry[0], device_geometry[0])
        np.testing.assert_array_equal(project_geometry[1], device_geometry[1])
        assert interface in window.project_device_plot.accessibleName()
        assert description in window.project_device_plot.accessibleName()
        assert description in window.interface_combo.toolTip()
        signatures.add(np.round(project_geometry[1], decimals=12).tobytes())

    assert len(signatures) == len(PROFILE_INTERFACE_DESCRIPTIONS)
    window.close()


def test_project_round_trip_preserves_gui_state(
    app: QApplication, tmp_path: Path
) -> None:
    window = MainWindow()
    window.show()
    window.project_name.setCurrentText("Round Trip Study")
    window.period_um.setValue(0.42)
    window.height_um.setValue(0.08)
    window.harmonics.setValue(3)
    window.compatibility_mode_combo.setCurrentText("corrected")
    window.interface_combo.setCurrentText("DE4")
    window.interface_smooth.setChecked(True)
    window.supergauss_sigma.setValue(0.09)
    window.periodic_stack.setChecked(True)
    window.periodic_count.setValue(5)
    window.flat_substrate.setChecked(True)
    window.calc_fresnel.setChecked(True)
    window.result_view_combo.setCurrentText("Wavelength middle")
    window.nx_input.setValue(64)
    window.nz_input.setValue(4)
    window.layer_table.item(0, 1).setText("0.200")
    window.layer_table.item(1, 2).setText("1.700")
    _process_events()

    project_path = window.save_project_to(tmp_path / "roundtrip")
    assert project_path.suffix == ".zenscat"

    restored = MainWindow()
    restored.open_project_from(project_path)
    _process_events()

    assert restored.project_name.currentText() == "Round Trip Study"
    assert restored.period_um.value() == pytest.approx(0.42)
    assert restored.height_um.value() == pytest.approx(0.08)
    assert restored.harmonics.value() == 3
    assert restored.compatibility_mode_combo.currentText() == "corrected"
    assert restored.interface_combo.currentText() == "DE4"
    assert restored.interface_smooth.isChecked() is True
    assert restored.supergauss_sigma.value() == pytest.approx(0.09)
    assert restored.periodic_stack.isChecked() is True
    assert restored.periodic_count.value() == 5
    assert restored.flat_substrate.isChecked() is True
    assert restored.calc_fresnel.isChecked() is True
    assert restored.result_view_combo.currentText() == "Wavelength middle"
    assert restored.nx_input.value() == 64
    assert restored.nz_input.value() == 4
    assert restored.layer_table.item(0, 1).text() == "0.2"
    assert restored.layer_table.item(1, 2).text() == "1.7"

    restored.close()
    window.close()


@pytest.mark.parametrize(
    ("workflow", "nav_page"),
    [
        ("casual_rcwa", "RCWA Sweep"),
        ("custom_import_rcwa", "RCWA Sweep"),
        ("optimization", "Optimize"),
        ("fdfd_fields", "FDFD Fields"),
        ("casual_phc_rcwa", "RCWA Sweep"),
        ("harmonic_convergence", "RCWA Sweep"),
        ("phc_fdfd_fields", "FDFD Fields"),
    ],
)
def test_project_round_trip_preserves_active_workflow(
    app: QApplication, tmp_path: Path, workflow: str, nav_page: str
) -> None:
    window = MainWindow()
    window.show()
    window.project_name.setCurrentText(f"{workflow} Study")
    if workflow == "custom_import_rcwa":
        window.source_combo.setCurrentText("Import")
        window._import_path = tmp_path / "device.py.npy"
    elif workflow == "optimization":
        _select_nav(window, "Optimize")
    elif workflow == "fdfd_fields":
        _select_nav(window, "FDFD Fields")
    elif workflow == "phc_fdfd_fields":
        window.phc_shape_combo.setCurrentText("PhC Rectangle")
        window.fdfd_geometry_combo.setCurrentText("Selected PhC")
        _select_nav(window, "FDFD Fields")
    elif workflow == "harmonic_convergence":
        _select_nav(window, "RCWA Sweep")
        window.rcwa_run_mode_combo.setCurrentText("Harmonic convergence")
    elif workflow == "casual_phc_rcwa":
        _select_nav(window, "RCWA Sweep")
        window.phc_shape_combo.setCurrentText("PhC Rectangle")
    else:
        _select_nav(window, "RCWA Sweep")
    _process_events()

    project_path = window.save_project_to(tmp_path / workflow)
    document = ProjectDocument.load(project_path)

    restored = MainWindow()
    restored.open_project_from(project_path)
    _process_events()

    assert document.workflow == workflow
    assert _nav_label(restored) == nav_page
    if workflow == "custom_import_rcwa":
        assert restored.source_combo.currentText() == "Import"
        assert restored._imported_device is None
        assert restored._import_path == tmp_path / "device.py.npy"
    if workflow == "casual_phc_rcwa":
        assert restored.phc_shape_combo.currentText() == "PhC Rectangle"
    if workflow == "harmonic_convergence":
        assert restored.rcwa_run_mode_combo.currentText() == "Harmonic convergence"
    if workflow == "phc_fdfd_fields":
        assert restored.fdfd_geometry_combo.currentText() == "Selected PhC"

    restored.close()
    window.close()


def test_project_open_preserves_import_path_without_pickle_trust(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device_path = tmp_path / "device.py.npy"
    device_path.write_bytes(b"pickle-backed placeholder")
    project_path = ProjectDocument(
        name="Pickle Import",
        workflow="custom_import_rcwa",
        configuration={
            "source_mode": "Import",
            "import_path": str(device_path),
        },
    ).save(tmp_path / "pickle-import")
    calls: list[dict[str, Any]] = []

    def fail_if_called(path: str | Path, **kwargs: Any) -> object:
        calls.append({"path": path, **kwargs})
        raise AssertionError("project open must not load imported devices")

    monkeypatch.setattr("zenscat.legacy_io.load_imported_device", fail_if_called)

    window = MainWindow()
    window.open_project_from(project_path)
    _process_events()

    assert calls == []
    assert window.source_combo.currentText() == "Import"
    assert window._import_path == device_path
    assert window._imported_device is None
    assert _nav_label(window) == "RCWA Sweep"

    window.close()


def test_result_canvases_start_as_honest_empty_states(app: QApplication) -> None:
    window = MainWindow()
    window.show()
    _process_events()

    for object_name in (
        "dashboardPlotPreview",
        "sweepPlotPreview",
        "optimizePlotPreview",
        "fieldPlotPreview",
        "resultsPlotPreview",
    ):
        assert (
            "No computed results"
            in window.findChild(QWidget, object_name).accessibleName()
        )

    assert "No computed results" in window.results_text.toPlainText()

    window.close()


def test_imported_numpy_device_dispatches_import_workflow_and_exports(
    app: QApplication, tmp_path: Path
) -> None:
    x_um = np.linspace(-0.16, 0.16, 32)
    device_path = tmp_path / "device.npy"
    np.save(
        device_path,
        {
            "ER": np.vstack(
                [
                    np.full(x_um.shape, 1.8**2),
                    np.full(x_um.shape, 1.5**2),
                ]
            ),
            "sub_L": np.asarray([0.08, 0.04]),
            "x": x_um,
            "Lx": 0.32,
        },
    )

    window = MainWindow()
    window.show()
    _set_small_run(window)
    window.load_import_device_from(device_path)
    _process_events()

    assert window.source_combo.currentText() == "Import"
    assert "device.npy" in window.import_source_label.text()
    assert "imported device" in window.structure_plot.accessibleName()

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True

    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["workflow"] == "custom_import_rcwa"
    assert "Computed import RCWA results" in window.results_text.toPlainText()

    destination = window.export_results_to(tmp_path / "import-result")
    assert (destination / "TRN.mat").is_file()
    assert (destination / "REF.mat").is_file()
    assert not (destination / "Params.mat").exists()

    window.close()


def test_validation_run_small_grid_and_export_gating(
    app: QApplication, tmp_path: Path
) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)

    window.validate_button.click()
    _process_events()

    assert window.run_button.isEnabled() is True
    assert window.export_button.isEnabled() is False

    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window.run_button.isEnabled() is True
    assert window.cancel_button.isEnabled() is False
    assert window.export_button.isEnabled() is True
    assert "Computed analytic RCWA results" in window.results_text.toPlainText()
    assert "computed Transmission 0" in window.results_plot.accessibleName()
    navigation_states = {
        str(window.nav_list.item(index).data(Qt.ItemDataRole.UserRole)):
        window.nav_list.item(index).text()
        for index in range(window.nav_list.count())
    }
    assert navigation_states["Results"] == "Results\nready"
    assert navigation_states["Jobs"] == "Jobs\nidle"

    destination = window.export_results_to(tmp_path / "gui-result")
    assert (destination / "TRN.mat").is_file()
    assert (destination / "REF.mat").is_file()
    assert (destination / "manifest.json").is_file()
    assert (destination / "Data.txt").is_file()
    assert (destination / "RCWA_plot_data.csv").is_file()
    assert (destination / "Selected_result.png").is_file()
    assert (destination / "Selected_result.svg").is_file()

    window.close()


def test_gui_export_uses_default_no_overwrite(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_save_legacy_result_bundle(
        directory: str | Path, bundle: object, **kwargs: Any
    ) -> Path:
        captured["directory"] = Path(directory)
        captured["bundle"] = bundle
        captured["kwargs"] = kwargs
        Path(directory).mkdir(parents=True, exist_ok=True)
        return Path(directory)

    monkeypatch.setattr(
        "zenscat.legacy_io.save_legacy_result_bundle",
        fake_save_legacy_result_bundle,
    )

    window = MainWindow()
    window.show()
    window._last_inputs = window.collect_inputs()
    window._last_run = _LegacyExportRun(bundle=object())

    destination = window.export_results_to(tmp_path / "safe-export")

    assert destination == tmp_path / "safe-export"
    assert captured["kwargs"] == {"compatibility_mode": "modern"}

    window.close()


def test_gui_export_protects_existing_bundle_content(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "existing-result"
    destination.mkdir()
    note = destination / "user-note.txt"
    note.write_text("keep", encoding="utf-8")

    window = MainWindow()
    window.show()
    window._last_inputs = window.collect_inputs()
    window._last_run = _LegacyExportRun()

    with pytest.raises(FileExistsError):
        window.export_results_to(destination)
    assert note.read_text(encoding="utf-8") == "keep"

    monkeypatch.setattr(
        "zenscat.gui.main_window.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )
    window.export_results()

    assert "result directory is not empty" in window.problems_text.toPlainText()
    assert "Export failed" in window.job_status.text()
    assert note.read_text(encoding="utf-8") == "keep"

    window.close()


def test_optimize_page_runs_real_bounded_objective(
    app: QApplication, tmp_path: Path
) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)
    window.nav_list.setCurrentRow(3)
    _set_tiny_optimization(window)
    _process_events()

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True

    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["workflow"] == "optimization"
    assert window._last_run.final_run.metadata["workflow"] == "casual_rcwa"
    assert "best fitness" in window.optimize_plot.accessibleName()
    assert "Computed analytic RCWA results" in window.results_text.toPlainText()

    destination = window.export_results_to(tmp_path / "opt-result")
    assert (destination / "TRN.mat").is_file()
    assert (destination / "Params.mat").is_file()

    window.close()


def test_optimize_page_updates_merit_history_during_progress(app: QApplication) -> None:
    window = MainWindow()
    window.show()
    _select_nav(window, "Optimize")

    assert "No computed results" in window.optimize_plot.accessibleName()

    window._on_optimization_progress(
        OptimizationProgress(
            generation=7,
            evaluations=42,
            best_fitness=0.125,
            best_parameters=np.asarray([0.32, 0.15, 0.18], dtype=np.float64),
        )
    )
    _process_events()

    assert "computed best fitness" in window.optimize_plot.accessibleName()
    assert "Generation 7" in window.optimization_status.text()
    assert "0.125" in window.optimization_status.text()

    window.close()


def test_optimize_worker_streams_and_retains_partial_merit_history(
    app: QApplication,
) -> None:
    class StreamingOptimizationService:
        def validate(self, _inputs):
            return ValidationResult(True, ("ok",), "ready")

        def run(self, _inputs, cancel_event, progress):
            merit_progress = progress.optimization
            for generation, fitness in enumerate((-0.1, -0.2, -0.3), start=1):
                merit_progress(
                    SimpleNamespace(
                        generation=generation,
                        evaluations=generation * 5,
                        current_fitness=fitness + 0.05,
                        best_fitness=fitness,
                    )
                )
                progress(generation, 100)
                time.sleep(0.01)
            while not cancel_event.wait(0.002):
                pass
            raise RuntimeError("simulation cancelled")

    window = MainWindow(service=StreamingOptimizationService())
    window.show()
    _select_nav(window, "Optimize")
    window.validate_button.click()
    _process_events()

    window.run_button.click()
    _wait_until(lambda: len(window._optimization_history) == 3)

    assert "computed best fitness (live)" in window.optimize_plot.accessibleName()
    assert np.allclose(window.optimize_plot._series, [-0.1, -0.2, -0.3])
    assert "Generation 3/100" in window.optimization_status.text()
    assert "current -0.25" in window.optimization_status.text()
    assert "best -0.3" in window.optimization_status.text()

    window.cancel_button.click()
    _wait_until(lambda: window._thread is None)

    assert "Partial merit history retained" in window.optimization_status.text()
    assert np.allclose(window.optimize_plot._series, [-0.1, -0.2, -0.3])
    window.close()


def test_fdfd_page_runs_fields_updates_view_and_exports(
    app: QApplication, tmp_path: Path
) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)
    window.nav_list.setCurrentRow(4)
    _set_tiny_fdfd(window)
    _process_events()

    assert "510 nm" in window.fdfd_setup_summary.text()
    assert "period 0.32 um" in window.fdfd_setup_summary.text()
    assert "NPML 1/1" in window.fdfd_setup_summary.text()

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True

    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["workflow"] == "fdfd_fields"
    assert "abs(f) heatmap" in window.field_plot.accessibleName()
    assert "Computed FDFD field results" in window.results_text.toPlainText()

    window.fdfd_field_combo.setCurrentText("real(f)")
    _process_events()
    assert "real(f) heatmap" in window.field_plot.accessibleName()
    window.fdfd_field_combo.setCurrentText("ER2 + contours")
    _process_events()
    assert "ER2 + contours heatmap" in window.field_plot.accessibleName()

    destination = window.export_results_to(tmp_path / "fdfd-result")
    assert (destination / "Field.mat").is_file()
    assert (destination / "ER2.mat").is_file()
    assert (destination / "manifest.json").is_file()

    window.close()


def test_phc_rectangle_workflow_runs_and_result_selectors_update(
    app: QApplication,
) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)
    window.nav_list.setCurrentRow(2)
    window.phc_shape_combo.setCurrentText("PhC Rectangle")
    window.nx_input.setValue(16)
    window.nz_input.setValue(4)
    _process_events()

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True

    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["workflow"] == "casual_phc_rcwa"
    assert window._last_run.metadata["repeat_mode"] == "legacy"
    assert "Computed PhC RCWA results" in window.results_text.toPlainText()

    window.result_family_combo.setCurrentText("Energy")
    window.result_order_combo.setCurrentText("sum")
    _process_events()
    assert "Energy sum" in window.results_plot.accessibleName()

    window.close()


def test_harmonic_convergence_runs_and_recommends_from_gui(app: QApplication) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)
    _select_nav(window, "RCWA Sweep")
    window.rcwa_run_mode_combo.setCurrentText("Harmonic convergence")
    window.convergence_max_harmonics.setValue(2)
    window.convergence_tolerance.setValue(1.0)

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True
    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert len(window._last_run.samples) == 2
    assert window._last_run.metadata["workflow"] == "harmonic_convergence"
    assert "Computed harmonic convergence" in window.results_text.toPlainText()
    assert "harmonic energy error" in window.sweep_plot.accessibleName()
    window.close()


def test_dual_fdfd_gui_shows_and_exports_both_polarizations(
    app: QApplication, tmp_path: Path
) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)
    _select_nav(window, "FDFD Fields")
    _set_tiny_fdfd(window)
    window.fdfd_polarization_combo.setCurrentText("Both E + H")

    window.validate_button.click()
    _process_events()
    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["workflow"] == "dual_fdfd_fields"
    assert "computed abs(f) heatmap" in window.field_plot.accessibleName()
    assert "computed abs(f) heatmap" in window.field_plot_h.accessibleName()
    assert window.field_plot_h.isVisible()
    destination = window.export_results_to(tmp_path / "dual-fdfd")
    assert (destination / "E" / "Field.mat").is_file()
    assert (destination / "H" / "Field.mat").is_file()
    assert (destination / "manifest.json").is_file()
    window.close()


def test_corrected_phc_fdfd_runs_from_gui(app: QApplication) -> None:
    window = MainWindow()
    window.show()
    _set_small_run(window)
    window.phc_shape_combo.setCurrentText("PhC Honeycomb")
    _select_nav(window, "FDFD Fields")
    _set_tiny_fdfd(window)
    window.fdfd_geometry_combo.setCurrentText("Selected PhC")
    window.compatibility_mode_combo.setCurrentText("corrected")

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True
    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["workflow"] == "phc_fdfd_fields"
    assert window._last_run.metadata["legacy_exact"] is False
    assert "Computed FDFD field results" in window.results_text.toPlainText()
    window.close()


def test_optimization_checkpoint_is_written_and_loadable(
    app: QApplication, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "optimizer-checkpoint.json"
    window = MainWindow()
    window.show()
    _set_small_run(window)
    _select_nav(window, "Optimize")
    _set_tiny_optimization(window)
    window.optimization_checkpoint.setCurrentText(str(checkpoint_path))

    window.validate_button.click()
    _process_events()
    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["generations_completed"] >= 1
    window.optimization_resume.setChecked(True)
    request = window._service.build_request(window.collect_inputs())
    assert request.checkpoint["best_parameters"] == payload["best_parameters"]
    window.close()


def test_external_cli_backend_executes_registered_solver(
    app: QApplication, tmp_path: Path
) -> None:
    solver = tmp_path / "external_solver.py"
    solver.write_text(
        """
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument('--zenscat-request')
parser.add_argument('--zenscat-result')
args = parser.parse_args()
result = {
  'schema': 'zenscat.external-result', 'kind': 'rcwa',
  'wavelengths_m': [5.1e-7], 'angles_rad': [0.0],
  'transmission': {'minus_1': [[0.0]], 'plus_1': [[0.0]], 'TRN0': [[0.7]], 'sum': [[0.7]]},
  'reflection': {'minus_1': [[0.0]], 'plus_1': [[0.0]], 'REF0': [[0.3]], 'sum': [[0.3]]},
  'metadata': {'workflow': 'external_rcwa'}
}
with open(args.zenscat_result, 'w', encoding='utf-8') as stream:
    json.dump(result, stream)
""",
        encoding="utf-8",
    )
    window = MainWindow()
    window.show()
    _set_small_run(window)
    window.backend_combo.setCurrentText("External CLI")
    window.external_command_combo.setCurrentText(f'"{sys.executable}" "{solver}"')

    window.validate_button.click()
    _process_events()
    assert window.run_button.isEnabled() is True
    window.run_button.click()
    _wait_until(lambda: window._thread is None and window._last_run is not None)

    assert window._last_run.metadata["backend"] == "external"
    assert window._last_run.transmission.TRN0.item() == pytest.approx(0.7)
    window.close()


def test_release_build_disables_pyqt_fallback(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    script = """
import importlib.abc
import sys

class BlockPySide(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "PySide6" or fullname.startswith("PySide6."):
            raise ImportError("blocked PySide6")
        return None

sys.meta_path.insert(0, BlockPySide())
import zenscat.gui.qt_compat
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "ZENSCAT_RELEASE_BUILD": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "release builds must bundle PySide6" in result.stderr


def test_app_smoke_hook_exits_after_show() -> None:
    repo_root = Path(__file__).parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "ZENSCAT_SMOKE_TEST": "1",
        "ZENSCAT_QT_BINDING": QT_BINDING,
    }
    result = subprocess.run(
        [sys.executable, "-m", "zenscat.app"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cancellation_contract_sets_event_and_preserves_inputs(
    app: QApplication,
) -> None:
    class BlockingService:
        def __init__(self) -> None:
            self.cancel_seen = False

        def validate(self, _inputs):
            return ValidationResult(True, ("ok",), "ready")

        def run(self, _inputs, cancel_event, progress):
            for index in range(200):
                progress(index, 200)
                QCoreApplication.processEvents()
                if cancel_event.is_set():
                    self.cancel_seen = True
                    raise RuntimeError("simulation cancelled")
                time.sleep(0.002)
            raise AssertionError("cancel was not requested")

    service = BlockingService()
    window = MainWindow(service=service)
    window.show()
    _set_small_run(window)
    window.validate_button.click()
    _process_events()

    window.run_button.click()
    _wait_until(lambda: window.cancel_button.isEnabled())
    window.cancel_button.click()
    _wait_until(lambda: window._thread is None)

    assert service.cancel_seen is True
    assert window.run_button.isEnabled() is True
    assert window.export_button.isEnabled() is False
    assert "cancelled" in window.results_text.toPlainText().lower()
    assert window.layer_table.item(0, 1).text() == "0.182"

    window.close()
