"""Precision-lab Qt shell for ZenScat workflows."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from tempfile import gettempdir

import numpy as np

from .qt_compat import (
    QAbstractSpinBox,
    QAction,
    QApplication,
    QColor,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QObject,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPushButton,
    QRectF,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QThread,
    QVBoxLayout,
    QWidget,
    Signal,
)

NAV_ITEMS = (
    "Project",
    "Device",
    "RCWA Sweep",
    "Optimize",
    "FDFD Fields",
    "Results",
    "Jobs",
)

NAV_STATE = {
    "Project": "ready",
    "Device": "configured",
    "RCWA Sweep": "ready",
    "Optimize": "locked",
    "FDFD Fields": "locked",
    "Results": "empty",
    "Jobs": "idle",
}

PROJECT_WORKFLOW_PAGES = {
    "casual_rcwa": "RCWA Sweep",
    "custom_import_rcwa": "RCWA Sweep",
    "optimization": "Optimize",
    "fdfd_fields": "FDFD Fields",
    "casual_phc_rcwa": "RCWA Sweep",
}

PROFILE_INTERFACE_DESCRIPTIONS = {
    "sin": "sinusoid",
    "DE1": "trapezium",
    "DE4": "soft trapezium / super-Gaussian",
    "tri": "triangle",
}

PROFILE_LAYER_COLORS = (
    "#2b8fb8",
    "#dfa044",
    "#7c6bb3",
    "#4a9a78",
    "#c36c75",
)

APP_STYLE = """
QMainWindow, QWidget#rootShell {
    background: #eef2f5;
    color: #17212b;
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}
QFrame#topStatusBar {
    background: #fbfcfd;
    border: 0;
    border-bottom: 1px solid #cfd8df;
}
QLabel#projectStatusLabel {
    font-size: 16px;
    font-weight: 700;
    color: #0d2436;
}
QLabel#backendStatusLabel, QLabel#jobStatusLabel {
    color: #475867;
}
QListWidget#navigationList {
    background: #142331;
    color: #d9e3eb;
    border: 0;
    padding: 10px 8px;
    outline: 0;
}
QListWidget#navigationList::item {
    border-radius: 7px;
    margin: 3px 0;
    padding: 10px 12px;
}
QListWidget#navigationList::item:selected {
    background: #24516f;
    color: #ffffff;
}
QFrame.card, QGroupBox {
    background: #ffffff;
    border: 1px solid #d4dde4;
    border-radius: 8px;
}
QGroupBox {
    margin-top: 10px;
    padding: 14px 12px 12px 12px;
    font-weight: 700;
    color: #263846;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QLabel.title {
    font-size: 21px;
    font-weight: 800;
    color: #10283a;
}
QLabel.sectionTitle {
    font-size: 15px;
    font-weight: 800;
    color: #203442;
}
QLabel.muted {
    color: #607282;
}
QLabel.badge {
    background: #e7f1f6;
    border: 1px solid #bdd2df;
    border-radius: 9px;
    color: #24485f;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
}
QLabel[locked="true"] {
    background: #f1e9e4;
    border-color: #d4b8a8;
    color: #7a4732;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #b8c6d0;
    border-radius: 6px;
    padding: 7px 13px;
    font-weight: 700;
}
QPushButton#validateButton {
    background: #1f6f8b;
    color: #ffffff;
    border-color: #1f6f8b;
}
QPushButton:disabled {
    color: #8c9aa5;
    background: #edf1f4;
}
QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTableWidget {
    background: #fbfcfd;
    border: 1px solid #c9d3dc;
    border-radius: 5px;
    padding: 5px;
}
QTableWidget {
    gridline-color: #d9e1e8;
    selection-background-color: #d9ebf3;
}
QHeaderView::section {
    background: #e8eef3;
    color: #344756;
    border: 0;
    border-right: 1px solid #d0d9e1;
    padding: 7px;
    font-weight: 700;
}
QTabWidget::pane {
    border: 0;
}
QTabBar::tab {
    background: #e4ebf0;
    border: 1px solid #cbd6df;
    padding: 8px 10px;
}
QTabBar::tab:selected {
    background: #ffffff;
    border-bottom-color: #ffffff;
}
"""

COMBO_POPUP_STYLE = """
QListView {
    background-color: #fbfcfd;
    color: #17212b;
    border: 1px solid #aebfcb;
    border-radius: 6px;
    padding: 4px;
    outline: 0;
    show-decoration-selected: 1;
    selection-background-color: #b9dce9;
    selection-color: #17212b;
}
QListView::item {
    min-height: 24px;
    padding: 3px 8px;
    border-radius: 4px;
}
QListView::item:hover {
    background-color: #b9dce9;
    color: #17212b;
    font-weight: 600;
}
QListView::item:selected {
    background-color: #b9dce9;
    color: #17212b;
    font-weight: 600;
}
"""


@dataclass(frozen=True)
class ShellInputs:
    project_name: str
    backend: str
    workflow: str
    source_mode: str
    import_path: str | None
    imported_device: object | None
    matrix_method: str
    interface: str
    polarization: str
    distribution: str
    period_um: float
    height_um: float
    harmonics: int
    layer_count: int
    n_superstrate: float
    n_substrate: float
    wavelength_start_nm: float
    wavelength_stop_nm: float
    angle_start_deg: float
    angle_stop_deg: float
    sweep_points: int
    nx: int
    nz: int
    params: tuple[float, ...]
    result_family: str
    result_order: str
    optimization_objective: str
    optimization_lower_bounds: tuple[float, ...]
    optimization_upper_bounds: tuple[float, ...]
    optimization_generations: int
    optimization_population: int
    fdfd_interface: str
    fdfd_palette: str
    fdfd_nres: float
    fdfd_period_num: int
    fdfd_npml: tuple[int, int]
    fdfd_spacer_um: tuple[float, float]
    phc_shape: str
    phc_repeat_mode: str
    phc_layer_count: int
    phc_pitch_um: float
    phc_background_n: float
    phc_inclusion_n: float
    phc_wx: float
    phc_wy: float
    phc_rotation_deg: float
    phc_ax: float
    phc_ay: float
    phc_radius: float


@dataclass(frozen=True)
class RunResult:
    summary: str
    run: object


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    messages: tuple[str, ...]
    summary: str


class PlotPreview(QWidget):
    """Small deterministic plot surface drawn with Qt primitives."""

    def __init__(
        self,
        object_name: str,
        title: str,
        x_label: str,
        y_label: str,
        mode: str = "empty",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self._title = title
        self._x_label = x_label
        self._y_label = y_label
        self._mode = mode
        self._watermark = "No computed results"
        self._transmission: np.ndarray | None = None
        self._reflection: np.ndarray | None = None
        self._series: np.ndarray | None = None
        self._series_label: str | None = None
        self._heatmap: np.ndarray | None = None
        self._heatmap_label: str | None = None
        self._profile_thicknesses = np.asarray([0.182, 0.120], dtype=np.float64)
        self._profile_indices = np.asarray([1.781, 1.650], dtype=np.float64)
        self._profile_interface = "sin"
        self._profile_x_um: np.ndarray | None = None
        self._profile_z_um: np.ndarray | None = None
        self._profile_period_um = 0.32
        self._profile_height_um = 0.154
        self._profile_n_superstrate = 1.0
        self._profile_n_substrate = 1.516
        self.setAccessibleName(
            title if mode == "stack" else f"{title}: {self._watermark}"
        )
        self.setMinimumHeight(260 if mode == "stack" else 210)

    def set_profile(
        self,
        thicknesses: np.ndarray,
        indices: np.ndarray,
        interface: str,
        *,
        x_um: np.ndarray | None = None,
        z_um: np.ndarray | None = None,
        period_um: float = 0.32,
        height_um: float = 0.154,
        n_superstrate: float = 1.0,
        n_substrate: float = 1.516,
    ) -> None:
        self._profile_thicknesses = np.asarray(thicknesses, dtype=np.float64)
        self._profile_indices = np.asarray(indices, dtype=np.float64)
        self._profile_interface = interface
        self._profile_x_um = (
            None if x_um is None else np.asarray(x_um, dtype=np.float64).copy()
        )
        self._profile_z_um = (
            None if z_um is None else np.asarray(z_um, dtype=np.float64).copy()
        )
        if (self._profile_x_um is None) != (self._profile_z_um is None):
            raise ValueError("profile x and z coordinates must be provided together")
        if (
            self._profile_x_um is not None
            and self._profile_x_um.shape != self._profile_z_um.shape
        ):
            raise ValueError("profile x and z coordinates must have matching shapes")
        self._profile_period_um = float(period_um)
        self._profile_height_um = float(height_um)
        self._profile_n_superstrate = float(n_superstrate)
        self._profile_n_substrate = float(n_substrate)
        description = PROFILE_INTERFACE_DESCRIPTIONS.get(interface, interface)
        index_text = ", ".join(f"{value:.4g}" for value in self._profile_indices)
        accessible = (
            f"{self._title}: {interface} profile ({description}); "
            f"period {self._profile_period_um:.4g} um; "
            f"height {self._profile_height_um:.4g} um; "
            f"{self._profile_thicknesses.size} layers; indices {index_text}"
        )
        self.setAccessibleName(accessible)
        self.setToolTip(accessible)
        self.update()

    def profile_geometry(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a defensive copy of the currently displayed analytic curve."""

        if self._profile_x_um is None or self._profile_z_um is None:
            return None
        return self._profile_x_um.copy(), self._profile_z_um.copy()

    def clear_results(self) -> None:
        self._transmission = None
        self._reflection = None
        self._series = None
        self._series_label = None
        self._heatmap = None
        self._heatmap_label = None
        if self._mode != "stack":
            self.setAccessibleName(f"{self._title}: {self._watermark}")
        self.update()

    def set_results(self, transmission: np.ndarray, reflection: np.ndarray) -> None:
        self._transmission = np.asarray(transmission, dtype=np.float64)
        self._reflection = np.asarray(reflection, dtype=np.float64)
        self.setAccessibleName(f"{self._title}: computed TRN0 and REF0")
        self.update()

    def set_series(self, values: np.ndarray, label: str) -> None:
        self._transmission = None
        self._reflection = None
        self._heatmap = None
        self._heatmap_label = None
        self._series = np.asarray(values, dtype=np.float64).ravel()
        self._series_label = label
        self.setAccessibleName(f"{self._title}: computed {label}")
        self.update()

    def set_heatmap(self, values: np.ndarray, label: str) -> None:
        self._transmission = None
        self._reflection = None
        self._series = None
        self._series_label = None
        heatmap = np.asarray(values)
        if np.iscomplexobj(heatmap):
            heatmap = np.abs(heatmap)
        self._heatmap = np.asarray(heatmap, dtype=np.float64)
        self._heatmap_label = label
        self.setAccessibleName(f"{self._title}: computed {label} heatmap")
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, QColor("#fbfcfd"))
        painter.setPen(QPen(QColor("#c6d2dc"), 1))
        painter.drawRoundedRect(QRectF(rect), 8, 8)

        plot = rect.adjusted(52, 74 if self._mode == "stack" else 34, -18, -38)
        painter.setPen(QPen(QColor("#9aaab7"), 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())

        painter.setPen(QPen(QColor("#dde5eb"), 1))
        for index in range(1, 4):
            x = plot.left() + plot.width() * index / 4
            y = plot.top() + plot.height() * index / 4
            painter.drawLine(int(x), plot.top(), int(x), plot.bottom())
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        painter.setPen(QPen(QColor("#223747"), 1))
        painter.drawText(
            rect.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignTop, self._title
        )
        if self._mode == "stack":
            self._draw_profile_header(painter, rect)
        painter.setPen(QPen(QColor("#607282"), 1))
        x_label = self._x_label
        y_label = self._y_label
        if self._mode == "stack":
            x_label = f"one period  {self._profile_period_um:.4g} um"
            y_label = f"height {self._profile_height_um:.4g} um"
        painter.drawText(
            rect.adjusted(14, 0, -14, -10),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            x_label,
        )
        painter.save()
        painter.translate(12, plot.center().y() + 46)
        painter.rotate(-90)
        painter.drawText(QRectF(0, 0, 150, 18), Qt.AlignmentFlag.AlignCenter, y_label)
        painter.restore()

        if self._mode == "stack":
            self._draw_profile_cross_section(painter, plot)
        elif self._heatmap is not None:
            self._draw_heatmap(painter, plot, self._heatmap)
            painter.setPen(QPen(QColor("#223747"), 1))
            painter.drawText(
                QRectF(plot),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                self._heatmap_label or "field",
            )
        elif self._series is not None:
            self._draw_series(painter, plot, self._series, QColor("#1f6f8b"))
            painter.setPen(QPen(QColor("#1f6f8b"), 2))
            painter.drawText(
                plot.left() + 10, plot.top() + 18, self._series_label or "series"
            )
        elif self._transmission is None or self._reflection is None:
            painter.setPen(QPen(QColor("#8a9aaa"), 1))
            painter.drawText(
                QRectF(plot),
                Qt.AlignmentFlag.AlignCenter,
                self._watermark,
            )
        else:
            self._draw_series(painter, plot, self._transmission, QColor("#1f6f8b"))
            self._draw_series(painter, plot, self._reflection, QColor("#b66d2d"))
            painter.setPen(QPen(QColor("#1f6f8b"), 2))
            painter.drawText(plot.left() + 10, plot.top() + 18, "TRN0")
            painter.setPen(QPen(QColor("#b66d2d"), 2))
            painter.drawText(plot.left() + 68, plot.top() + 18, "REF0")

    def _draw_profile_header(self, painter: QPainter, rect) -> None:
        description = PROFILE_INTERFACE_DESCRIPTIONS.get(
            self._profile_interface, self._profile_interface
        )
        painter.setPen(QPen(QColor("#607282"), 1))
        painter.drawText(
            QRectF(rect.left() + 14, rect.top() + 30, rect.width() - 28, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "One lateral period; color = refractive index; curves = layer interfaces.",
        )

        badge_text = f"{self._profile_interface}  |  {description}"
        badge_width = painter.fontMetrics().horizontalAdvance(badge_text) + 20
        badge = QRectF(
            rect.right() - badge_width - 14,
            rect.top() + 8,
            badge_width,
            22,
        )
        painter.fillRect(badge, QColor("#e7f1f6"))
        painter.setPen(QPen(QColor("#9ebdce"), 1))
        painter.drawRoundedRect(badge, 7, 7)
        painter.setPen(QPen(QColor("#24485f"), 1))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, badge_text)

        entries = [
            (QColor("#f1f5f7"), f"ambient n={self._profile_n_superstrate:.4g}"),
        ]
        indices = np.resize(self._profile_indices, self._profile_thicknesses.shape)
        for index, refractive_index in enumerate(indices[:3]):
            entries.append(
                (
                    QColor(PROFILE_LAYER_COLORS[index % len(PROFILE_LAYER_COLORS)]),
                    f"L{index + 1} n={refractive_index:.4g}",
                )
            )
        entries.append(
            (QColor("#718493"), f"substrate n={self._profile_n_substrate:.4g}")
        )

        x = float(rect.left() + 14)
        y = float(rect.top() + 52)
        for color, label in entries:
            label_width = painter.fontMetrics().horizontalAdvance(label)
            if x + label_width + 24 > rect.right() - 14:
                break
            swatch = QRectF(x, y, 11, 11)
            painter.fillRect(swatch, color)
            painter.setPen(QPen(QColor("#71818d"), 1))
            painter.drawRect(swatch)
            painter.setPen(QPen(QColor("#344957"), 1))
            painter.drawText(
                QRectF(x + 16, y - 2, label_width + 2, 16),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            x += label_width + 34

    def _draw_profile_cross_section(self, painter: QPainter, plot) -> None:
        thicknesses = np.maximum(self._profile_thicknesses, 1e-9)
        if self._profile_x_um is None or self._profile_z_um is None:
            x_um = np.linspace(-0.5, 0.5, 64)
            z_um = np.zeros_like(x_um)
        else:
            x_um = self._profile_x_um
            z_um = self._profile_z_um

        cumulative = np.concatenate(([0.0], np.cumsum(thicknesses)))
        boundaries = [z_um - offset for offset in cumulative]
        stack_span = max(
            float(np.max(boundaries[0]) - np.min(boundaries[-1])),
            self._profile_height_um + float(np.sum(thicknesses)),
            1e-9,
        )
        padding = 0.10 * stack_span
        z_max = float(np.max(boundaries[0])) + padding
        z_min = float(np.min(boundaries[-1])) - padding
        x_min = float(np.min(x_um))
        x_span = max(float(np.ptp(x_um)), 1e-9)
        z_span = max(z_max - z_min, 1e-9)

        px = plot.left() + (x_um - x_min) / x_span * plot.width()

        def map_y(values: np.ndarray) -> np.ndarray:
            return plot.bottom() - (values - z_min) / z_span * plot.height()

        mapped = [map_y(boundary) for boundary in boundaries]

        ambient = QPainterPath()
        ambient.moveTo(float(plot.left()), float(plot.top()))
        ambient.lineTo(float(plot.right()), float(plot.top()))
        for x_value, y_value in zip(px[::-1], mapped[0][::-1], strict=True):
            ambient.lineTo(float(x_value), float(y_value))
        ambient.closeSubpath()
        painter.fillPath(ambient, QColor("#f1f5f7"))

        for index in range(len(thicknesses)):
            band = QPainterPath()
            band.moveTo(float(px[0]), float(mapped[index][0]))
            for x_value, y_value in zip(px[1:], mapped[index][1:], strict=True):
                band.lineTo(float(x_value), float(y_value))
            for x_value, y_value in zip(px[::-1], mapped[index + 1][::-1], strict=True):
                band.lineTo(float(x_value), float(y_value))
            band.closeSubpath()
            painter.fillPath(
                band,
                QColor(PROFILE_LAYER_COLORS[index % len(PROFILE_LAYER_COLORS)]),
            )

        substrate = QPainterPath()
        substrate.moveTo(float(px[0]), float(mapped[-1][0]))
        for x_value, y_value in zip(px[1:], mapped[-1][1:], strict=True):
            substrate.lineTo(float(x_value), float(y_value))
        substrate.lineTo(float(plot.right()), float(plot.bottom()))
        substrate.lineTo(float(plot.left()), float(plot.bottom()))
        substrate.closeSubpath()
        painter.fillPath(substrate, QColor("#718493"))

        painter.setPen(QPen(QColor("#d7e0e6"), 1))
        center_x = plot.center().x()
        painter.drawLine(int(center_x), plot.top(), int(center_x), plot.bottom())
        painter.setPen(QPen(QColor("#263f4e"), 1))
        painter.drawRect(plot)
        for index, y_values in enumerate(mapped):
            boundary_path = QPainterPath()
            boundary_path.moveTo(float(px[0]), float(y_values[0]))
            for x_value, y_value in zip(px[1:], y_values[1:], strict=True):
                boundary_path.lineTo(float(x_value), float(y_value))
            painter.setPen(
                QPen(
                    QColor("#17384b" if index == 0 else "#ffffff"),
                    2 if index == 0 else 1,
                )
            )
            painter.drawPath(boundary_path)

        ambient_label = QRectF(plot.left() + 8, plot.top() + 6, 82, 20)
        painter.fillRect(ambient_label, QColor("#f8fafb"))
        painter.setPen(QPen(QColor("#a8b7c1"), 1))
        painter.drawRoundedRect(ambient_label, 4, 4)
        painter.setPen(QPen(QColor("#334c5b"), 1))
        painter.drawText(
            ambient_label,
            Qt.AlignmentFlag.AlignCenter,
            "ambient",
        )
        substrate_label = QRectF(plot.left() + 8, plot.bottom() - 26, 88, 20)
        painter.fillRect(substrate_label, QColor("#607786"))
        painter.setPen(QPen(QColor("#dce5ea"), 1))
        painter.drawRoundedRect(substrate_label, 4, 4)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawText(
            substrate_label,
            Qt.AlignmentFlag.AlignCenter,
            "substrate",
        )

    def _draw_series(
        self, painter: QPainter, plot, values: np.ndarray, color: QColor
    ) -> None:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim == 2:
            data = data[:, 0]
        data = data.ravel()
        if data.size == 0:
            return
        y_min = 0.0
        y_max = max(1.0, float(np.max(data)))
        points: list[tuple[float, float]] = []
        for index, value in enumerate(data):
            x_ratio = 0.0 if data.size == 1 else index / (data.size - 1)
            y_ratio = (float(value) - y_min) / (y_max - y_min)
            points.append(
                (
                    plot.left() + plot.width() * x_ratio,
                    plot.bottom() - plot.height() * y_ratio,
                )
            )
        painter.setPen(QPen(color, 2))
        for (x0, y0), (x1, y1) in pairwise(points):
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

    def _draw_heatmap(self, painter: QPainter, plot, values: np.ndarray) -> None:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.size == 0:
            return
        data = data[:: max(1, data.shape[0] // 48), :: max(1, data.shape[1] // 80)]
        low = float(np.nanmin(data))
        high = float(np.nanmax(data))
        span = high - low if high > low else 1.0
        cell_w = plot.width() / max(data.shape[1], 1)
        cell_h = plot.height() / max(data.shape[0], 1)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                ratio = (float(data[row, column]) - low) / span
                color = QColor(
                    35 + int(170 * ratio),
                    75 + int(110 * (1.0 - ratio)),
                    120 + int(90 * ratio),
                )
                painter.fillRect(
                    QRectF(
                        plot.left() + column * cell_w,
                        plot.top() + row * cell_h,
                        cell_w + 0.5,
                        cell_h + 0.5,
                    ),
                    color,
                )


class SolverService:
    """Lazy boundary around numerical-core interactions."""

    def validate(self, inputs: ShellInputs) -> ValidationResult:
        messages: list[str] = []
        if not inputs.project_name.strip():
            messages.append("Project name is required.")
        if inputs.source_mode == "Import" and inputs.imported_device is None:
            messages.append("Import source requires a loaded MATLAB or NumPy device.")
        if inputs.workflow == "optimization":
            if len(inputs.optimization_lower_bounds) != len(
                inputs.optimization_upper_bounds
            ):
                messages.append("Optimization lower and upper bounds must match.")
            if (
                inputs.source_mode == "Analytic"
                and len(inputs.optimization_lower_bounds) != 3
            ):
                messages.append("Analytic optimization requires 3 legacy bounds.")
            if (
                inputs.source_mode == "Import"
                and len(inputs.optimization_lower_bounds) < 4
            ):
                messages.append("Imported optimization requires 4 legacy bounds.")
            if any(
                lower > upper
                for lower, upper in zip(
                    inputs.optimization_lower_bounds,
                    inputs.optimization_upper_bounds,
                )
            ):
                messages.append("Optimization lower bounds must be <= upper bounds.")
            if inputs.source_mode == "Import" and (
                inputs.imported_device is None
                or inputs.imported_device.sub_L_um.size < 3
            ):
                messages.append(
                    "Imported optimization requires a loaded device with 3 layers."
                )
        if inputs.period_um <= 0:
            messages.append("Device period must be positive.")
        if inputs.height_um <= 0:
            messages.append("Interface height must be positive.")
        if inputs.layer_count <= 0:
            messages.append("At least one layer is required.")
        if inputs.harmonics < 1:
            messages.append("Harmonics must be at least one.")
        if len(inputs.params) != inputs.layer_count * 2:
            messages.append(
                "Legacy Params must contain thicknesses followed by indices."
            )
        if any(value <= 0 for value in inputs.params):
            messages.append(
                "Layer thicknesses and refractive indices must be positive."
            )
        if inputs.wavelength_start_nm > inputs.wavelength_stop_nm:
            messages.append("Wavelength start must be less than or equal to stop.")
        if inputs.angle_start_deg > inputs.angle_stop_deg:
            messages.append("Angle start must be less than or equal to stop.")
        if inputs.sweep_points < 1:
            messages.append("Sweep points must be at least one.")
        if inputs.n_superstrate <= 0 or inputs.n_substrate <= 0:
            messages.append("Ambient and substrate indices must be positive.")

        if messages:
            return ValidationResult(False, tuple(messages), "Validation failed")

        try:
            self.build_request(inputs)
        except (ImportError, TypeError, ValueError) as exc:
            return ValidationResult(False, (str(exc),), "Workflow validation failed")

        return ValidationResult(
            True,
            (f"Inputs are ready for {inputs.workflow} execution.",),
            f"{inputs.workflow} workflow ready",
        )

    def build_request(self, inputs: ShellInputs):
        if inputs.workflow == "optimization":
            from zenscat.optimization import OptimizationRequest

            source = "imported" if inputs.source_mode == "Import" else "analytic"
            return OptimizationRequest(
                source=source,
                template=self._build_rcwa_request(inputs),
                objective=inputs.optimization_objective,
                lower_bounds=np.asarray(inputs.optimization_lower_bounds),
                upper_bounds=np.asarray(inputs.optimization_upper_bounds),
                max_generations=inputs.optimization_generations,
                population_size=inputs.optimization_population,
                seed=0,
                polish=False,
            )
        if inputs.workflow == "fdfd":
            from zenscat.workflows import FDFDRequest

            return FDFDRequest(
                params=np.asarray(inputs.params, dtype=np.float64),
                layer_num=inputs.layer_count,
                wavelengths_um=np.linspace(
                    inputs.wavelength_start_nm,
                    inputs.wavelength_stop_nm,
                    inputs.sweep_points,
                )
                / 1000.0,
                angles_rad=np.linspace(
                    inputs.angle_start_deg,
                    inputs.angle_stop_deg,
                    inputs.sweep_points,
                )
                * np.pi
                / 180,
                polarization=inputs.polarization,
                distribution=inputs.distribution,
                interface=inputs.fdfd_interface,
                Lx_um=inputs.period_um,
                h_um=inputs.height_um,
                n_superstrate=inputs.n_superstrate,
                n_substrate=inputs.n_substrate,
                nres=inputs.fdfd_nres,
                spacer_um=np.asarray(inputs.fdfd_spacer_um, dtype=np.float64),
                npml=np.asarray(inputs.fdfd_npml, dtype=np.int64),
                is_periodic=True,
                period_num=inputs.fdfd_period_num,
                refractive_idx=inputs.fdfd_palette != "manual legacy",
                dispersion_mode="corrected_um"
                if inputs.fdfd_palette == "material corrected"
                else "legacy_fdfd",
            )
        if inputs.workflow == "phc":
            from zenscat.core import PhCInterfaceParams
            from zenscat.workflows import PhCRCWARequest

            interface = {
                "PhC Rectangle": "PhC_rec_square",
                "PhC Ellipse": "PhC_rec_circ",
                "PhC Hex": "PhC_hex_columns",
            }[inputs.phc_shape]
            return PhCRCWARequest(
                params=np.asarray(
                    [
                        inputs.phc_pitch_um,
                        inputs.phc_background_n,
                        inputs.phc_inclusion_n,
                    ],
                    dtype=np.float64,
                ),
                wavelengths_m=np.linspace(
                    inputs.wavelength_start_nm,
                    inputs.wavelength_stop_nm,
                    inputs.sweep_points,
                )
                * 1e-9,
                angles_rad=np.linspace(
                    inputs.angle_start_deg,
                    inputs.angle_stop_deg,
                    inputs.sweep_points,
                )
                * np.pi
                / 180,
                layer_count=inputs.phc_layer_count,
                interface=interface,
                interface_params=PhCInterfaceParams(
                    rec_2D_wx=inputs.phc_wx,
                    rec_2D_wy=inputs.phc_wy,
                    rec_rot_angle=inputs.phc_rotation_deg,
                    ax=inputs.phc_ax,
                    ay=inputs.phc_ay,
                    ellipse_rot_angle=inputs.phc_rotation_deg,
                    radius_star_ellipse=inputs.phc_radius,
                ),
                harmonic_count=inputs.harmonics,
                polarization=inputs.polarization,
                Nx=inputs.nx,
                Nz=inputs.nz,
                n_superstrate=inputs.n_superstrate,
                n_substrate=inputs.n_substrate,
                repeat_mode="corrected"
                if inputs.phc_repeat_mode.startswith("corrected")
                else "legacy",
            )
        return self._build_rcwa_request(inputs)

    def _build_rcwa_request(self, inputs: ShellInputs):
        wavelengths = (
            np.linspace(
                inputs.wavelength_start_nm,
                inputs.wavelength_stop_nm,
                inputs.sweep_points,
            )
            * 1e-9
        )
        angles = (
            np.linspace(
                inputs.angle_start_deg, inputs.angle_stop_deg, inputs.sweep_points
            )
            * np.pi
            / 180
        )
        if inputs.source_mode == "Import":
            from zenscat.workflows import ImportedRCWARequest

            return ImportedRCWARequest(
                device=inputs.imported_device,
                wavelengths_m=wavelengths,
                angles_rad=angles,
                harmonic_count=inputs.harmonics,
                matrix_method=inputs.matrix_method,
                polarization=inputs.polarization,
                n_superstrate=inputs.n_superstrate,
                n_substrate=inputs.n_substrate,
            )

        from zenscat.workflows import AnalyticRCWARequest

        return AnalyticRCWARequest(
            params=np.asarray(inputs.params, dtype=np.float64),
            layer_num=inputs.layer_count,
            wavelengths_m=wavelengths,
            angles_rad=angles,
            harmonic_count=inputs.harmonics,
            matrix_method=inputs.matrix_method,
            polarization=inputs.polarization,
            distribution=inputs.distribution,
            interface=inputs.interface,
            Lx_um=inputs.period_um,
            h_um=inputs.height_um,
            Nx=inputs.nx,
            Nz=inputs.nz,
            n_superstrate=inputs.n_superstrate,
            n_substrate=inputs.n_substrate,
        )

    def run(
        self,
        inputs: ShellInputs,
        cancel_event: threading.Event,
        progress: Callable[[int, int], None],
    ):
        validation = self.validate(inputs)
        if not validation.ok:
            raise ValueError("; ".join(validation.messages))
        if inputs.workflow == "optimization":
            from zenscat.optimization import run_optimization

            return run_optimization(
                self.build_request(inputs),
                cancel_event=cancel_event,
                progress=lambda snapshot: progress(
                    snapshot.generation, inputs.optimization_generations
                ),
                solver_progress=progress,
            )
        if inputs.workflow == "fdfd":
            from zenscat.workflows import run_fdfd

            return run_fdfd(
                self.build_request(inputs),
                cancel_token=cancel_event,
                progress=progress,
            )
        if inputs.workflow == "phc":
            from zenscat.workflows import run_phc_rcwa

            return run_phc_rcwa(
                self.build_request(inputs),
                cancel_token=cancel_event,
                progress=progress,
            )
        if inputs.source_mode == "Import":
            from zenscat.workflows import run_imported_rcwa

            return run_imported_rcwa(
                self.build_request(inputs),
                cancel_token=cancel_event,
                progress=progress,
            )

        from zenscat.workflows import run_analytic_rcwa

        return run_analytic_rcwa(
            self.build_request(inputs), cancel_token=cancel_event, progress=progress
        )

    def export(self, run, inputs: ShellInputs, destination: str | Path) -> Path:
        from zenscat.legacy_io import save_fdfd_result_bundle, save_legacy_result_bundle

        if hasattr(run, "fdfd_bundle"):
            return save_fdfd_result_bundle(
                destination,
                run.fdfd_bundle(params=np.asarray(inputs.params, dtype=np.float64)),
            )
        if hasattr(run, "legacy_bundle") and inputs.workflow == "optimization":
            return save_legacy_result_bundle(destination, run.legacy_bundle())

        params = None
        if inputs.workflow == "phc":
            params = np.asarray(
                [
                    inputs.phc_pitch_um,
                    inputs.phc_background_n,
                    inputs.phc_inclusion_n,
                ],
                dtype=np.float64,
            )
        elif inputs.source_mode != "Import":
            params = np.asarray(inputs.params, dtype=np.float64)

        return save_legacy_result_bundle(
            destination,
            run.legacy_bundle(params=params),
        )


class RunWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self, service: SolverService, inputs: ShellInputs, cancel_event: threading.Event
    ) -> None:
        super().__init__()
        self._service = service
        self._inputs = inputs
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:
            result = self._service.run(
                self._inputs, self._cancel_event, self.progress.emit
            )
        except Exception as exc:  # noqa: BLE001 - GUI worker must signal backend failures.
            if (
                self._cancel_event.is_set()
                or exc.__class__.__name__ == "SimulationCancelled"
            ):
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
            return
        if self._cancel_event.is_set():
            self.cancelled.emit()
        else:
            self.finished.emit(result)


class MainWindow(QMainWindow):
    """Persistent ZenScat shell window."""

    def __init__(self, service: SolverService | None = None) -> None:
        super().__init__()
        self.setObjectName("zenscatMainWindow")
        self.setWindowTitle("ZenScat Studio")
        self.resize(1280, 780)
        self.setStyleSheet(APP_STYLE)

        self._service = service or SolverService()
        self._thread: QThread | None = None
        self._worker: RunWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._last_run: object | None = None
        self._last_inputs: ShellInputs | None = None
        self._project_path: Path | None = None
        self._imported_device: object | None = None
        self._import_path: Path | None = None
        self._validated = False

        self._build_actions()
        self._build_ui()
        self._connect_signals()
        self._set_backend_status("Local Python", "Idle")
        self._append_log("Shell initialized.")
        self._update_project_status()
        self._update_grid_summary()
        self._refresh_context_inspector("Project")
        self.nav_list.setFocus(Qt.FocusReason.OtherFocusReason)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._refresh_combo_popup_widths()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_combo_popup_widths()

    def _build_actions(self) -> None:
        self.validate_action = QAction("Validate", self)
        self.validate_action.setObjectName("validateAction")
        self.run_action = QAction("Run", self)
        self.run_action.setObjectName("runAction")
        self.run_action.setEnabled(False)
        self.run_action.setToolTip("Run is available after validation.")
        self.cancel_action = QAction("Cancel", self)
        self.cancel_action.setObjectName("cancelAction")
        self.cancel_action.setEnabled(False)
        self.open_project_action = QAction("Open", self)
        self.open_project_action.setObjectName("openProjectAction")
        self.save_project_action = QAction("Save", self)
        self.save_project_action.setObjectName("saveProjectAction")
        self.load_import_action = QAction("Import Device", self)
        self.load_import_action.setObjectName("loadImportAction")
        self.export_action = QAction("Export", self)
        self.export_action.setObjectName("exportAction")
        self.export_action.setEnabled(False)
        self.export_action.setToolTip("Export is available after solver results exist.")

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("rootShell")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_top_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navigationList")
        self.nav_list.setFixedWidth(210)
        for label in NAV_ITEMS:
            item = QListWidgetItem(f"{label}\n{NAV_STATE[label]}")
            item.setData(Qt.ItemDataRole.UserRole, label)
            self.nav_list.addItem(item)
        self.nav_list.setCurrentRow(0)
        body.addWidget(self.nav_list)

        self.workspace = QStackedWidget()
        self.workspace.setObjectName("workspaceStack")
        self._pages: dict[str, QWidget] = {}
        for label in NAV_ITEMS:
            page = self._build_page(label)
            self._pages[label] = page
            self.workspace.addWidget(self._scroll_page(label, page))
        body.addWidget(self.workspace, 1)

        body.addWidget(self._build_inspector())

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topStatusBar")
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        self.project_status = QLabel()
        self.project_status.setObjectName("projectStatusLabel")
        self.backend_status = QLabel()
        self.backend_status.setObjectName("backendStatusLabel")
        self.job_status = QLabel()
        self.job_status.setObjectName("jobStatusLabel")

        layout.addWidget(self.project_status)
        layout.addWidget(self._separator())
        layout.addWidget(self.backend_status)
        layout.addWidget(self._separator())
        layout.addWidget(self.job_status, 1)

        self.validate_button = QPushButton("Validate")
        self.validate_button.setObjectName("validateButton")
        self.validate_button.setDefault(True)
        self.open_project_button = QPushButton("Open")
        self.open_project_button.setObjectName("openProjectButton")
        self.save_project_button = QPushButton("Save")
        self.save_project_button.setObjectName("saveProjectButton")
        self.load_import_button = QPushButton("Import")
        self.load_import_button.setObjectName("loadImportButton")
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("runButton")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("Run is available after validation.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Export")
        self.export_button.setObjectName("exportButton")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("Export is locked until computed results exist.")

        layout.addWidget(self.open_project_button)
        layout.addWidget(self.save_project_button)
        layout.addWidget(self.load_import_button)
        layout.addWidget(self.validate_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.export_button)
        return bar

    def _card(self, object_name: str) -> QFrame:
        card = QFrame()
        card.setObjectName(object_name)
        card.setProperty("class", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        return card

    def _metric_card(
        self, object_name: str, label: str, value: str, detail: str
    ) -> QFrame:
        card = self._card(object_name)
        title = QLabel(label)
        title.setProperty("class", "muted")
        title.setObjectName(f"{object_name}Label")
        value_label = QLabel(value)
        value_label.setObjectName(f"{object_name}Value")
        value_label.setProperty("class", "title")
        detail_label = QLabel(detail)
        detail_label.setObjectName(f"{object_name}Detail")
        detail_label.setProperty("class", "muted")
        setattr(self, f"{object_name}Value", value_label)
        setattr(self, f"{object_name}Detail", detail_label)
        card.layout().addWidget(title)
        card.layout().addWidget(value_label)
        card.layout().addWidget(detail_label)
        return card

    def _section_label(self, text: str, object_name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setProperty("class", "sectionTitle")
        return label

    def _badge(self, text: str, object_name: str) -> QLabel:
        badge = QLabel(text)
        badge.setObjectName(object_name)
        badge.setProperty("class", "badge")
        setattr(self, object_name, badge)
        if any(
            token in text.lower() for token in ("locked", "unavailable", "disabled")
        ):
            badge.setProperty("locked", True)
        return badge

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _inline_pair(self, first: QWidget, second: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(first)
        layout.addWidget(second)
        return container

    def _build_page(self, label: str) -> QWidget:
        builders: dict[str, Callable[[], QWidget]] = {
            "Project": self._build_project_page,
            "Device": self._build_device_page,
            "RCWA Sweep": self._build_rcwa_page,
            "Optimize": self._build_optimize_page,
            "FDFD Fields": self._build_fdfd_page,
            "Results": self._build_results_page,
            "Jobs": self._build_jobs_page,
        }
        return builders[label]()

    def _build_project_page(self) -> QWidget:
        page = self._page("projectPage")
        page.layout().addWidget(
            self._section_label("Scientific Project Dashboard", "projectDashboardTitle")
        )

        summary_grid = QGridLayout()
        summary_grid.setSpacing(12)
        summary_grid.addWidget(
            self._metric_card(
                "metricBackend", "Backend", "Local Python", "Validation boundary active"
            ),
            0,
            0,
        )
        summary_grid.addWidget(
            self._metric_card(
                "metricDevice", "Device", "4 layers", "320 nm total stack"
            ),
            0,
            1,
        )
        summary_grid.addWidget(
            self._metric_card(
                "metricSweep", "Sweep", "1550 nm", "Normal-incidence seed"
            ),
            0,
            2,
        )
        page.layout().addLayout(summary_grid)

        row = QHBoxLayout()
        row.setSpacing(14)

        setup_card = self._card("projectSetupCard")
        form = self._form_layout()

        self.backend_combo = QComboBox()
        self.backend_combo.setObjectName("backendCombo")
        self.backend_combo.addItems(("Local Python", "External solver"))
        self._configure_combo(self.backend_combo)
        self.project_name = QComboBox()
        self.project_name.setObjectName("projectNameCombo")
        self.project_name.setEditable(True)
        self.project_name.addItem("Untitled RCWA Study")
        self._configure_combo(self.project_name)
        self.source_combo = QComboBox()
        self.source_combo.setObjectName("sourceCombo")
        self.source_combo.addItems(("Analytic", "Import"))
        self._configure_combo(self.source_combo)
        self.method_combo = QComboBox()
        self.method_combo.setObjectName("methodCombo")
        self.method_combo.addItems(("S", "T"))
        self._configure_combo(self.method_combo)
        self.interface_combo = QComboBox()
        self.interface_combo.setObjectName("interfaceCombo")
        self.interface_combo.addItems(("sin", "DE1", "DE4", "tri"))
        self._configure_combo(self.interface_combo)
        self._configure_interface_combo(self.interface_combo)
        self.distribution_combo = QComboBox()
        self.distribution_combo.setObjectName("distributionCombo")
        self.distribution_combo.addItems(("all", "two"))
        self._configure_combo(self.distribution_combo)

        form.addRow("Project", self.project_name)
        form.addRow("Source", self.source_combo)
        form.addRow("Backend", self.backend_combo)
        form.addRow("Method", self.method_combo)
        form.addRow("Interface", self.interface_combo)
        form.addRow("Distribution", self.distribution_combo)
        setup_card.layout().addLayout(form)
        self.interface_guide_label = QLabel(
            "Interface shapes — sin: sinusoid · DE1: trapezium · "
            "DE4: soft trapezium / super-Gaussian · tri: triangle"
        )
        self.interface_guide_label.setObjectName("interfaceGuideLabel")
        self.interface_guide_label.setProperty("class", "muted")
        self.interface_guide_label.setWordWrap(True)
        setup_card.layout().addWidget(self.interface_guide_label)
        self.import_source_label = QLabel("No imported device loaded.")
        self.import_source_label.setObjectName("importSourceLabel")
        self.import_source_label.setProperty("class", "muted")
        self.import_source_label.setWordWrap(True)
        setup_card.layout().addWidget(self.import_source_label)
        self.project_science_summary = QLabel()
        self.project_science_summary.setObjectName("projectScienceSummaryLabel")
        self.project_science_summary.setWordWrap(True)
        setup_card.layout().addWidget(self.project_science_summary)
        setup_card.layout().addWidget(
            self._badge("Ready for validation", "projectProfileBadge")
        )
        next_step = QLabel(
            "Next step: validate the analytic stack, then run the selected "
            "S/T workflow."
        )
        next_step.setObjectName("projectNextStepLabel")
        next_step.setWordWrap(True)
        setup_card.layout().addWidget(next_step)
        row.addWidget(setup_card, 1)

        workflow = self._card("workflowOverviewCard")
        workflow.layout().addWidget(
            self._section_label("Legacy analytic Params", "workflowTitle")
        )
        self.project_stack_table = QTableWidget(2, 4)
        self.project_stack_table.setObjectName("projectStackSummaryTable")
        self.project_stack_table.setHorizontalHeaderLabels(
            ("Layer", "Thickness um", "Index n", "Pattern")
        )
        for row_index, values in enumerate(
            (
                ("1", "0.182", "1.781", "analytic interface"),
                ("2", "0.120", "1.650", "analytic interface"),
            )
        ):
            for column_index, value in enumerate(values):
                self.project_stack_table.setItem(
                    row_index, column_index, QTableWidgetItem(value)
                )
        self._fit_project_stack_table_header(self.project_stack_table)
        self.project_stack_table.verticalHeader().setVisible(False)
        self.project_stack_table.setMinimumHeight(126)
        workflow.layout().addWidget(self.project_stack_table)
        for name, state in (
            ("1 Project metadata", "ready"),
            ("2 Device stack", "configured"),
            ("3 RCWA sweep", "ready"),
            ("4 Solve/export", "disabled"),
        ):
            workflow.layout().addWidget(
                self._badge(f"{name}  {state}", f"workflow{name[0]}Badge")
            )
        row.addWidget(workflow, 1)
        page.layout().addLayout(row)

        self.project_device_plot = PlotPreview(
            "projectDevicePreview",
            "Live device profile",
            "lateral pitch",
            "stack depth",
            mode="stack",
        )
        page.layout().addWidget(self.project_device_plot, 1)

        self.dashboard_plot = PlotPreview(
            "dashboardPlotPreview",
            "Project result canvas",
            "wavelength / nm",
            "R, T",
        )
        page.layout().addWidget(self.dashboard_plot, 1)
        page.layout().addStretch(1)
        return page

    def _build_device_page(self) -> QWidget:
        page = self._page("devicePage")
        page.layout().addWidget(
            self._section_label("Device Stack Editor", "deviceEditorTitle")
        )

        row = QHBoxLayout()
        row.setSpacing(14)

        control_card = self._card("deviceParameterCard")
        form = self._form_layout()

        self.period_um = self._double("periodUmInput", 0.001, 1000.0, 0.32, " um")
        self.height_um = self._double("heightUmInput", 0.001, 1000.0, 0.154, " um")
        self.layer_count = self._integer("layerCountInput", 1, 128, 2)
        self.n_superstrate = self._double("nSuperstrateInput", 0.01, 20.0, 1.0, "")
        self.n_substrate = self._double("nSubstrateInput", 0.01, 20.0, 1.516, "")

        form.addRow("Period", self.period_um)
        form.addRow("Interface height", self.height_um)
        form.addRow("Layers", self.layer_count)
        form.addRow("Ambient n", self.n_superstrate)
        form.addRow("Substrate n", self.n_substrate)
        control_card.layout().addLayout(form)
        control_card.layout().addWidget(
            self._badge("Grid synchronized", "deviceGridBadge")
        )
        row.addWidget(control_card, 1)

        table_card = self._card("layerTableCard")
        table_card.layout().addWidget(
            self._section_label("Layer stack", "layerTableTitle")
        )
        self.layer_table = QTableWidget(2, 4)
        self.layer_table.setObjectName("layerTable")
        self.layer_table.setHorizontalHeaderLabels(
            ("Layer", "Thickness um", "Index n", "Pattern")
        )
        for row_index, values in enumerate(
            (
                ("1", "0.182", "1.781", "analytic interface"),
                ("2", "0.120", "1.650", "analytic interface"),
            )
        ):
            for column_index, value in enumerate(values):
                self.layer_table.setItem(
                    row_index, column_index, QTableWidgetItem(value)
                )
        self.layer_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setMinimumHeight(170)
        table_card.layout().addWidget(self.layer_table)
        layer_buttons = QHBoxLayout()
        self.add_layer_button = QPushButton("Add")
        self.add_layer_button.setObjectName("addLayerButton")
        self.remove_layer_button = QPushButton("Remove")
        self.remove_layer_button.setObjectName("removeLayerButton")
        self.move_layer_up_button = QPushButton("Up")
        self.move_layer_up_button.setObjectName("moveLayerUpButton")
        self.move_layer_down_button = QPushButton("Down")
        self.move_layer_down_button.setObjectName("moveLayerDownButton")
        for button in (
            self.add_layer_button,
            self.remove_layer_button,
            self.move_layer_up_button,
            self.move_layer_down_button,
        ):
            layer_buttons.addWidget(button)
        table_card.layout().addLayout(layer_buttons)
        row.addWidget(table_card, 2)
        page.layout().addLayout(row)

        self.structure_plot = PlotPreview(
            "structurePlotPreview",
            "Layer geometry preview",
            "lateral pitch",
            "stack depth",
            mode="stack",
        )
        page.layout().addWidget(self.structure_plot, 1)
        return page

    def _build_rcwa_page(self) -> QWidget:
        page = self._page("rcwaSweepPage")
        page.layout().addWidget(
            self._section_label("RCWA Sweep Setup", "rcwaSweepTitle")
        )

        row = QHBoxLayout()
        row.setSpacing(14)
        control_card = self._card("rcwaControlCard")
        form = self._form_layout()

        self.wavelength_nm = self._double(
            "wavelengthNmInput", 1.0, 100000.0, 510.0, " nm"
        )
        self.theta_deg = self._double("thetaDegInput", -89.9, 89.9, 0.0, " deg")
        self.wavelength_start_nm = self._double(
            "wavelengthStartNmInput", 1.0, 100000.0, 500.0, " nm"
        )
        self.wavelength_stop_nm = self._double(
            "wavelengthStopNmInput", 1.0, 100000.0, 520.0, " nm"
        )
        self.angle_start_deg = self._double(
            "angleStartDegInput", -89.9, 89.9, 0.0, " deg"
        )
        self.angle_stop_deg = self._double(
            "angleStopDegInput", -89.9, 89.9, 5.0, " deg"
        )
        self.sweep_points = self._integer("sweepPointsInput", 1, 10001, 3)
        self.harmonics = self._integer("harmonicsInput", 1, 64, 2)
        self.nx_input = self._integer("nxInput", 16, 4096, 128)
        self.nz_input = self._integer("nzInput", 1, 512, 5)
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("modeCombo")
        self.mode_combo.addItems(("E", "H"))
        self._configure_combo(self.mode_combo)

        form.addRow("Polarization", self.mode_combo)
        form.addRow("Center wavelength", self.wavelength_nm)
        form.addRow("Center angle", self.theta_deg)
        form.addRow("Wavelength start", self.wavelength_start_nm)
        form.addRow("Wavelength stop", self.wavelength_stop_nm)
        form.addRow("Angle start", self.angle_start_deg)
        form.addRow("Angle stop", self.angle_stop_deg)
        form.addRow("Points", self.sweep_points)
        form.addRow("Harmonics", self.harmonics)
        form.addRow("Nx", self.nx_input)
        form.addRow("Nz", self.nz_input)
        control_card.layout().addLayout(form)
        row.addWidget(control_card, 1)

        summary_card = self._card("gridSummaryCard")
        summary_card.layout().addWidget(
            self._section_label("Mode and grid", "gridSummaryTitle")
        )
        self.grid_summary = QLabel()
        self.grid_summary.setObjectName("gridSummaryLabel")
        self.grid_summary.setWordWrap(True)
        summary_card.layout().addWidget(self.grid_summary)
        summary_card.layout().addWidget(
            self._badge("Nonblocking validation job", "rcwaJobBadge")
        )
        summary_card.layout().addWidget(
            self._badge("Validate, then Run selected workflow", "rcwaSolveBadge")
        )
        phc_form = self._form_layout()
        self.phc_shape_combo = QComboBox()
        self.phc_shape_combo.setObjectName("phcShapeCombo")
        self.phc_shape_combo.addItems(
            ("1D Analytic/Import", "PhC Rectangle", "PhC Ellipse", "PhC Hex")
        )
        self._configure_combo(self.phc_shape_combo)
        self.phc_repeat_combo = QComboBox()
        self.phc_repeat_combo.setObjectName("phcRepeatCombo")
        self.phc_repeat_combo.addItems(("legacy repeat", "corrected repeat"))
        self._configure_combo(self.phc_repeat_combo)
        self.phc_layer_count = self._integer("phcLayerCountInput", 1, 8, 1)
        self.phc_pitch_um = self._double("phcPitchUmInput", 0.01, 10.0, 0.32, " um")
        self.phc_background_n = self._double("phcBackgroundInput", 0.01, 10.0, 1.5, "")
        self.phc_inclusion_n = self._double("phcInclusionInput", 0.01, 10.0, 2.0, "")
        self.phc_wx = self._double("phcWxInput", 0.001, 10.0, 0.2, " um")
        self.phc_wy = self._double("phcWyInput", 0.001, 10.0, 0.25, " um")
        self.phc_rotation = self._double(
            "phcRotationInput", -180.0, 180.0, 15.0, " deg"
        )
        self.phc_ax = self._double("phcAxInput", 0.001, 10.0, 1.0, "")
        self.phc_ay = self._double("phcAyInput", 0.001, 10.0, 0.5, "")
        self.phc_radius = self._double("phcRadiusInput", 0.001, 10.0, 0.45, "")
        phc_form.addRow("PhC mode", self.phc_shape_combo)
        phc_form.addRow("Repeat", self.phc_repeat_combo)
        phc_form.addRow("Layers", self.phc_layer_count)
        phc_form.addRow("Pitch", self.phc_pitch_um)
        phc_form.addRow("n background", self.phc_background_n)
        phc_form.addRow("n inclusion", self.phc_inclusion_n)
        phc_form.addRow("wx / wy", self._inline_pair(self.phc_wx, self.phc_wy))
        phc_form.addRow("rot / ax", self._inline_pair(self.phc_rotation, self.phc_ax))
        phc_form.addRow("ay / radius", self._inline_pair(self.phc_ay, self.phc_radius))
        summary_card.layout().addLayout(phc_form)
        row.addWidget(summary_card, 1)
        page.layout().addLayout(row)

        self.sweep_plot = PlotPreview(
            "sweepPlotPreview",
            "Sweep result canvas",
            "wavelength / nm",
            "diffraction efficiency",
        )
        page.layout().addWidget(self.sweep_plot, 1)
        return page

    def _build_optimize_page(self) -> QWidget:
        page = self._page("optimizePage")
        page.layout().addWidget(self._section_label("Optimize", "optimizePageTitle"))
        row = QHBoxLayout()
        row.setSpacing(14)

        controls = self._card("optimizePanel")
        form = self._form_layout()
        self.optimization_objective = QComboBox()
        self.optimization_objective.setObjectName("optimizationObjectiveCombo")
        self.optimization_objective.addItems(
            (
                "R(-1)",
                "R(0)",
                "R(+1)",
                "T(-1)",
                "T(0)",
                "T(+1)",
                "Absorption",
                "Gain",
            )
        )
        self.optimization_objective.setCurrentText("R(+1)")
        self._configure_combo(self.optimization_objective)
        self.optimization_generations = self._integer(
            "optimizationGenerationsInput", 1, 1000, 100
        )
        self.optimization_population = self._integer(
            "optimizationPopulationInput", 1, 200, 15
        )
        form.addRow("Objective", self.optimization_objective)
        form.addRow("Generations", self.optimization_generations)
        form.addRow("Population", self.optimization_population)
        controls.layout().addLayout(form)
        self.optimization_bounds_table = QTableWidget(3, 3)
        self.optimization_bounds_table.setObjectName("optimizationBoundsTable")
        self.optimization_bounds_table.setHorizontalHeaderLabels(
            ("Parameter", "Lower", "Upper")
        )
        for row_index, values in enumerate(
            (
                ("period_um", "0.6", "5"),
                ("height_um", "0.01", "0.5"),
                ("first_thickness_um", "0.2", "1"),
            )
        ):
            for column_index, value in enumerate(values):
                self.optimization_bounds_table.setItem(
                    row_index, column_index, QTableWidgetItem(value)
                )
        self.optimization_bounds_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        controls.layout().addWidget(self.optimization_bounds_table)
        controls.layout().addWidget(
            self._badge("Executes bounded legacy objective", "optimizeReadyBadge")
        )
        row.addWidget(controls, 1)

        preview = self._card("optimizePreviewPanel")
        self.optimization_status = QLabel("No optimization run yet.")
        self.optimization_status.setObjectName("optimizationStatusLabel")
        self.optimization_status.setWordWrap(True)
        preview.layout().addWidget(self.optimization_status)
        self.optimize_plot = PlotPreview(
            "optimizePlotPreview", "Merit history", "generation", "fitness"
        )
        preview.layout().addWidget(self.optimize_plot)
        row.addWidget(preview, 1)
        page.layout().addLayout(row)
        return page

    def _build_fdfd_page(self) -> QWidget:
        page = self._page("fdfdFieldsPage")
        page.layout().addWidget(self._section_label("FDFD Fields", "fdfdPageTitle"))
        row = QHBoxLayout()
        row.setSpacing(14)

        controls = self._card("fdfdPanel")
        form = self._form_layout()
        self.fdfd_interface_combo = QComboBox()
        self.fdfd_interface_combo.setObjectName("fdfdInterfaceCombo")
        self.fdfd_interface_combo.addItems(("sin", "DE1", "DE4", "tri"))
        self._configure_combo(self.fdfd_interface_combo)
        self._configure_interface_combo(self.fdfd_interface_combo)
        self.fdfd_palette_combo = QComboBox()
        self.fdfd_palette_combo.setObjectName("fdfdPaletteCombo")
        self.fdfd_palette_combo.addItems(
            ("manual legacy", "material legacy", "material corrected")
        )
        self._configure_combo(self.fdfd_palette_combo)
        self.fdfd_field_combo = QComboBox()
        self.fdfd_field_combo.setObjectName("fdfdFieldCombo")
        self.fdfd_field_combo.addItems(("abs(f)", "real(f)", "ER2"))
        self._configure_combo(self.fdfd_field_combo)
        self.fdfd_nres = self._double("fdfdNresInput", 1.0, 200.0, 20.0, "")
        self.fdfd_period_num = self._integer("fdfdPeriodNumInput", 1, 101, 11)
        self.fdfd_npml_x = self._integer("fdfdNpmlXInput", 0, 200, 20)
        self.fdfd_npml_y = self._integer("fdfdNpmlYInput", 0, 200, 20)
        self.fdfd_spacer_top = self._double(
            "fdfdSpacerTopInput", 0.0, 100.0, 2.0, " um"
        )
        self.fdfd_spacer_bottom = self._double(
            "fdfdSpacerBottomInput", 0.0, 100.0, 2.0, " um"
        )
        form.addRow("Interface", self.fdfd_interface_combo)
        form.addRow("Palette", self.fdfd_palette_combo)
        form.addRow("Field view", self.fdfd_field_combo)
        form.addRow("NRES", self.fdfd_nres)
        form.addRow("Period num", self.fdfd_period_num)
        form.addRow("NPML x/y", self._inline_pair(self.fdfd_npml_x, self.fdfd_npml_y))
        form.addRow(
            "Spacer top/bottom",
            self._inline_pair(self.fdfd_spacer_top, self.fdfd_spacer_bottom),
        )
        controls.layout().addLayout(form)
        self.fdfd_setup_summary = QLabel()
        self.fdfd_setup_summary.setObjectName("fdfdSetupSummaryLabel")
        self.fdfd_setup_summary.setWordWrap(True)
        controls.layout().addWidget(self.fdfd_setup_summary)
        controls.layout().addWidget(
            self._badge("Executes actual FDFD field solve", "fdfdReadyBadge")
        )
        controls.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(controls, 1)

        preview = self._card("fdfdPreviewPanel")
        self.fdfd_status = QLabel("No FDFD run yet.")
        self.fdfd_status.setObjectName("fdfdStatusLabel")
        self.fdfd_status.setWordWrap(True)
        preview.layout().addWidget(self.fdfd_status)
        self.field_plot = PlotPreview(
            "fieldPlotPreview", "Field result canvas", "x", "z"
        )
        preview.layout().addWidget(self.field_plot)
        preview.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(preview, 2)
        page.layout().addLayout(row, 1)
        return page

    def _build_results_page(self) -> QWidget:
        page = self._page("resultsPage")
        page.layout().addWidget(self._section_label("Results", "resultsPageTitle"))
        results_card = self._card("resultsCard")
        results_card.layout().addWidget(self._badge("Empty", "resultsEmptyBadge"))
        selector_row = QHBoxLayout()
        self.result_family_combo = QComboBox()
        self.result_family_combo.setObjectName("resultFamilyCombo")
        self.result_family_combo.addItems(("Transmission", "Reflection", "Energy"))
        self._configure_combo(self.result_family_combo)
        self.result_order_combo = QComboBox()
        self.result_order_combo.setObjectName("resultOrderCombo")
        self.result_order_combo.addItems(("-1", "0", "+1", "sum"))
        self._configure_combo(self.result_order_combo)
        selector_row.addWidget(self.result_family_combo)
        selector_row.addWidget(self.result_order_combo)
        results_card.layout().addLayout(selector_row)
        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText(
            "No computed results. Validate a configured workflow, then Run to "
            "populate transmission, reflection, energy, or FDFD field outputs. "
            "Export unlocks after a completed run."
        )
        results_card.layout().addWidget(self.results_text)
        self.results_plot = PlotPreview(
            "resultsPlotPreview", "Results canvas", "wavelength", "TRN0, REF0"
        )
        results_card.layout().addWidget(self.results_plot)
        page.layout().addWidget(results_card, 1)
        return page

    def _build_jobs_page(self) -> QWidget:
        page = self._page("jobsPage")
        page.layout().addWidget(self._section_label("Jobs", "jobsPageTitle"))
        jobs_card = self._card("jobsCard")
        jobs_card.layout().addWidget(self._badge("Idle queue", "jobsIdleBadge"))
        self.jobs_text = QTextEdit()
        self.jobs_text.setObjectName("jobsText")
        self.jobs_text.setReadOnly(True)
        self.jobs_text.setPlainText("No active jobs.")
        jobs_card.layout().addWidget(self.jobs_text)
        page.layout().addWidget(jobs_card, 1)
        return page

    def _page(self, object_name: str) -> QWidget:
        page = QWidget()
        page.setObjectName(object_name)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)
        return page

    def _fit_project_stack_table_header(self, table: QTableWidget) -> None:
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = table.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(True)
        padding = 26
        for column in range(3):
            item = table.horizontalHeaderItem(column)
            label = "" if item is None else item.text()
            header.resizeSection(
                column, table.fontMetrics().horizontalAdvance(label) + padding
            )

    def _scroll_page(self, label: str, page: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        object_token = "".join(part.lower().capitalize() for part in label.split())
        scroll_area.setObjectName(
            f"{object_token[0].lower()}{object_token[1:]}ScrollArea"
        )
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(page)
        return scroll_area

    def _form_layout(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _configure_combo(self, combo: QComboBox) -> None:
        popup_view = QListView(combo)
        popup_view.setObjectName(f"{combo.objectName()}PopupView")
        popup_view.setMouseTracking(True)
        popup_view.viewport().setMouseTracking(True)
        popup_view.setUniformItemSizes(True)
        popup_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        popup_view.setStyleSheet(COMBO_POPUP_STYLE)
        popup_palette = popup_view.palette()
        popup_palette.setColor(QPalette.ColorRole.Base, QColor("#fbfcfd"))
        popup_palette.setColor(QPalette.ColorRole.Text, QColor("#17212b"))
        popup_palette.setColor(QPalette.ColorRole.Highlight, QColor("#b9dce9"))
        popup_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#17212b"))
        popup_view.setPalette(popup_palette)
        popup_view.entered.connect(popup_view.setCurrentIndex)
        combo.setView(popup_view)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.setMinimumContentsLength(
            max(
                (len(combo.itemText(index)) for index in range(combo.count())),
                default=8,
            )
        )
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        width = self._combo_text_width(combo)
        popup_width = max(
            self._combo_popup_text_width(combo), combo.view().sizeHintForColumn(0)
        )
        combo.setMinimumWidth(width)
        combo.view().setMinimumWidth(popup_width)
        if combo.isEditable():
            combo.setToolTip(combo.currentText())
            combo.editTextChanged.connect(combo.setToolTip)

    def _configure_interface_combo(self, combo: QComboBox) -> None:
        def update_tooltip(interface: str) -> None:
            description = PROFILE_INTERFACE_DESCRIPTIONS.get(interface, interface)
            tooltip = f"{interface} — {description}"
            combo.setToolTip(tooltip)
            combo.setAccessibleDescription(tooltip)

        for index in range(combo.count()):
            interface = combo.itemText(index)
            description = PROFILE_INTERFACE_DESCRIPTIONS.get(interface, interface)
            combo.setItemData(
                index,
                f"{interface} — {description}",
                Qt.ItemDataRole.ToolTipRole,
            )
        update_tooltip(combo.currentText())
        combo.currentTextChanged.connect(update_tooltip)

    def _refresh_combo_popup_widths(self) -> None:
        for combo in self.findChildren(QComboBox):
            width = self._combo_text_width(combo)
            popup_width = max(
                self._combo_popup_text_width(combo), combo.view().sizeHintForColumn(0)
            )
            combo.setMinimumWidth(width)
            combo.view().setMinimumWidth(popup_width)

    def _combo_text_width(self, combo: QComboBox) -> int:
        longest = max(
            (combo.itemText(index) for index in range(combo.count())),
            key=len,
            default=combo.currentText(),
        )
        return max(132, combo.fontMetrics().horizontalAdvance(longest) + 56)

    def _combo_popup_text_width(self, combo: QComboBox) -> int:
        return max(
            self._combo_text_width(combo),
            combo.fontMetrics().horizontalAdvance(combo.currentText()) + 56,
        )

    def _build_inspector(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setObjectName("inspectorTabs")
        tabs.setFixedWidth(340)

        self.inspector_text = QTextEdit()
        self.inspector_text.setObjectName("contextInspectorText")
        self.inspector_text.setReadOnly(True)

        self.problems_text = QTextEdit()
        self.problems_text.setObjectName("problemsText")
        self.problems_text.setReadOnly(True)
        self.problems_text.setPlainText(
            "Validation hints:\n"
            "- Project name is required.\n"
            "- Period and wavelength must be positive.\n"
            "- Layer count must be at least one.\n"
            "- Run unlocks after validation; Export unlocks after computed results."
        )

        self.job_log = QTextEdit()
        self.job_log.setObjectName("jobLogText")
        self.job_log.setReadOnly(True)

        tabs.addTab(self.inspector_text, "Inspector")
        tabs.addTab(self.problems_text, "Problems")
        tabs.addTab(self.job_log, "Job Log")
        return tabs

    def _double(
        self, name: str, minimum: float, maximum: float, value: float, suffix: str
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setObjectName(name)
        widget.setRange(minimum, maximum)
        widget.setDecimals(4)
        widget.setValue(value)
        widget.setSingleStep(max((maximum - minimum) / 1000.0, 0.1))
        widget.setSuffix(suffix)
        widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        return widget

    def _integer(self, name: str, minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setObjectName(name)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        return widget

    def _connect_signals(self) -> None:
        self.nav_list.currentRowChanged.connect(self.workspace.setCurrentIndex)
        self.nav_list.currentItemChanged.connect(self._on_navigation_changed)
        self.validate_button.clicked.connect(self.validate_action.trigger)
        self.open_project_button.clicked.connect(self.open_project_action.trigger)
        self.save_project_button.clicked.connect(self.save_project_action.trigger)
        self.load_import_button.clicked.connect(self.load_import_action.trigger)
        self.run_button.clicked.connect(self.run_action.trigger)
        self.cancel_button.clicked.connect(self.cancel_action.trigger)
        self.export_button.clicked.connect(self.export_action.trigger)
        self.validate_action.triggered.connect(self.validate_inputs)
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action.triggered.connect(self.save_project)
        self.load_import_action.triggered.connect(self.load_import_device)
        self.run_action.triggered.connect(self.run_solver)
        self.cancel_action.triggered.connect(self.cancel_run)
        self.export_action.triggered.connect(self.export_results)
        self.backend_combo.currentTextChanged.connect(
            lambda value: self._set_backend_status(value, "Idle")
        )
        self.backend_combo.currentTextChanged.connect(
            lambda _value: self._update_project_status()
        )
        self.project_name.currentTextChanged.connect(
            lambda _value: self._update_project_status()
        )
        self.project_name.currentTextChanged.connect(
            lambda _value: self._refresh_context_inspector()
        )
        self.source_combo.currentTextChanged.connect(
            lambda _value: self._sync_optimization_bounds_for_source()
        )
        self.source_combo.currentTextChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.distribution_combo.currentTextChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.wavelength_start_nm.valueChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.wavelength_stop_nm.valueChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.angle_start_deg.valueChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.angle_stop_deg.valueChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.sweep_points.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.period_um.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.height_um.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.harmonics.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.layer_count.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.layer_count.valueChanged.connect(
            lambda _value: self._refresh_context_inspector()
        )
        self.n_superstrate.valueChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.n_substrate.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.mode_combo.currentTextChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.method_combo.currentTextChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.interface_combo.currentTextChanged.connect(
            lambda _value: self._on_inputs_changed()
        )
        self.nx_input.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.nz_input.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.layer_table.itemChanged.connect(lambda _item: self._on_inputs_changed())
        self.add_layer_button.clicked.connect(self.add_layer)
        self.remove_layer_button.clicked.connect(self.remove_layer)
        self.move_layer_up_button.clicked.connect(lambda: self.move_layer(-1))
        self.move_layer_down_button.clicked.connect(lambda: self.move_layer(1))
        for combo in (
            self.optimization_objective,
            self.fdfd_interface_combo,
            self.fdfd_palette_combo,
            self.phc_shape_combo,
            self.phc_repeat_combo,
        ):
            combo.currentTextChanged.connect(lambda _value: self._on_inputs_changed())
        for spinbox in (
            self.optimization_generations,
            self.optimization_population,
            self.fdfd_nres,
            self.fdfd_period_num,
            self.fdfd_npml_x,
            self.fdfd_npml_y,
            self.fdfd_spacer_top,
            self.fdfd_spacer_bottom,
            self.phc_layer_count,
            self.phc_pitch_um,
            self.phc_background_n,
            self.phc_inclusion_n,
            self.phc_wx,
            self.phc_wy,
            self.phc_rotation,
            self.phc_ax,
            self.phc_ay,
            self.phc_radius,
        ):
            spinbox.valueChanged.connect(lambda _value: self._on_inputs_changed())
        self.optimization_bounds_table.itemChanged.connect(
            lambda _item: self._on_inputs_changed()
        )
        self.result_family_combo.currentTextChanged.connect(
            lambda _value: self._update_result_plot()
        )
        self.result_order_combo.currentTextChanged.connect(
            lambda _value: self._update_result_plot()
        )
        self.fdfd_field_combo.currentTextChanged.connect(
            lambda _value: self._update_field_plot()
        )

    def _on_navigation_changed(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        self._refresh_context_inspector(str(name))

    def collect_inputs(self) -> ShellInputs:
        params = self._collect_params()
        return ShellInputs(
            project_name=self.project_name.currentText().strip(),
            backend=self.backend_combo.currentText(),
            workflow=self._active_workflow(),
            source_mode=self.source_combo.currentText(),
            import_path=None if self._import_path is None else str(self._import_path),
            imported_device=self._imported_device,
            matrix_method=self.method_combo.currentText(),
            interface=self.interface_combo.currentText(),
            polarization=self.mode_combo.currentText(),
            distribution=self.distribution_combo.currentText(),
            period_um=self.period_um.value(),
            height_um=self.height_um.value(),
            harmonics=self.harmonics.value(),
            layer_count=self.layer_table.rowCount(),
            n_superstrate=self.n_superstrate.value(),
            n_substrate=self.n_substrate.value(),
            wavelength_start_nm=self.wavelength_start_nm.value(),
            wavelength_stop_nm=self.wavelength_stop_nm.value(),
            angle_start_deg=self.angle_start_deg.value(),
            angle_stop_deg=self.angle_stop_deg.value(),
            sweep_points=self.sweep_points.value(),
            nx=self.nx_input.value(),
            nz=self.nz_input.value(),
            params=params,
            result_family=self.result_family_combo.currentText(),
            result_order=self.result_order_combo.currentText(),
            optimization_objective=self.optimization_objective.currentText(),
            optimization_lower_bounds=self._optimization_bounds(1),
            optimization_upper_bounds=self._optimization_bounds(2),
            optimization_generations=self.optimization_generations.value(),
            optimization_population=self.optimization_population.value(),
            fdfd_interface=self.fdfd_interface_combo.currentText(),
            fdfd_palette=self.fdfd_palette_combo.currentText(),
            fdfd_nres=self.fdfd_nres.value(),
            fdfd_period_num=self.fdfd_period_num.value(),
            fdfd_npml=(self.fdfd_npml_x.value(), self.fdfd_npml_y.value()),
            fdfd_spacer_um=(
                self.fdfd_spacer_top.value(),
                self.fdfd_spacer_bottom.value(),
            ),
            phc_shape=self.phc_shape_combo.currentText(),
            phc_repeat_mode=self.phc_repeat_combo.currentText(),
            phc_layer_count=self.phc_layer_count.value(),
            phc_pitch_um=self.phc_pitch_um.value(),
            phc_background_n=self.phc_background_n.value(),
            phc_inclusion_n=self.phc_inclusion_n.value(),
            phc_wx=self.phc_wx.value(),
            phc_wy=self.phc_wy.value(),
            phc_rotation_deg=self.phc_rotation.value(),
            phc_ax=self.phc_ax.value(),
            phc_ay=self.phc_ay.value(),
            phc_radius=self.phc_radius.value(),
        )

    def _active_workflow(self) -> str:
        current = (
            self.nav_list.currentItem().data(Qt.ItemDataRole.UserRole)
            if self.nav_list.currentItem() is not None
            else "Project"
        )
        if current == "Optimize":
            return "optimization"
        if current == "FDFD Fields":
            return "fdfd"
        if (
            current == "RCWA Sweep"
            and hasattr(self, "phc_shape_combo")
            and self.phc_shape_combo.currentText() != "1D Analytic/Import"
        ):
            return "phc"
        return "rcwa"

    def _optimization_bounds(self, column: int) -> tuple[float, ...]:
        values: list[float] = []
        for row in range(self.optimization_bounds_table.rowCount()):
            item = self.optimization_bounds_table.item(row, column)
            if item is None:
                raise ValueError("optimization bounds table contains empty cells")
            values.append(float(item.text()))
        return tuple(values)

    def validate_inputs(self) -> None:
        inputs = self.collect_inputs()
        result = self._service.validate(inputs)
        self._last_inputs = inputs
        self._validated = result.ok
        self.problems_text.setPlainText(
            "\n".join(result.messages) if not result.ok else "No problems."
        )
        self._append_log(
            f"Validate {'passed' if result.ok else 'failed'}: {result.summary}"
        )
        self._set_backend_status(inputs.backend, result.summary)
        self.run_button.setEnabled(result.ok)
        self.run_action.setEnabled(result.ok)
        self.export_button.setEnabled(False)
        self.export_action.setEnabled(False)
        if result.ok:
            source = inputs.source_mode.lower()
            run_label = (
                inputs.workflow if inputs.workflow != "rcwa" else f"{source} RCWA"
            )
            self.results_text.setPlainText(
                f"No computed results. Inputs validated; run {run_label} "
                "to compute outputs."
            )
        self._update_live_metrics()
        self._refresh_context_inspector()

    def run_solver(self) -> None:
        if self._thread is not None:
            self._append_log("Run ignored: another workflow job is active.")
            return
        inputs = self.collect_inputs()
        validation = self._service.validate(inputs)
        if not validation.ok:
            self.problems_text.setPlainText("\n".join(validation.messages))
            self._append_log(f"Run blocked: {validation.summary}")
            self.run_button.setEnabled(False)
            self.run_action.setEnabled(False)
            self._update_live_metrics()
            return

        self._last_inputs = inputs
        self._last_run = None
        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        self._worker = RunWorker(self._service, inputs, self._cancel_event)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_run_progress)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.cancelled.connect(self._on_run_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._set_running(True)
        self._append_log(f"Run started: {inputs.workflow} workflow.")
        self._set_backend_status(inputs.backend, f"Running {inputs.workflow}")
        self._update_live_metrics()
        self._thread.start()

    def cancel_run(self) -> None:
        if self._cancel_event is None or self._thread is None:
            return
        self._cancel_event.set()
        self.cancel_button.setEnabled(False)
        self.cancel_action.setEnabled(False)
        self._append_log("Cancel requested.")
        self._set_backend_status(self.backend_combo.currentText(), "Cancelling")
        self._update_live_metrics()

    def open_project(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open ZenScat Project",
            gettempdir(),
            "ZenScat Project (*.zenscat);;JSON (*.json);;All Files (*)",
        )
        if path:
            self.open_project_from(path)

    def open_project_from(self, path: str | Path) -> Path:
        from zenscat.project import ProjectDocument

        source = Path(path).expanduser().resolve()
        document = ProjectDocument.load(source)
        self._project_path = source
        self._apply_configuration(document.configuration or {})
        self._select_project_workflow(document.workflow)
        self.project_name.setCurrentText(document.name)
        self._refresh_combo_popup_widths()
        self._validated = False
        self._last_run = None
        self._last_inputs = None
        self._append_log(f"Opened project: {source}")
        self._set_backend_status(self.backend_combo.currentText(), "Project opened")
        self._update_project_status()
        self._on_inputs_changed()
        return source

    def save_project(self) -> None:
        initial = (
            str(self._project_path)
            if self._project_path is not None
            else str(Path(gettempdir()) / "zenscat-project.zenscat")
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save ZenScat Project",
            initial,
            "ZenScat Project (*.zenscat);;JSON (*.json);;All Files (*)",
        )
        if path:
            self.save_project_to(path)

    def save_project_to(self, path: str | Path) -> Path:
        from zenscat.project import ProjectDocument

        inputs = self.collect_inputs()
        document = ProjectDocument(
            name=inputs.project_name or "Untitled RCWA Study",
            workflow=self._project_workflow_from_inputs(inputs),
            configuration=self._configuration_from_ui(),
        )
        saved = document.save(path)
        self._project_path = saved
        self._append_log(f"Saved project: {saved}")
        self._set_backend_status(inputs.backend, "Project saved")
        self._update_live_metrics()
        return saved

    def load_import_device(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load MATLAB/NumPy Device",
            gettempdir(),
            "Legacy Device (*.mat *.npy *.py.npy);;All Files (*)",
        )
        if path:
            self.load_import_device_from(path)

    def load_import_device_from(self, path: str | Path) -> Path:
        from zenscat.legacy_io import load_imported_device

        source = Path(path).expanduser().resolve()
        trusted_pickle = source.suffix.lower() == ".npy" or source.suffixes[-2:] == [
            ".py",
            ".npy",
        ]
        self._imported_device = load_imported_device(
            source, trusted_pickle=trusted_pickle
        )
        self._import_path = source
        self.source_combo.setCurrentText("Import")
        self._append_log(f"Loaded imported device: {source}")
        self._on_inputs_changed()
        return source

    def _configuration_from_ui(self) -> dict[str, object]:
        inputs = self.collect_inputs()
        layers = [
            {
                "layer": row + 1,
                "thickness_um": self._table_float(row, 1),
                "index_n": self._table_float(row, 2),
                "pattern": self.layer_table.item(row, 3).text()
                if self.layer_table.item(row, 3) is not None
                else "analytic interface",
            }
            for row in range(self.layer_table.rowCount())
        ]
        return {
            "backend": inputs.backend,
            "source_mode": inputs.source_mode,
            "import_path": inputs.import_path,
            "matrix_method": inputs.matrix_method,
            "interface": inputs.interface,
            "polarization": inputs.polarization,
            "distribution": inputs.distribution,
            "period_um": inputs.period_um,
            "height_um": inputs.height_um,
            "harmonics": inputs.harmonics,
            "n_superstrate": inputs.n_superstrate,
            "n_substrate": inputs.n_substrate,
            "wavelength_start_nm": inputs.wavelength_start_nm,
            "wavelength_stop_nm": inputs.wavelength_stop_nm,
            "angle_start_deg": inputs.angle_start_deg,
            "angle_stop_deg": inputs.angle_stop_deg,
            "sweep_points": inputs.sweep_points,
            "nx": inputs.nx,
            "nz": inputs.nz,
            "result_family": inputs.result_family,
            "result_order": inputs.result_order,
            "optimization_objective": inputs.optimization_objective,
            "optimization_lower_bounds": list(inputs.optimization_lower_bounds),
            "optimization_upper_bounds": list(inputs.optimization_upper_bounds),
            "optimization_generations": inputs.optimization_generations,
            "optimization_population": inputs.optimization_population,
            "fdfd_interface": inputs.fdfd_interface,
            "fdfd_palette": inputs.fdfd_palette,
            "fdfd_nres": inputs.fdfd_nres,
            "fdfd_period_num": inputs.fdfd_period_num,
            "fdfd_npml": list(inputs.fdfd_npml),
            "fdfd_spacer_um": list(inputs.fdfd_spacer_um),
            "phc_shape": inputs.phc_shape,
            "phc_repeat_mode": inputs.phc_repeat_mode,
            "phc_layer_count": inputs.phc_layer_count,
            "phc_pitch_um": inputs.phc_pitch_um,
            "phc_background_n": inputs.phc_background_n,
            "phc_inclusion_n": inputs.phc_inclusion_n,
            "phc_wx": inputs.phc_wx,
            "phc_wy": inputs.phc_wy,
            "phc_rotation_deg": inputs.phc_rotation_deg,
            "phc_ax": inputs.phc_ax,
            "phc_ay": inputs.phc_ay,
            "phc_radius": inputs.phc_radius,
            "layers": layers,
        }

    def _apply_configuration(self, configuration: dict[str, object]) -> None:
        self._syncing_layers = True
        try:
            self.backend_combo.setCurrentText(
                str(configuration.get("backend", "Local Python"))
            )
            self.source_combo.setCurrentText(
                str(configuration.get("source_mode", "Analytic"))
            )
            self.method_combo.setCurrentText(
                str(configuration.get("matrix_method", "S"))
            )
            self.interface_combo.setCurrentText(
                str(configuration.get("interface", "sin"))
            )
            self.mode_combo.setCurrentText(str(configuration.get("polarization", "E")))
            self.distribution_combo.setCurrentText(
                str(configuration.get("distribution", "all"))
            )
            self.period_um.setValue(float(configuration.get("period_um", 0.32)))
            self.height_um.setValue(float(configuration.get("height_um", 0.154)))
            self.harmonics.setValue(int(configuration.get("harmonics", 2)))
            self.n_superstrate.setValue(float(configuration.get("n_superstrate", 1.0)))
            self.n_substrate.setValue(float(configuration.get("n_substrate", 1.516)))
            self.wavelength_start_nm.setValue(
                float(configuration.get("wavelength_start_nm", 500.0))
            )
            self.wavelength_stop_nm.setValue(
                float(configuration.get("wavelength_stop_nm", 520.0))
            )
            self.angle_start_deg.setValue(
                float(configuration.get("angle_start_deg", 0.0))
            )
            self.angle_stop_deg.setValue(
                float(configuration.get("angle_stop_deg", 5.0))
            )
            self.sweep_points.setValue(int(configuration.get("sweep_points", 3)))
            self.nx_input.setValue(int(configuration.get("nx", 128)))
            self.nz_input.setValue(int(configuration.get("nz", 5)))
            self.result_family_combo.setCurrentText(
                str(configuration.get("result_family", "Transmission"))
            )
            self.result_order_combo.setCurrentText(
                str(configuration.get("result_order", "0"))
            )
            self.optimization_objective.setCurrentText(
                str(configuration.get("optimization_objective", "R(+1)"))
            )
            self.optimization_generations.setValue(
                int(configuration.get("optimization_generations", 100))
            )
            self.optimization_population.setValue(
                int(configuration.get("optimization_population", 15))
            )
            self._apply_optimization_bounds(
                configuration.get("optimization_lower_bounds", ()),
                configuration.get("optimization_upper_bounds", ()),
            )
            self.fdfd_interface_combo.setCurrentText(
                str(configuration.get("fdfd_interface", "sin"))
            )
            self.fdfd_palette_combo.setCurrentText(
                str(configuration.get("fdfd_palette", "manual legacy"))
            )
            self.fdfd_nres.setValue(float(configuration.get("fdfd_nres", 20.0)))
            self.fdfd_period_num.setValue(int(configuration.get("fdfd_period_num", 11)))
            fdfd_npml = list(configuration.get("fdfd_npml", (20, 20)))
            fdfd_spacer = list(configuration.get("fdfd_spacer_um", (2.0, 2.0)))
            self.fdfd_npml_x.setValue(int(fdfd_npml[0]))
            self.fdfd_npml_y.setValue(int(fdfd_npml[1]))
            self.fdfd_spacer_top.setValue(float(fdfd_spacer[0]))
            self.fdfd_spacer_bottom.setValue(float(fdfd_spacer[1]))
            self.phc_shape_combo.setCurrentText(
                str(configuration.get("phc_shape", "1D Analytic/Import"))
            )
            self.phc_repeat_combo.setCurrentText(
                str(configuration.get("phc_repeat_mode", "legacy repeat"))
            )
            self.phc_layer_count.setValue(int(configuration.get("phc_layer_count", 1)))
            self.phc_pitch_um.setValue(float(configuration.get("phc_pitch_um", 0.32)))
            self.phc_background_n.setValue(
                float(configuration.get("phc_background_n", 1.5))
            )
            self.phc_inclusion_n.setValue(
                float(configuration.get("phc_inclusion_n", 2.0))
            )
            self.phc_wx.setValue(float(configuration.get("phc_wx", 0.2)))
            self.phc_wy.setValue(float(configuration.get("phc_wy", 0.25)))
            self.phc_rotation.setValue(
                float(configuration.get("phc_rotation_deg", 15.0))
            )
            self.phc_ax.setValue(float(configuration.get("phc_ax", 1.0)))
            self.phc_ay.setValue(float(configuration.get("phc_ay", 0.5)))
            self.phc_radius.setValue(float(configuration.get("phc_radius", 0.45)))
            self._apply_layers(configuration.get("layers", ()))
        finally:
            self._syncing_layers = False

        self._imported_device = None
        import_path = configuration.get("import_path")
        self._import_path = Path(str(import_path)).expanduser() if import_path else None
        if self._import_path is not None:
            self._append_log(
                "Import path restored; use Import to explicitly load and trust it."
            )
        self._renumber_layers()

    def _project_workflow_from_inputs(self, inputs: ShellInputs) -> str:
        if inputs.workflow == "optimization":
            return "optimization"
        if inputs.workflow == "fdfd":
            return "fdfd_fields"
        if inputs.workflow == "phc":
            return "casual_phc_rcwa"
        if inputs.source_mode == "Import":
            return "custom_import_rcwa"
        return "casual_rcwa"

    def _select_project_workflow(self, workflow: str) -> None:
        page = PROJECT_WORKFLOW_PAGES.get(workflow, "Project")
        for index in range(self.nav_list.count()):
            item = self.nav_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == page:
                self.nav_list.setCurrentRow(index)
                return

    def _apply_layers(self, layers: object) -> None:
        rows = list(layers) if isinstance(layers, list | tuple) else []
        if not rows:
            return
        self.layer_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            if isinstance(row, dict):
                values = (
                    str(row_index + 1),
                    str(row.get("thickness_um", 0.1)),
                    str(row.get("index_n", 1.5)),
                    str(row.get("pattern", "analytic interface")),
                )
            else:
                values = (str(row_index + 1), "0.1", "1.5", "analytic interface")
            for column_index, value in enumerate(values):
                self.layer_table.setItem(
                    row_index, column_index, QTableWidgetItem(value)
                )

    def _apply_optimization_bounds(self, lower: object, upper: object) -> None:
        lower_values = list(lower) if isinstance(lower, list | tuple) else []
        upper_values = list(upper) if isinstance(upper, list | tuple) else []
        if not lower_values or len(lower_values) != len(upper_values):
            return
        self.optimization_bounds_table.setRowCount(len(lower_values))
        names = ("period_um", "height_um", "first_thickness_um", "sub_L3_um")
        for row, (low, high) in enumerate(zip(lower_values, upper_values)):
            self.optimization_bounds_table.setItem(
                row, 0, QTableWidgetItem(names[row] if row < len(names) else f"p{row}")
            )
            self.optimization_bounds_table.setItem(row, 1, QTableWidgetItem(str(low)))
            self.optimization_bounds_table.setItem(row, 2, QTableWidgetItem(str(high)))

    def _sync_optimization_bounds_for_source(self) -> None:
        if not hasattr(self, "optimization_bounds_table"):
            return
        self.optimization_bounds_table.blockSignals(True)
        try:
            if self.source_combo.currentText() == "Import":
                if self.optimization_bounds_table.rowCount() < 4:
                    row = self.optimization_bounds_table.rowCount()
                    self.optimization_bounds_table.insertRow(row)
                    for column, value in enumerate(("sub_L3_um", "0.04", "0.12")):
                        self.optimization_bounds_table.setItem(
                            row, column, QTableWidgetItem(value)
                        )
                return
            if self.optimization_bounds_table.rowCount() > 3:
                self.optimization_bounds_table.setRowCount(3)
        finally:
            self.optimization_bounds_table.blockSignals(False)

    def export_results(self) -> None:
        if self._last_run is None or self._last_inputs is None:
            self._append_log("Export skipped: no computed results.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Export ZenScat Result Bundle", gettempdir()
        )
        if directory:
            try:
                self.export_results_to(Path(directory))
            except (FileExistsError, OSError, ValueError) as exc:
                self.problems_text.setPlainText(str(exc))
                self._append_log(f"Export failed: {exc}")
                self._set_backend_status(
                    self.backend_combo.currentText(), "Export failed"
                )

    def export_results_to(self, destination: str | Path) -> Path:
        if self._last_run is None or self._last_inputs is None:
            raise RuntimeError("no computed results to export")
        exported = self._service.export(self._last_run, self._last_inputs, destination)
        self._append_log(f"Exported result bundle: {exported}")
        self._set_backend_status(self.backend_combo.currentText(), "Exported")
        return exported

    def _on_run_progress(self, completed: int, total: int) -> None:
        self._set_backend_status(
            self.backend_combo.currentText(), f"Running {completed}/{total}"
        )

    def _on_run_finished(self, run: object) -> None:
        self._last_run = run
        if hasattr(run, "fdfd_bundle"):
            self._show_fdfd_result(run)
        elif hasattr(run, "final_run"):
            self._show_optimization_result(run)
        else:
            self._show_rcwa_result(run)
        self.problems_text.setPlainText("No problems.")
        self.export_button.setEnabled(True)
        self.export_action.setEnabled(True)
        self._append_log("Run completed: computed results available.")
        self._set_backend_status(self.backend_combo.currentText(), "Completed")
        self._update_live_metrics()
        self._refresh_context_inspector()

    def _show_rcwa_result(self, run: object) -> None:
        trn0 = np.asarray(run.transmission.TRN0, dtype=np.float64)
        ref0 = np.asarray(run.reflection.REF0, dtype=np.float64)
        self.dashboard_plot.set_results(trn0, ref0)
        self.sweep_plot.set_results(trn0, ref0)
        self.results_plot.set_results(trn0, ref0)
        workflow = run.metadata.get("workflow", self._last_inputs.workflow)
        if workflow == "casual_rcwa":
            workflow = "analytic RCWA"
        elif workflow == "custom_import_rcwa":
            workflow = "import RCWA"
        elif workflow == "casual_phc_rcwa":
            workflow = "PhC RCWA"
        self.results_text.setPlainText(
            f"Computed {workflow} results\n"
            f"TRN0 range: {float(np.min(trn0)):.6g} to {float(np.max(trn0)):.6g}\n"
            f"REF0 range: {float(np.min(ref0)):.6g} to {float(np.max(ref0)):.6g}\n"
            f"Elapsed: {float(run.elapsed_s):.3f} s"
        )

    def _show_optimization_result(self, run: object) -> None:
        history = np.asarray(
            [snapshot.best_fitness for snapshot in run.optimization.history],
            dtype=np.float64,
        )
        if history.size == 0:
            history = np.asarray([run.final_fitness], dtype=np.float64)
        self.optimize_plot.set_series(history, "best fitness")
        self.optimization_status.setText(
            f"{run.objective}: final fitness {run.final_fitness:.6g}, "
            f"{run.optimization.evaluations} evaluations."
        )
        self._show_rcwa_result(run.final_run)

    def _show_fdfd_result(self, run: object) -> None:
        self._update_field_plot(run)
        self._update_result_plot()
        field = np.asarray(run.result.f)
        er2 = np.asarray(run.device.ER2)
        self.fdfd_status.setText(
            f"FDFD field solved: f {field.shape[0]}x{field.shape[1]}, "
            f"ER2 {er2.shape[0]}x{er2.shape[1]}."
        )
        self.results_text.setPlainText(
            "Computed FDFD field results\n"
            f"Field shape: {field.shape}\n"
            f"ER2 shape: {er2.shape}\n"
            f"Elapsed: {float(run.elapsed_s):.3f} s"
        )

    def _on_run_failed(self, message: str) -> None:
        self.problems_text.setPlainText(message)
        self.results_text.setPlainText(
            "No computed results. Inputs were preserved after the solver error."
        )
        self._append_log(f"Run failed: {message}")
        self._set_backend_status(self.backend_combo.currentText(), "Run failed")
        self._update_live_metrics()

    def _on_run_cancelled(self) -> None:
        self.results_text.setPlainText(
            "No computed results. Run cancelled; inputs were preserved."
        )
        self._append_log("Run cancelled.")
        self._set_backend_status(self.backend_combo.currentText(), "Cancelled")
        self._update_live_metrics()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self._set_running(False)

    def _update_result_plot(self) -> None:
        if not hasattr(self, "results_plot") or self._last_run is None:
            return
        run = (
            self._last_run.final_run
            if hasattr(self._last_run, "final_run")
            else self._last_run
        )
        family = self.result_family_combo.currentText()
        order = self.result_order_combo.currentText()
        try:
            values = self._selected_result_values(run, family, order)
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        label = f"{family} {order}"
        self.results_plot.set_series(values, label)

    def _selected_result_values(
        self, run: object, family: str, order: str
    ) -> np.ndarray:
        if hasattr(run, "result"):
            transmission = run.result.TRN
            reflection = run.result.REF
            if family == "Energy":
                return self._dict_result(transmission, order) + self._dict_result(
                    reflection, order
                )
            source = transmission if family == "Transmission" else reflection
            return self._dict_result(source, order)

        if family == "Energy":
            return self._rcwa_result(run.transmission, order) + self._rcwa_result(
                run.reflection, order
            )
        source = run.transmission if family == "Transmission" else run.reflection
        return self._rcwa_result(source, order)

    def _rcwa_result(self, result: object, order: str) -> np.ndarray:
        name = {"-1": "minus_1", "0": "TRN0", "+1": "plus_1", "sum": "sum"}[order]
        if not hasattr(result, name):
            name = {"-1": "minus_1", "0": "REF0", "+1": "plus_1", "sum": "sum"}[order]
        return np.asarray(getattr(result, name), dtype=np.float64)

    def _dict_result(self, result: dict[str, object], order: str) -> np.ndarray:
        for key in (
            {"-1": "minus_1", "0": "TRN0", "+1": "plus_1", "sum": "sum"}[order],
            {"-1": "TRN_minus1", "0": "TRN0", "+1": "TRN_plus1", "sum": "sum"}[order],
            {"-1": "REF_minus1", "0": "REF0", "+1": "REF_plus1", "sum": "sum"}[order],
        ):
            if key in result:
                return np.asarray(result[key], dtype=np.float64)
        raise KeyError(order)

    def _update_field_plot(self, run: object | None = None) -> None:
        if not hasattr(self, "field_plot"):
            return
        target = run or self._last_run
        if target is None or not hasattr(target, "result"):
            return
        view = self.fdfd_field_combo.currentText()
        if view == "real(f)":
            values = np.real(target.result.f)
        elif view == "ER2":
            values = target.device.ER2
        else:
            values = np.abs(target.result.f)
        self.field_plot.set_heatmap(np.asarray(values), view)

    def _set_running(self, running: bool) -> None:
        self.validate_button.setEnabled(not running)
        self.validate_action.setEnabled(not running)
        self.run_button.setEnabled(not running and self._validated)
        self.run_action.setEnabled(not running and self._validated)
        self.cancel_button.setEnabled(running)
        self.cancel_action.setEnabled(running)
        self.export_button.setEnabled(not running and self._last_run is not None)
        self.export_action.setEnabled(not running and self._last_run is not None)
        QApplication.setOverrideCursor(
            Qt.CursorShape.BusyCursor
        ) if running else QApplication.restoreOverrideCursor()

    def _on_inputs_changed(self) -> None:
        if getattr(self, "_syncing_layers", False):
            return
        self._validated = False
        self._last_run = None
        self.run_button.setEnabled(False)
        self.run_action.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_action.setEnabled(False)
        if hasattr(self, "dashboard_plot"):
            self.dashboard_plot.clear_results()
            self.sweep_plot.clear_results()
            self.results_plot.clear_results()
            self.field_plot.clear_results()
            self.optimize_plot.clear_results()
        self.results_text.setPlainText(
            "No computed results. Validate the active workflow, then Run to "
            "populate the selected output view. Export unlocks after completion."
        )
        self._sync_project_stack_table()
        self._update_grid_summary()
        self._refresh_context_inspector()

    def _collect_params(self) -> tuple[float, ...]:
        thicknesses: list[float] = []
        indices: list[float] = []
        for row in range(self.layer_table.rowCount()):
            thicknesses.append(self._table_float(row, 1))
            indices.append(self._table_float(row, 2))
        return tuple(thicknesses + indices)

    def _table_float(self, row: int, column: int) -> float:
        item = self.layer_table.item(row, column)
        if item is None:
            raise ValueError("layer table contains empty cells")
        return float(item.text())

    def add_layer(self) -> None:
        self.layer_table.blockSignals(True)
        row = self.layer_table.rowCount()
        try:
            self.layer_table.insertRow(row)
            values = (str(row + 1), "0.100", "1.500", "analytic interface")
            for column, value in enumerate(values):
                self.layer_table.setItem(row, column, QTableWidgetItem(value))
            self._renumber_layers()
        finally:
            self.layer_table.blockSignals(False)
        self._on_inputs_changed()

    def remove_layer(self) -> None:
        if self.layer_table.rowCount() <= 1:
            return
        row = self.layer_table.currentRow()
        if row < 0:
            row = self.layer_table.rowCount() - 1
        self.layer_table.blockSignals(True)
        try:
            self.layer_table.removeRow(row)
            self._renumber_layers()
        finally:
            self.layer_table.blockSignals(False)
        self._on_inputs_changed()

    def move_layer(self, direction: int) -> None:
        row = self.layer_table.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.layer_table.rowCount():
            return
        self.layer_table.blockSignals(True)
        values = [
            [
                self.layer_table.item(r, c).text()
                if self.layer_table.item(r, c) is not None
                else ""
                for c in range(4)
            ]
            for r in (row, target)
        ]
        try:
            for column, value in enumerate(values[1]):
                self.layer_table.setItem(row, column, QTableWidgetItem(value))
            for column, value in enumerate(values[0]):
                self.layer_table.setItem(target, column, QTableWidgetItem(value))
            self.layer_table.setCurrentCell(target, 0)
            self._renumber_layers()
        finally:
            self.layer_table.blockSignals(False)
        self._on_inputs_changed()

    def _renumber_layers(self) -> None:
        self._syncing_layers = True
        try:
            for row in range(self.layer_table.rowCount()):
                self.layer_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.layer_count.setValue(self.layer_table.rowCount())
            self._sync_project_stack_table()
        finally:
            self._syncing_layers = False

    def _sync_project_stack_table(self) -> None:
        if not hasattr(self, "project_stack_table"):
            return
        self._syncing_layers = True
        try:
            self.project_stack_table.setRowCount(self.layer_table.rowCount())
            for row in range(self.layer_table.rowCount()):
                for column in range(4):
                    source = self.layer_table.item(row, column)
                    self.project_stack_table.setItem(
                        row,
                        column,
                        QTableWidgetItem("" if source is None else source.text()),
                    )
        finally:
            self._syncing_layers = False

    def _set_backend_status(self, backend: str, state: str) -> None:
        self.backend_status.setText(f"Backend: {backend}")
        self.job_status.setText(f"Status: {state}")
        if hasattr(self, "metricBackendValue"):
            self._update_live_metrics()

    def _update_project_status(self) -> None:
        name = self.project_name.currentText().strip() or "Untitled"
        self.project_status.setText(f"Project: {name}")

    def _update_grid_summary(self) -> None:
        order_count = 2 * self.harmonics.value() + 1
        cells = order_count * max(self.layer_count.value(), 1)
        method = self.method_combo.currentText()
        polarization = self.mode_combo.currentText()
        interface = self.interface_combo.currentText()
        wavelength_range = (
            f"{self.wavelength_start_nm.value():.1f}-"
            f"{self.wavelength_stop_nm.value():.1f} nm"
        )
        angle_range = (
            f"{self.angle_start_deg.value():.2f}-{self.angle_stop_deg.value():.2f} deg"
        )
        self.grid_summary.setText(
            f"{method}/{polarization} {interface} with {order_count} Fourier orders "
            f"({self.harmonics.value()} harmonics each side), "
            f"{cells} layer-order cells, "
            f"{self.period_um.value():.4g} um period, "
            f"Nx {self.nx_input.value()}, Nz {self.nz_input.value()}, "
            f"{wavelength_range}, "
            f"{angle_range}, "
            f"{self.sweep_points.value()} points."
        )
        self._update_profile_preview()
        self._update_live_metrics()

    def _update_profile_preview(self) -> None:
        if not hasattr(self, "structure_plot"):
            return
        try:
            inputs = self.collect_inputs()
        except (AttributeError, TypeError, ValueError):
            return
        if inputs.source_mode == "Import" and inputs.imported_device is not None:
            device = inputs.imported_device
            thicknesses = np.asarray(device.sub_L_um, dtype=np.float64)
            indices = np.sqrt(np.maximum(np.mean(np.real(device.ER), axis=1), 0.0))
            profile_kwargs = {
                "period_um": float(device.Lx_um),
                "height_um": float(np.sum(thicknesses)),
                "n_superstrate": inputs.n_superstrate,
                "n_substrate": inputs.n_substrate,
            }
            self.structure_plot.set_profile(
                thicknesses, indices, "imported device", **profile_kwargs
            )
            self.project_device_plot.set_profile(
                thicknesses, indices, "imported device", **profile_kwargs
            )
            return
        from zenscat.core import sample_interface_profile

        layer_count = max(inputs.layer_count, 1)
        thicknesses = np.asarray(inputs.params[:layer_count], dtype=np.float64)
        indices = np.asarray(inputs.params[layer_count:], dtype=np.float64)
        x_um, z_um = sample_interface_profile(
            inputs.interface,
            period_um=inputs.period_um,
            height_um=inputs.height_um,
            sample_count=256,
        )
        profile_kwargs = {
            "x_um": x_um,
            "z_um": z_um,
            "period_um": inputs.period_um,
            "height_um": inputs.height_um,
            "n_superstrate": inputs.n_superstrate,
            "n_substrate": inputs.n_substrate,
        }
        self.structure_plot.set_profile(
            thicknesses, indices, inputs.interface, **profile_kwargs
        )
        self.project_device_plot.set_profile(
            thicknesses, indices, inputs.interface, **profile_kwargs
        )

    def _update_live_metrics(self) -> None:
        if not hasattr(self, "metricBackendValue"):
            return
        try:
            inputs = self.collect_inputs()
        except (AttributeError, TypeError, ValueError):
            return

        total_nm = sum(inputs.params[: inputs.layer_count]) * 1000.0
        imported = inputs.source_mode == "Import" and inputs.imported_device is not None
        device_layers = (
            int(inputs.imported_device.ER.shape[0]) if imported else inputs.layer_count
        )
        device_total_nm = (
            float(np.sum(inputs.imported_device.sub_L_um) * 1000.0)
            if imported
            else total_nm
        )

        self.metricBackendValue.setText(inputs.source_mode)
        backend_detail = inputs.backend
        if imported:
            backend_detail = f"{backend_detail}; {Path(inputs.import_path or '').name}"
        self.metricBackendDetail.setText(backend_detail)
        self.metricDeviceValue.setText(f"{device_layers} layers")
        self.metricDeviceDetail.setText(
            f"{self._format_number(device_total_nm)} nm total, "
            f"{'imported profile' if imported else inputs.interface + ' profile'}"
        )
        self.metricSweepValue.setText(
            self._format_range(
                inputs.wavelength_start_nm,
                inputs.wavelength_stop_nm,
                "nm",
            )
        )
        angle_range = self._format_range(
            inputs.angle_start_deg, inputs.angle_stop_deg, "deg"
        )
        wavelength_range = self._format_range(
            inputs.wavelength_start_nm, inputs.wavelength_stop_nm, "nm"
        )
        self.metricSweepDetail.setText(
            f"{angle_range}, {inputs.sweep_points} pts, "
            f"{inputs.matrix_method}/{inputs.polarization}"
        )

        self.project_science_summary.setText(
            f"{inputs.source_mode} {inputs.matrix_method}/{inputs.polarization}; "
            f"period {self._format_number(inputs.period_um)} um, "
            f"height {self._format_number(inputs.height_um)} um, "
            f"harmonics {inputs.harmonics}, Nx {inputs.nx}, Nz {inputs.nz}; "
            f"ambient n {self._format_number(inputs.n_superstrate)}, "
            f"substrate n {self._format_number(inputs.n_substrate)}; "
            f"sweep {wavelength_range} x {angle_range}."
        )
        self.fdfd_setup_summary.setText(
            "Bound setup: "
            f"{wavelength_range}, {angle_range}; "
            f"period {self._format_number(inputs.period_um)} um, "
            f"height {self._format_number(inputs.height_um)} um; "
            f"{inputs.polarization} mode, {inputs.distribution} distribution; "
            f"ambient n {self._format_number(inputs.n_superstrate)}, "
            f"substrate n {self._format_number(inputs.n_substrate)}; "
            f"{inputs.layer_count} layers, NRES {self._format_number(inputs.fdfd_nres)}, "
            f"periods {inputs.fdfd_period_num}, NPML {inputs.fdfd_npml[0]}/{inputs.fdfd_npml[1]}."
        )
        self.import_source_label.setText(self._import_summary(inputs))
        self._update_navigation_states(inputs)

    def _update_navigation_states(self, inputs: ShellInputs) -> None:
        states = {
            "Project": "ready",
            "Device": f"{inputs.layer_count} layers",
            "RCWA Sweep": "validated" if self._validated else "configured",
            "Optimize": "ready",
            "FDFD Fields": "ready",
            "Results": "ready" if self._last_run is not None else "empty",
            "Jobs": "running" if self._thread is not None else "idle",
        }
        for index in range(self.nav_list.count()):
            item = self.nav_list.item(index)
            label = str(item.data(Qt.ItemDataRole.UserRole))
            item.setText(f"{label}\n{states.get(label, 'ready')}")

        self.workflow1Badge.setText("1 Project metadata  ready")
        self.workflow2Badge.setText(f"2 Device stack  {states['Device']}")
        self.workflow3Badge.setText(f"3 RCWA sweep  {states['RCWA Sweep']}")
        self.workflow4Badge.setText(
            "4 Solve/export  ready"
            if self._last_run is not None
            else "4 Solve/export  waiting"
        )

    def _import_summary(self, inputs: ShellInputs) -> str:
        if inputs.imported_device is None:
            return "No imported device loaded."
        device = inputs.imported_device
        source = Path(inputs.import_path or "").name or "memory"
        thickness_nm = float(np.sum(device.sub_L_um) * 1000.0)
        return (
            f"Imported source: {source}; ER {device.ER.shape[0]}x{device.ER.shape[1]}, "
            f"{self._format_number(device.Lx_um)} um period, "
            f"{self._format_number(thickness_nm)} nm total."
        )

    def _format_range(self, start: float, stop: float, unit: str) -> str:
        return f"{self._format_number(start)}-{self._format_number(stop)} {unit}"

    def _format_number(self, value: float) -> str:
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return text or "0"

    def _refresh_context_inspector(self, context: str | None = None) -> None:
        current = context
        if (
            current is None
            and hasattr(self, "nav_list")
            and self.nav_list.currentItem() is not None
        ):
            current = str(self.nav_list.currentItem().data(Qt.ItemDataRole.UserRole))
        current = current or "Project"
        inputs = self.collect_inputs()
        self.inspector_text.setPlainText(self._inspector_body(current, inputs))

    def _inspector_body(self, current: str, inputs: ShellInputs) -> str:
        source = inputs.source_mode
        if inputs.import_path:
            source = f"{source} ({Path(inputs.import_path).name})"
        sweep = self._format_range(
            inputs.wavelength_start_nm, inputs.wavelength_stop_nm, "nm"
        )
        angles = self._format_range(
            inputs.angle_start_deg, inputs.angle_stop_deg, "deg"
        )
        interface_description = PROFILE_INTERFACE_DESCRIPTIONS.get(
            inputs.interface, inputs.interface
        )
        fdfd_interface_description = PROFILE_INTERFACE_DESCRIPTIONS.get(
            inputs.fdfd_interface, inputs.fdfd_interface
        )
        header = (
            "Context inspector\n"
            f"Section: {current}\n"
            f"Project: {inputs.project_name or 'Untitled'}\n"
            f"Backend: {inputs.backend}\n"
            f"Source: {source}\n"
        )
        if current == "RCWA Sweep":
            if inputs.workflow == "phc":
                return (
                    header + f"PhC: {inputs.phc_shape}, {inputs.phc_repeat_mode}, "
                    f"{inputs.phc_layer_count} layers\n"
                    f"Grid: {inputs.harmonics} harmonics, Nx {inputs.nx}, Nz {inputs.nz}\n"
                    f"Sweep: {sweep}, {angles}, {inputs.sweep_points} points\n\n"
                    "Validate builds a PhC S-matrix request. Run computes real "
                    "transmission/reflection orders; Export writes the RCWA bundle."
                )
            return (
                header + f"RCWA: {inputs.matrix_method}/{inputs.polarization}, "
                f"{inputs.interface} ({interface_description}), "
                f"{inputs.distribution}\n"
                f"Grid: {inputs.harmonics} harmonics, Nx {inputs.nx}, Nz {inputs.nz}\n"
                f"Sweep: {sweep}, {angles}, {inputs.sweep_points} points\n\n"
                "Validate builds the analytic or imported RCWA request. Run computes "
                "TRN/REF orders on the worker thread; Export writes the legacy bundle."
            )
        if current == "Optimize":
            return (
                header + f"Objective: {inputs.optimization_objective}\n"
                f"Generations/population: {inputs.optimization_generations}/"
                f"{inputs.optimization_population}\n"
                f"Bounds lower: {list(inputs.optimization_lower_bounds)}\n"
                f"Bounds upper: {list(inputs.optimization_upper_bounds)}\n\n"
                "Validate checks legacy bounds for the current source. Run evaluates "
                "the bounded objective and verifies the final physical RCWA result; "
                "Export writes that final RCWA bundle."
            )
        if current == "FDFD Fields":
            return (
                header + f"FDFD: {inputs.fdfd_interface} "
                f"({fdfd_interface_description}), {inputs.fdfd_palette}, "
                f"{inputs.polarization} mode\n"
                f"Domain: period {self._format_number(inputs.period_um)} um, "
                f"height {self._format_number(inputs.height_um)} um, "
                f"NRES {self._format_number(inputs.fdfd_nres)}, "
                f"periods {inputs.fdfd_period_num}\n"
                f"Boundary: NPML {inputs.fdfd_npml[0]}/{inputs.fdfd_npml[1]}, "
                f"spacer {inputs.fdfd_spacer_um[0]}/{inputs.fdfd_spacer_um[1]} um\n\n"
                "Validate builds the FDFD field request from the current stack. Run "
                "computes f and ER2; Export writes Field.mat, ER2.mat, TRN/REF, "
                "Lam/Theta, Params, and manifest."
            )
        if current == "Results":
            status = "available" if self._last_run is not None else "empty"
            return (
                header + f"Results: {status}\n"
                f"Selector: {inputs.result_family} {inputs.result_order}\n\n"
                "After a run, use the selectors to plot transmission, reflection, "
                "or energy for -1/0/+1/sum. Export is enabled only when computed "
                "results are present."
            )
        if current == "Jobs":
            state = "running" if self._thread is not None else "idle"
            return (
                header + f"Job state: {state}\n\n"
                "Run jobs execute on one QThread with a shared cancellation event. "
                "Cancel requests are preserved in the log, and inputs remain editable "
                "after the worker exits."
            )
        return (
            header + f"Device: {inputs.layer_count} analytic layers, "
            f"Params {list(inputs.params)}\n"
            f"Interface: {inputs.interface} ({interface_description})\n"
            f"Media: ambient n {self._format_number(inputs.n_superstrate)}, "
            f"substrate n {self._format_number(inputs.n_substrate)}\n"
            f"Sweep: {sweep}, {angles}, {inputs.sweep_points} points\n\n"
            "Validate checks the current workflow. Run starts the selected computation; "
            "Export unlocks after computed results exist."
        )

    def _set_busy(self, busy: bool) -> None:
        self.validate_button.setEnabled(not busy)
        self.run_button.setEnabled(False)
        self.validate_action.setEnabled(not busy)
        self.run_action.setEnabled(False)
        QApplication.setOverrideCursor(
            Qt.CursorShape.BusyCursor
        ) if busy else QApplication.restoreOverrideCursor()

    def _append_log(self, message: str) -> None:
        if hasattr(self, "job_log"):
            self.job_log.append(message)
        if hasattr(self, "jobs_text"):
            self.jobs_text.append(message)


__all__ = ["MainWindow", "ShellInputs", "SolverService", "ValidationResult"]
