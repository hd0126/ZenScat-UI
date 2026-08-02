"""Qt binding compatibility layer.

PySide6 is preferred for MIT-friendly redistribution.  PyQt6 remains a
development/test fallback for environments where PySide6 is not installed.
"""

from __future__ import annotations

import os

QT_BINDING = "PySide6"
_REQUESTED_BINDING = os.environ.get("ZENSCAT_QT_BINDING", "PySide6")

try:  # pragma: no cover - exercised when PySide6 is installed.
    if _REQUESTED_BINDING == "PyQt6":
        raise ImportError("PyQt6 requested for development/test run")
    from PySide6.QtCore import (
        QCoreApplication,
        QObject,
        QRectF,
        Qt,
        QThread,
        QTimer,
        Signal,
    )
    from PySide6.QtGui import QAction, QColor, QPainter, QPainterPath, QPalette, QPen
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QListView,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - environment dependent.
    if os.environ.get("ZENSCAT_RELEASE_BUILD") == "1":
        raise RuntimeError(
            "release builds must bundle PySide6; PyQt6 fallback disabled"
        )
    QT_BINDING = "PyQt6"
    from PyQt6.QtCore import (
        QCoreApplication,
        QObject,
        QRectF,
        Qt,
        QThread,
        QTimer,
    )
    from PyQt6.QtCore import (
        pyqtSignal as Signal,
    )
    from PyQt6.QtGui import QAction, QColor, QPainter, QPainterPath, QPalette, QPen
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QListView,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )


__all__ = [
    "QT_BINDING",
    "QAbstractSpinBox",
    "QAction",
    "QApplication",
    "QColor",
    "QComboBox",
    "QCoreApplication",
    "QDoubleSpinBox",
    "QFileDialog",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QLabel",
    "QListView",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QObject",
    "QPainter",
    "QPainterPath",
    "QPalette",
    "QPen",
    "QPushButton",
    "QRectF",
    "QScrollArea",
    "QSizePolicy",
    "QSpinBox",
    "QStackedWidget",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QTextEdit",
    "QThread",
    "QTimer",
    "QVBoxLayout",
    "QWidget",
    "Qt",
    "Signal",
]
