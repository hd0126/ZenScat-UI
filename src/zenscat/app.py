"""Application entry point for the ZenScat desktop shell."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Launch the Qt desktop application."""

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    if os.environ.get("ZENSCAT_SMOKE_TEST") == "1" and sys.platform == "darwin":
        # Qt for Python's macOS wheel exposes ``minimal`` as its supported
        # headless platform plugin; normal launches continue to use Cocoa.
        os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

    from zenscat.gui import MainWindow
    from zenscat.gui.qt_compat import QApplication, QTimer

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if os.environ.get("ZENSCAT_SMOKE_TEST") == "1":
        QTimer.singleShot(0, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
