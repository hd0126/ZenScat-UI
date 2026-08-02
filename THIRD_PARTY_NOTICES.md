# Third-party notices

ZenScat's Python source is distributed under the repository's MIT license.
Runtime dependencies retain their own licenses.

- NumPy — BSD-3-Clause.
- SciPy — BSD-3-Clause and compatible bundled notices.
- PySide6 / Qt for Python — LGPL-3.0-only, GPL alternatives, or a commercial
  Qt license, depending on the distributor's chosen terms. The corresponding
  GNU GPLv3 and LGPLv3 texts are included under `licenses/`.
- PyInstaller — GPL-2.0-or-later with a bootloader exception that permits
  distribution of the bundled application.

The release build must keep Qt libraries dynamically replaceable and include
these license texts together with PySide6 distribution metadata. A distributor
choosing Qt's commercial license must follow the corresponding commercial
agreement instead.
