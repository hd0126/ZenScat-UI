from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_macos_packaging_targets_native_arm64_application() -> None:
    spec = (ROOT / "packaging" / "zenscat.spec").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "build_macos_arm64.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_macos_arm64.sh").read_text(encoding="utf-8")

    assert 'target_arch="arm64"' in spec
    assert "repo_root = Path(SPECPATH).parent\n" in spec
    assert 'name="ZenScat.app"' in spec
    assert "LSMinimumSystemVersion" in spec
    assert 'excludes=["PyQt5", "PyQt6", "PySide2"]' in spec
    assert "from PyInstaller.utils.hooks import copy_metadata" in spec
    assert '"PySide6",' in spec
    assert '(str(repo_root / "licenses"), "licenses")' in spec
    assert '"$(uname -m)" != "arm64"' in script
    assert "ZENSCAT_PYTHON" in script
    assert "ZENSCAT_RELEASE_BUILD=1" in script
    assert "chflags -R nohidden" in script
    assert "ditto -c -k --norsrc --noextattr --keepParent" in script
    assert "ZenScat-arm64.zip" in script
    assert "archive-verify" in script
    assert "codesign --verify --deep --strict" in script
    assert "mktemp -d" in launcher
    assert "cp -X" in launcher
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" in launcher
    assert "libqcocoa.dylib" in launcher
    assert 'PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"' in launcher
    assert 'exec "$zenscat_python_bin" -m zenscat.app' not in launcher
    assert '"$zenscat_python_bin" -m zenscat.app' in launcher


def test_pyproject_exposes_gui_and_cli_entrypoint() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'PySide6>=6.7,<7' in pyproject
    assert 'zenscat = "zenscat.app:main"' in pyproject
    assert 'PyInstaller>=6.10,<7' in pyproject
    assert 'requires-python = ">=3.12,<3.15"' in pyproject


def test_release_bundle_declares_and_packages_qt_license_texts() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    gpl = ROOT / "licenses" / "GPL-3.0.txt"
    lgpl = ROOT / "licenses" / "LGPL-3.0.txt"

    assert "PySide6 / Qt for Python" in notices
    assert "GNU GENERAL PUBLIC LICENSE" in gpl.read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in lgpl.read_text(encoding="utf-8")
