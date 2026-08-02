# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


repo_root = Path(SPECPATH).parent
source_root = repo_root / "src"
qt_license_metadata = []
for distribution in (
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
):
    qt_license_metadata += copy_metadata(distribution)

a = Analysis(
    [str(source_root / "zenscat" / "app.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=[
        (str(repo_root / "compatibility" / "manifest-v1.json"), "compatibility"),
        (str(repo_root / "LICENSE"), "."),
        (str(repo_root / "THIRD_PARTY_NOTICES.md"), "."),
        (str(repo_root / "licenses"), "licenses"),
    ]
    + qt_license_metadata,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZenScat",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
contents = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ZenScat",
)
app = BUNDLE(
    contents,
    name="ZenScat.app",
    icon=None,
    bundle_identifier="io.github.hd0126.zenscat",
    version="0.1.0",
    info_plist={
        "CFBundleDisplayName": "ZenScat",
        "CFBundleName": "ZenScat",
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    },
)
