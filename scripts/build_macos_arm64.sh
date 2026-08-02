#!/bin/zsh
set -euo pipefail

repo_root=${0:A:h:h}
cd "$repo_root"
zenscat_python_bin=${ZENSCAT_PYTHON:-python3}
staging_dir=""
cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$staging_dir" && "$staging_dir" == "${TMPDIR:-/tmp}/zenscat-sign."* && -d "$staging_dir" ]]; then
    rm -rf "$staging_dir"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "Apple Silicon macOS is required for this build."
  exit 2
fi

if ! command -v "$zenscat_python_bin" >/dev/null 2>&1; then
  print -u2 "Python interpreter not found: $zenscat_python_bin"
  exit 2
fi

python_arch=$($zenscat_python_bin -c 'import platform; print(platform.machine())')
if [[ "$python_arch" != "arm64" ]]; then
  print -u2 "The active Python interpreter is $python_arch; an arm64 interpreter is required."
  exit 2
fi

qt_plugin_root=$($zenscat_python_bin -c 'from PySide6.QtCore import QLibraryInfo; print(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))')
if [[ -d "$qt_plugin_root" ]]; then
  # Files created below a hidden virtualenv can inherit UF_HIDDEN on macOS;
  # Qt's plugin scanner then skips Cocoa even though the dylib is valid.
  chflags -R nohidden "$qt_plugin_root"
  xattr -cr "$qt_plugin_root"
fi

export ZENSCAT_RELEASE_BUILD=1
$zenscat_python_bin -m PyInstaller --noconfirm --clean packaging/zenscat.spec

app_path="$repo_root/dist/ZenScat.app"
binary_path="$app_path/Contents/MacOS/ZenScat"
if [[ ! -x "$binary_path" ]]; then
  print -u2 "PyInstaller did not produce the expected executable: $binary_path"
  exit 1
fi

if ! file "$binary_path" | grep -q 'arm64'; then
  print -u2 "Built executable is not arm64: $(file "$binary_path")"
  exit 1
fi

for notice in LICENSE THIRD_PARTY_NOTICES.md; do
  if [[ ! -f "$app_path/Contents/Resources/$notice" ]]; then
    print -u2 "Release notice was not packaged: $notice"
    exit 1
  fi
done

for qt_license in licenses/GPL-3.0.txt licenses/LGPL-3.0.txt; do
  if [[ ! -f "$app_path/Contents/Resources/$qt_license" ]]; then
    print -u2 "Qt license text was not packaged: $qt_license"
    exit 1
  fi
done

if ! find "$app_path/Contents" -type f -ipath '*pyside6*.dist-info/*' -print -quit | grep -q .; then
  print -u2 "PySide6 distribution metadata and license files were not packaged."
  exit 1
fi

chflags -R nohidden "$app_path"
xattr -cr "$app_path"

staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/zenscat-sign.XXXXXX")
sanitized_app_path="$staging_dir/ZenScat.app"
archive_verify_dir="$staging_dir/archive-verify"
zip_path="$repo_root/dist/ZenScat-arm64.zip"
staged_zip_path="$staging_dir/ZenScat-arm64.zip"
ditto --norsrc --noextattr "$app_path" "$sanitized_app_path"
find "$sanitized_app_path" -xattrname com.apple.FinderInfo -exec xattr -d com.apple.FinderInfo {} +

codesign --force --deep --sign - "$sanitized_app_path"
codesign --verify --deep --strict "$sanitized_app_path"

ditto -c -k --norsrc --noextattr --keepParent "$sanitized_app_path" "$staged_zip_path"
mkdir -p "$archive_verify_dir"
ditto -x -k "$staged_zip_path" "$archive_verify_dir"
codesign --verify --deep --strict "$archive_verify_dir/ZenScat.app"
mv "$staged_zip_path" "$zip_path"

rm -rf "$app_path"
ditto --norsrc --noextattr "$sanitized_app_path" "$app_path"
print "Built and ad-hoc signed $app_path"
print "Built stable signed archive $zip_path"
