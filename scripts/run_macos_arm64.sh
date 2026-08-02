#!/bin/zsh
set -euo pipefail

repo_root=${0:A:h:h}
cd "$repo_root"
zenscat_python_bin=${ZENSCAT_PYTHON:-.venv/bin/python}
plugin_staging_dir=""
cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$plugin_staging_dir" && "$plugin_staging_dir" == "${TMPDIR:-/tmp}/zenscat-qt."* && -d "$plugin_staging_dir" ]]; then
    rm -rf "$plugin_staging_dir"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "Apple Silicon macOS is required by this launcher."
  exit 2
fi

if [[ ! -x "$zenscat_python_bin" ]]; then
  print -u2 "Python interpreter not found: $zenscat_python_bin"
  exit 2
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

qt_plugin_root=$($zenscat_python_bin -c 'from PySide6.QtCore import QLibraryInfo; print(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))')
qt_platform_root="$qt_plugin_root/platforms"
if [[ -d "$qt_platform_root" ]]; then
  # FileProvider-backed checkouts can immediately re-add UF_HIDDEN to the
  # venv plugins. Run from a clean temporary plugin directory instead.
  plugin_staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/zenscat-qt.XXXXXX")
  staged_platform_root="$plugin_staging_dir/platforms"
  mkdir -p "$staged_platform_root"
  for plugin in libqcocoa.dylib libqminimal.dylib libqoffscreen.dylib; do
    if [[ -f "$qt_platform_root/$plugin" ]]; then
      cp -X "$qt_platform_root/$plugin" "$staged_platform_root/$plugin"
    fi
  done
  if [[ ! -f "$staged_platform_root/libqcocoa.dylib" ]]; then
    print -u2 "Required Qt Cocoa platform plugin was not found below: $qt_platform_root"
    exit 1
  fi
  chflags -R nohidden "$plugin_staging_dir"
  xattr -cr "$plugin_staging_dir"
  export QT_QPA_PLATFORM_PLUGIN_PATH="$staged_platform_root"
fi

"$zenscat_python_bin" -m zenscat.app
