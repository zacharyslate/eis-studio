#!/bin/sh
# EIS Studio installer for macOS and Linux.
#
#   curl -LsSf https://zacharyslate.github.io/eis-studio/install.sh | sh
#
# What it does (nothing needs administrator rights):
#   1. installs uv (Astral's Python package manager) into ~/.local/bin if it is missing;
#   2. lets uv download a private Python 3.12 and install EIS Studio into its own isolated
#      environment (your system Python is not touched);
#   3. adds an "eis-studio" command and a double-clickable shortcut.
# Uninstall:  uv tool uninstall eis-studio   (and delete the shortcut)
set -eu

SOURCE="${EIS_STUDIO_SOURCE:-eis-studio @ https://github.com/zacharyslate/eis-studio/archive/refs/heads/main.zip}"
PYVER="${EIS_STUDIO_PYTHON:-3.12}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "Installing EIS Studio"

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
  else
    say "Installing uv (Python package manager, https://docs.astral.sh/uv/)"
    TMP_UV="$(mktemp)"
    # download first, then run: a failed download must stop us, not pipe an error page into sh
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh -o "$TMP_UV" || { echo "Could not download uv's installer." >&2; exit 1; }
    else
      wget -qO "$TMP_UV" https://astral.sh/uv/install.sh || { echo "Could not download uv's installer." >&2; exit 1; }
    fi
    sh "$TMP_UV"
    rm -f "$TMP_UV"
    PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { echo "uv was not found after installing it; see https://docs.astral.sh/uv/" >&2; exit 1; }
  fi
  export PATH
fi

say "Downloading EIS Studio and its scientific libraries (first install: a few minutes)"
# --no-build-package: never compile these from source; if the newest release has no wheel for
# this OS (e.g. cvxopt 1.3.3 on macOS < 15), uv falls back to the newest version that has one.
uv tool install --force --refresh-package eis-studio --python "$PYVER" \
  --no-build-package cvxopt --no-build-package numpy --no-build-package scipy \
  --no-build-package scikit-learn --no-build-package matplotlib --no-build-package pandas \
  --no-build-package pyarrow --no-build-package contourpy --no-build-package kiwisolver \
  "$SOURCE"

uv tool update-shell >/dev/null 2>&1 || true
BIN="$(uv tool dir --bin)/eis-studio"

ICON_PNG="$("$BIN" --icon-path png)"
VERSION="$("$BIN" --version | awk '{print $NF}')"

case "$(uname -s)" in
  Darwin)
    # A real application bundle, so the icon shows in Finder, Launchpad, Spotlight and the Dock.
    # It opens Terminal running eis-studio; closing that window quits the app (same as on Windows).
    APPDIR="$HOME/Applications/EIS Studio.app"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/Contents/MacOS" "$APPDIR/Contents/Resources"
    cp "$("$BIN" --icon-path icns)" "$APPDIR/Contents/Resources/eis-studio.icns"
    cat > "$APPDIR/Contents/MacOS/eis-studio-launcher" <<LAUNCH
#!/bin/sh
exec open -a Terminal "$BIN"
LAUNCH
    chmod +x "$APPDIR/Contents/MacOS/eis-studio-launcher"
    cat > "$APPDIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>EIS Studio</string>
  <key>CFBundleDisplayName</key><string>EIS Studio</string>
  <key>CFBundleIdentifier</key><string>io.github.zacharyslate.eisstudio</string>
  <key>CFBundleExecutable</key><string>eis-studio-launcher</string>
  <key>CFBundleIconFile</key><string>eis-studio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>10.13</string>
</dict>
</plist>
PLIST
    touch "$APPDIR"   # makes Finder pick up the new icon
    rm -f "$HOME/Desktop/EIS Studio.command"   # shortcut from earlier versions
    ln -sfn "$APPDIR" "$HOME/Desktop/EIS Studio"
    SHORTCUT="$APPDIR (also linked on your Desktop)"
    ;;
  Linux)
    ICONDIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons"
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$ICONDIR" "$APPS"
    cp "$ICON_PNG" "$ICONDIR/eis-studio.png"
    SHORTCUT="$APPS/eis-studio.desktop"
    cat > "$SHORTCUT" <<DESKTOP
[Desktop Entry]
Type=Application
Name=EIS Studio
Comment=Impedance analysis: equivalent circuits and DRT
Exec="$BIN"
Icon=$ICONDIR/eis-studio.png
Terminal=true
Categories=Science;Education;
DESKTOP
    ;;
  *) SHORTCUT="" ;;
esac

say "Done."
echo "Start EIS Studio with the command:  eis-studio"
[ -n "$SHORTCUT" ] && echo "or with the shortcut:              $SHORTCUT"
echo "(Open a new terminal window first if the command is not found.)"
echo "It opens in your web browser and runs only on this computer; your data is not uploaded anywhere."
