#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="${1:-/Applications}"
APP_NAME="${2:-LTX Desktop Local.app}"
APP_PATH="$INSTALL_DIR/$APP_NAME"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ "$APP_NAME" != *.app ]]; then
  APP_NAME="$APP_NAME.app"
  APP_PATH="$INSTALL_DIR/$APP_NAME"
fi

if [[ ! -x "$REPO_ROOT/scripts/start-mac-local.sh" ]]; then
  echo "Missing launcher script: $REPO_ROOT/scripts/start-mac-local.sh"
  exit 1
fi

mkdir -p "$INSTALL_DIR"
cat > "$TMP_DIR/launcher.applescript" <<APPLESCRIPT
set repoPath to "$REPO_ROOT"
tell application "Terminal"
  activate
  do script "cd " & quoted form of repoPath & " && exec ./scripts/start-mac-local.sh"
end tell
APPLESCRIPT

rm -rf "$APP_PATH"
osacompile -o "$APP_PATH" "$TMP_DIR/launcher.applescript"

if [[ -f "$REPO_ROOT/resources/icon.icns" ]]; then
  cp "$REPO_ROOT/resources/icon.icns" "$APP_PATH/Contents/Resources/applet.icns"
  /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1 || true
fi

/usr/libexec/PlistBuddy -c "Set :CFBundleName LTX Desktop Local" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1 || true
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName LTX Desktop Local" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1 || true
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.local.ltx-desktop-launcher" "$APP_PATH/Contents/Info.plist" >/dev/null 2>&1 || true

touch "$APP_PATH"
echo "Installed launcher: $APP_PATH"
echo "Double-click it to start LTX Desktop with project-local MLX settings."
