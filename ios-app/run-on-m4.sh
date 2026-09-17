#!/bin/sh
# Run the MoFFT benchmark on the paired M4 iPad entirely from the terminal.
# PREREQ (one-time, GUI, unavoidable for Apple dev signing):
#   Xcode -> Settings -> Accounts -> select your Apple ID -> sign in (2FA)
#   -> Manage Certificates -> + -> Apple Development.
#   Verify with:  security find-identity -p codesigning -v   (must list >=1)
# After that, this script is fully terminal + repeatable for re-runs/debug.
set -e
cd "$(dirname "$0")/.."

: "${DEV:?set DEV to the device identifier from 'xcrun devicectl list devices'}"
: "${TEAM:?set TEAM to your Apple Developer team identifier}"
BID="${BID:-org.mofft.mofftbench}"

echo "[1/4] configure the iOS Xcode project"
cmake -S ios-app -B ios-app/build -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
  -DMOFFT_IOS_BUNDLE_IDENTIFIER="$BID"

echo "[2/4] build + auto-sign the app"
xcodebuild -project ios-app/build/mofftbench_ios.xcodeproj -target mofftbench \
  -sdk iphoneos -configuration Release build \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic -allowProvisioningUpdates \
  2>&1 | tail -6

APP=ios-app/build/Release-iphoneos/mofftbench.app
[ -d "$APP" ] || { echo "ERROR: app bundle not found: $APP"; exit 1; }
echo "app: $APP"

echo "[3/4] install on iPad"
xcrun devicectl device install app --device "$DEV" "$APP" 2>&1 | tail -3

echo "[4/4] launch benchmark on iPad and capture GFLOPS records to stdout"
if [ "$#" -eq 0 ]; then
  set -- --full-dry-run --batches 15
fi
xcrun devicectl device process launch --device "$DEV" --console -- \
  "$BID" "$@"
