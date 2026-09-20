#!/usr/bin/env bash
#
# Service-script tests. These use a stub launchctl and a temporary HOME (plus a
# temporary plist directory), so they never register anything with the real
# login session and never touch the operator's LaunchAgents.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="$ROOT/scripts/service.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

STUB="$TMP/launchctl"
CALLS="$TMP/launchctl.calls"
# A stateful stub: `load`/`unload` keep a set of loaded labels in
# $LAYAD_TEST_STATE and `list` answers from it, so "is it still loaded?" is a
# real question in the tests. LAYAD_TEST_FAIL_UNLOAD=1 makes unload fail while
# leaving the label loaded, which is the case that must never be reported as
# success.
cat >"$STUB" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$LAYAD_TEST_CALLS"
cmd="${1:-}"
label="${2:-}"
# `load`/`unload` are given a plist path while `list` is given the label; keep
# the state file in terms of labels so "is it still loaded?" is meaningful.
if [[ "$label" == */* ]]; then
  label="$(basename "$label")"
  label="${label%.plist}"
fi
state="${LAYAD_TEST_STATE:-}"
case "$cmd" in
  load)
    [[ -n "$state" ]] && printf '%s\n' "$label" >>"$state"
    [[ "${LAYAD_TEST_FAIL_LOAD:-}" == "1" ]] && exit 1
    exit 0 ;;
  unload)
    [[ "${LAYAD_TEST_FAIL_UNLOAD:-}" == "1" ]] && exit 1
    if [[ -n "$state" && -f "$state" ]]; then
      grep -vx "$label" "$state" >"$state.new" || true
      mv "$state.new" "$state"
    fi
    exit 0 ;;
  list)
    if [[ -n "$state" && -f "$state" ]] && grep -qx "$label" "$state"; then exit 0; fi
    exit 1 ;;
  *) exit 0 ;;
esac
STUB
chmod +x "$STUB"
: >"$CALLS"

FAKE_HOME="$TMP/home with spaces & ampersand"
mkdir -p "$FAKE_HOME/Library/LaunchAgents"

export LAYAD_TEST_CALLS="$CALLS"
export LAYAD_LAUNCHCTL="$STUB"
export HOME="$FAKE_HOME"
export LAYAD_LABEL="dev.layad.test"
export LAYAD_BIND="127.0.0.1:9999"
# A runtime home whose path contains a space and an ampersand, so escaping is
# genuinely exercised rather than assumed.
export LAYAD_HOME_DIR="$TMP/runtime dir & logs"

failures=0
check() {
  if [[ "$2" != "$3" ]]; then
    printf 'FAIL %s: expected %q, got %q\n' "$1" "$3" "$2" >&2
    failures=$((failures + 1))
  else
    printf 'ok   %s\n' "$1" >&2
  fi
}

# ---------------------------------------------------------------- plist rendering
plist="$(bash "$SERVICE" plist)"
case "$plist" in
  *"runtime dir &amp; logs"*) check "ampersand and spaces are escaped" ok ok ;;
  *) check "ampersand and spaces are escaped" bad ok ;;
esac
case "$plist" in
  *"&"*)
    # No raw ampersand may survive outside an entity.
    if printf '%s' "$plist" | grep -q '&[^a-z#]' || printf '%s' "$plist" | grep -q '&$'; then
      check "no raw ampersands in the plist" bad ok
    else
      check "no raw ampersands in the plist" ok ok
    fi
    ;;
  *) check "no raw ampersands in the plist" ok ok ;;
esac
case "$plist" in
  *"<string>$ROOT/.venv/bin/python</string>"*)
    check "python path is absolute" ok ok
    ;;
  *) check "python path is absolute" bad ok ;;
esac
case "$plist" in
  *"<string>$ROOT/target/release/layad</string>"*) check "binary path is absolute" ok ok ;;
  *) check "binary path is absolute" bad ok ;;
esac
case "$plist" in
  *"<key>Label</key>"*"<string>dev.layad.test</string>"*) check "label is rendered" ok ok ;;
  *) check "label is rendered" bad ok ;;
esac
if command -v plutil >/dev/null 2>&1; then
  printf '%s\n' "$plist" >"$TMP/render.plist"
  if plutil -lint "$TMP/render.plist" >/dev/null 2>&1; then
    check "rendered plist is valid XML" ok ok
  else
    check "rendered plist is valid XML" bad ok
  fi
fi

# ---------------------------------------------------------------------- install
# The real repo may or may not have a built binary; the stubbed-worktree copy
# below runs install against fixture executables and insists on success, so this
# call only exercises the refusal paths. Its outcome is checked in both
# directions rather than noted away.
install_rc=0
bash "$SERVICE" install >"$TMP/install.raw" 2>&1 || install_rc=$?
if [[ "$install_rc" == 0 ]]; then
  check "install either succeeds or refuses because the repo has no build" ok ok
elif grep -q "is missing; run scripts/setup.sh first" "$TMP/install.raw"; then
  check "install either succeeds or refuses because the repo has no build" ok ok
else
  check "install either succeeds or refuses because the repo has no build (rc=$install_rc)" bad ok
  cat "$TMP/install.raw" >&2
fi

# ----------------------------------------------------- unmanaged plist is safe
UNMANAGED="$FAKE_HOME/Library/LaunchAgents/dev.layad.unmanaged.plist"
printf 'not ours\n' >"$UNMANAGED"
if LAYAD_LABEL=dev.layad.unmanaged bash "$SERVICE" uninstall >/dev/null 2>&1; then
  check "unmanaged plist is not removed" bad ok
else
  check "unmanaged plist is not removed" ok ok
fi
check "unmanaged plist survives" "$(cat "$UNMANAGED")" "not ours"

# ---------------------------------------------------------------------- status
# status is only meaningful when the install above produced a plist; in the
# stubbed worktree below it is asserted unconditionally.
if [[ -f "$FAKE_HOME/Library/LaunchAgents/dev.layad.test.plist" ]]; then
  if bash "$SERVICE" status >/dev/null 2>&1; then
    check "status succeeds while installed" ok ok
  else
    check "status succeeds while installed" bad ok
  fi
  # -------------------------------------------------------------------- uninstall
  if bash "$SERVICE" uninstall >/dev/null 2>&1; then
    check "uninstall succeeds for our own plist" ok ok
  else
    check "uninstall succeeds for our own plist" bad ok
  fi
  if [[ -e "$FAKE_HOME/Library/LaunchAgents/dev.layad.test.plist" ]]; then
    check "uninstall removes our plist" bad ok
  else
    check "uninstall removes our plist" ok ok
  fi
else
  check "uninstall path is exercised in the stubbed worktree below" ok ok
fi

if grep -q "unload" "$CALLS"; then
  check "stub launchctl was used" ok ok
else
  check "stub launchctl was used" bad ok
fi

# ------------------------------------------------ label and checkpoint safety
if LAYAD_LABEL='../escape' bash "$SERVICE" plist >/dev/null 2>&1; then
  check "a label that escapes the plist directory is refused" bad ok
else
  check "a label that escapes the plist directory is refused" ok ok
fi
if LAYAD_LABEL='dev.layad.daemon/../../x' bash "$SERVICE" plist >/dev/null 2>&1; then
  check "a label containing slashes is refused" bad ok
else
  check "a label containing slashes is refused" ok ok
fi
if LAYAD_CHECKPOINT=bogus bash "$SERVICE" plist >/dev/null 2>&1; then
  check "an unknown checkpoint is refused" bad ok
else
  check "an unknown checkpoint is refused" ok ok
fi
multilingual_plist="$(LAYAD_CHECKPOINT=multilingual bash "$SERVICE" plist 2>/dev/null || true)"
case "$multilingual_plist" in
  *"<string>--checkpoint</string>"*"<string>multilingual</string>"*)
    check "the plist passes --checkpoint multilingual" ok ok ;;
  *) check "the plist passes --checkpoint multilingual" bad ok ;;
esac
case "$multilingual_plist" in
  *"<string>english</string>"*)
    check "a non-English plist does not quietly say english" bad ok ;;
  *) check "a non-English plist does not quietly say english" ok ok ;;
esac

# ------------------- install / ownership / unload, in a stubbed worktree copy
# service.sh derives every path from its own location, so a copy inside a
# throwaway repo lets install run with stub python and binary executables. No
# real launchd and no real binary are involved.
STUB_REPO="$TMP/stub repo"
mkdir -p "$STUB_REPO/scripts" "$STUB_REPO/.venv/bin" "$STUB_REPO/target/release"
cp "$SERVICE" "$STUB_REPO/scripts/service.sh"
printf '#!/usr/bin/env bash\nexit 0\n' >"$STUB_REPO/.venv/bin/python"
printf '#!/usr/bin/env bash\nexit 0\n' >"$STUB_REPO/target/release/layad"
chmod +x "$STUB_REPO/.venv/bin/python" "$STUB_REPO/target/release/layad"
STUB_SERVICE="$STUB_REPO/scripts/service.sh"

OUR_HOME="$TMP/stub runtime"
OTHER_HOME="$TMP/other worktree runtime"
OUR_PLIST="$FAKE_HOME/Library/LaunchAgents/dev.layad.test.plist"
export LAYAD_TEST_STATE="$TMP/loaded.labels"
: >"$LAYAD_TEST_STATE"
export LAYAD_HOME_DIR="$OUR_HOME"

# The install must genuinely succeed here: a silent "install reported failure
# (missing built binary?)" note is exactly the soft pass this test had before.
if bash "$STUB_SERVICE" install >"$TMP/install.out" 2>&1; then
  check "install succeeds against the fixture binary and python" ok ok
else
  check "install succeeds against the fixture binary and python" bad ok
  cat "$TMP/install.out" >&2
fi
install_out="$(cat "$TMP/install.out")"
if [[ -f "$OUR_PLIST" ]]; then
  check "install writes our plist" ok ok
else
  check "install writes our plist" bad ok
fi
case "$install_out" in
  *"loaded"*) check "install loads the label" ok ok ;;
  *) check "install loads the label (output: $install_out)" bad ok ;;
esac
if bash "$STUB_SERVICE" status >/dev/null 2>&1; then
  check "status succeeds for our own plist" ok ok
else
  check "status succeeds for our own plist" bad ok
fi

# The same generic marker, but a different runtime home: another worktree's
# service. It must never be claimed, overwritten or removed.
if LAYAD_HOME_DIR="$OTHER_HOME" bash "$STUB_SERVICE" status >/dev/null 2>&1; then
  check "another worktree's plist is not claimed" bad ok
else
  check "another worktree's plist is not claimed" ok ok
fi
if LAYAD_HOME_DIR="$OTHER_HOME" bash "$STUB_SERVICE" uninstall >/dev/null 2>&1; then
  check "another worktree's plist is not removed" bad ok
else
  check "another worktree's plist is not removed" ok ok
fi
if [[ -f "$OUR_PLIST" ]]; then
  check "another worktree's plist survives an uninstall attempt" ok ok
else
  check "another worktree's plist survives an uninstall attempt" bad ok
fi
before_foreign="$(cat "$OUR_PLIST")"
if LAYAD_HOME_DIR="$OTHER_HOME" bash "$STUB_SERVICE" install >/dev/null 2>&1; then
  check "install refuses to overwrite another worktree's plist" bad ok
else
  check "install refuses to overwrite another worktree's plist" ok ok
fi
check "the foreign plist is byte-identical after the refused install" "$(cat "$OUR_PLIST")" "$before_foreign"

# A label that is loaded while no layad plist exists is somebody else's service.
rm -f "$OUR_PLIST"
printf '%s\n' "dev.layad.test" >"$LAYAD_TEST_STATE"
if bash "$STUB_SERVICE" install >/dev/null 2>&1; then
  check "install refuses a loaded label with no layad plist" bad ok
else
  check "install refuses a loaded label with no layad plist" ok ok
fi

# Unload failure that leaves the label loaded must be visible, and the plist must
# not be deleted out from under a still-loaded service.
: >"$LAYAD_TEST_STATE"
if bash "$STUB_SERVICE" install >/dev/null 2>&1; then
  check "install succeeds before the unload-failure checks" ok ok
else
  check "install succeeds before the unload-failure checks" bad ok
fi
# The stub launchctl must actually report the label as loaded: a stateful stub
# plus fixture binaries means there is no environment in which this is skipped.
if grep -qx "dev.layad.test" "$LAYAD_TEST_STATE"; then
  check "the stubbed launchctl reports the installed label as loaded" ok ok
else
  check "the stubbed launchctl reports the installed label as loaded" bad ok
fi
if [[ -f "$OUR_PLIST" ]]; then
  if LAYAD_TEST_FAIL_UNLOAD=1 bash "$STUB_SERVICE" uninstall >/dev/null 2>&1; then
    check "a failed unload that leaves the service loaded is an error" bad ok
  else
    check "a failed unload that leaves the service loaded is an error" ok ok
  fi
  if [[ -f "$OUR_PLIST" ]]; then
    check "the plist survives a failed unload" ok ok
  else
    check "the plist survives a failed unload" bad ok
  fi
  if LAYAD_TEST_FAIL_UNLOAD=1 bash "$STUB_SERVICE" uninstall 2>&1 | grep -q "removed"; then
    check "a failed unload is never reported as success" bad ok
  else
    check "a failed unload is never reported as success" ok ok
  fi
else
  check "the installed plist exists for the unload-failure checks" bad ok
fi

# A clean uninstall removes the plist and unloads the label.
if bash "$STUB_SERVICE" uninstall >/dev/null 2>&1; then
  check "a clean uninstall succeeds" ok ok
else
  check "a clean uninstall succeeds" bad ok
fi
if [[ -e "$OUR_PLIST" ]]; then
  check "a clean uninstall removes the plist" bad ok
else
  check "a clean uninstall removes the plist" ok ok
fi
if grep -qx "dev.layad.test" "$LAYAD_TEST_STATE"; then
  check "a clean uninstall unloads the label" bad ok
else
  check "a clean uninstall unloads the label" ok ok
fi

if [[ "$failures" -ne 0 ]]; then
  printf '%d service test(s) failed\n' "$failures" >&2
  exit 1
fi
printf 'service tests passed\n' >&2
