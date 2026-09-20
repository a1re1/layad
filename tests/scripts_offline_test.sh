#!/usr/bin/env bash
#
# Offline tests for scripts/setup.sh and scripts/smoke.sh.
#
# Nothing here downloads weights, builds Rust, touches the real venv, registers a
# service or runs the real model: the scripts are copied into a throwaway repo
# whose .venv/bin/python, target/release/layad, uv and cargo are stubs that
# record their arguments. The checks are about *what the scripts do*: the
# checkpoint they select, the failure they produce when it is missing, the
# locked build, and the pids the smoke test is allowed to signal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

failures=0
check() {
  if [[ "$2" != "$3" ]]; then
    printf 'FAIL %s: expected %q, got %q\n' "$1" "$3" "$2" >&2
    failures=$((failures + 1))
  else
    printf 'ok   %s\n' "$1" >&2
  fi
}

# ------------------------------------------------------------------ stub repo
REPO="$TMP/repo"
BIN="$TMP/bin"
mkdir -p "$REPO/scripts" "$REPO/python" "$REPO/.venv/bin" "$REPO/target/release" \
  "$REPO/.layad/model/multilingual" "$BIN"
cp "$ROOT/scripts/setup.sh" "$ROOT/scripts/smoke.sh" "$REPO/scripts/"
cp "$ROOT/python/requirements.in" "$ROOT/python/requirements.txt" "$REPO/python/"

PY_ARGS="$TMP/py.args"
UV_ARGS="$TMP/uv.args"
CARGO_ARGS="$TMP/cargo.args"
: >"$PY_ARGS"
: >"$UV_ARGS"
: >"$CARGO_ARGS"

export LAYAD_TEST_PY_ARGS="$PY_ARGS"
export LAYAD_TEST_UV_ARGS="$UV_ARGS"
export LAYAD_TEST_CARGO_ARGS="$CARGO_ARGS"

cat >"$REPO/.venv/bin/python" <<'PY'
#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then printf '3.11\n'; exit 0; fi
printf '%s\n' "$*" >>"$LAYAD_TEST_PY_ARGS"
[[ "${LAYAD_TEST_PY_FAIL:-}" == "1" ]] && exit 1
printf '{"ready": true, "protocol": 1, "pid": 1, "device": "cpu"}\n'
PY
chmod +x "$REPO/.venv/bin/python"

cat >"$BIN/uv" <<'UV'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$LAYAD_TEST_UV_ARGS"
if [[ "${1:-} ${2:-}" == "pip compile" ]]; then
  prev=""
  for a in "$@"; do
    if [[ "$prev" == "--output-file" ]]; then printf 'stub==1\n' >"$a"; fi
    prev="$a"
  done
fi
exit 0
UV
chmod +x "$BIN/uv"

cat >"$BIN/cargo" <<'CARGO'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$LAYAD_TEST_CARGO_ARGS"
if [[ "${LAYAD_TEST_LOCK_FAIL:-}" == "1" ]]; then
  for a in "$@"; do [[ "$a" == "--locked" ]] && { echo 'error: the lock file needs to be updated' >&2; exit 101; }; done
fi
exit 0
CARGO
chmod +x "$BIN/cargo"

# A daemon stub that cannot serve anything: every smoke run using it stops at
# "daemon exited before becoming ready" (or earlier), never at a prediction.
cat >"$REPO/target/release/layad" <<'LAYAD'
#!/usr/bin/env bash
echo 'stub daemon: refusing to run' >&2
exit 3
LAYAD
chmod +x "$REPO/target/release/layad"

export PATH="$BIN:$PATH"
export LAYAD_SKIP_DOWNLOAD=1

# --------------------------------------------------------- setup.sh regressions
setup_out="$(LAYAD_CHECKPOINT=multilingual LAYAD_DEVICE=mps bash "$REPO/scripts/setup.sh" 2>&1)"
check "setup succeeds with mps + multilingual (warm enabled)" "$?" 0
case "$(cat "$PY_ARGS")" in
  *"--subfolder multilingual"*) check "warmup passes --subfolder for a non-English checkpoint" ok ok ;;
  *) check "warmup passes --subfolder for a non-English checkpoint" bad ok ;;
esac
case "$setup_out" in
  *"--checkpoint multilingual"*) check "printed foreground command carries --checkpoint" ok ok ;;
  *) check "printed foreground command carries --checkpoint" bad ok ;;
esac

: >"$PY_ARGS"
LAYAD_DEVICE=mpsc bash "$REPO/scripts/setup.sh" >/dev/null 2>&1 && mpsc_status=0 || mpsc_status=$?
check "setup rejects the mpsc device typo" "$mpsc_status" 1

: >"$PY_ARGS"
missing_out="$(LAYAD_CHECKPOINT=typed-decisions bash "$REPO/scripts/setup.sh" 2>&1)" || true
case "$missing_out" in
  *"typed-decisions' is missing from"*) check "missing subfolder fails instead of warming English" ok ok ;;
  *) check "missing subfolder fails instead of warming English" bad ok ;;
esac
check "no worker was run when the subfolder was missing" "$(wc -l <"$PY_ARGS" | tr -d ' ')" 0

# the checkout's own requirements.txt must be present, so the platform argument
# only shows up when recompiling on request.
: >"$UV_ARGS"
LAYAD_RECOMPILE=1 LAYAD_SKIP_WARM=1 bash "$REPO/scripts/setup.sh" >/dev/null 2>&1 || true
platform="$(sed -n 's/.*--python-platform \([^ ]*\).*/\1/p' "$UV_ARGS" | head -1)"
case "$platform" in
  "" | *macosx* | *\ *) check "uv compiles for a uv target triple, not a sysconfig string" bad ok ;;
  *) check "uv compiles for a uv target triple, not a sysconfig string" ok ok ;;
esac

: >"$CARGO_ARGS"
out="$(LAYAD_TEST_LOCK_FAIL=1 bash "$REPO/scripts/setup.sh" 2>&1)" || true
check "a failing locked build fails setup" "$(grep -c 'build --release --locked' "$CARGO_ARGS")" 1
check "no unlocked retry after a locked build failure" "$(wc -l <"$CARGO_ARGS" | tr -d ' ')" 1
case "$out" in
  *"locked"*) check "the locked-build failure is visible" ok ok ;;
  *) check "the locked-build failure is visible" bad ok ;;
esac

# --------------------------------------------------------- smoke.sh regressions
smoke_fail() { # label, env-assignments..., expect-substring
  local label="$1" want="$3"
  local out
  out="$(eval "$2 bash \"$REPO/scripts/smoke.sh\"" 2>&1)" || true
  case "$out" in
    *"$want"*) check "$label" ok ok ;;
    *) check "$label" bad ok ;;
  esac
}

smoke_fail "smoke rejects an unknown checkpoint" "LAYAD_CHECKPOINT=bogus" "LAYAD_CHECKPOINT must be"
smoke_fail "smoke fails when the selected subfolder is missing" "LAYAD_CHECKPOINT=typed-decisions" "is missing from"

# A listener that answers 200 on every path must make the run refuse to start:
# otherwise a later /readyz could be somebody else's server.
PY_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "$PY_BIN" ]]; then
  printf 'ERROR: no python interpreter available for the offline script tests\n' >&2
  exit 1
fi
PORT=$((20000 + (RANDOM % 20000)))
"$PY_BIN" - "$PORT" <<'SRV' &
import http.server, sys

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('content-length', '2')
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, *args):
        pass

http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
SRV
OCCUPIED_PID=$!
for _ in $(seq 1 50); do
  curl -fsS --connect-timeout 1 --max-time 1 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.2
done
smoke_fail "smoke refuses an occupied port" "LAYAD_SMOKE_PORT=$PORT" "already serving"
kill "$OCCUPIED_PID" 2>/dev/null || true
wait "$OCCUPIED_PID" 2>/dev/null || true

# An unrelated process that looks like a worker for the same model must survive a
# smoke run, and its own log must not be overwritten by ours.
mkdir -p "$TMP/unrelated/python"
printf 'import time\ntime.sleep(120)\n' >"$TMP/unrelated/python/worker.py"
# Give it the command line the old pattern (`python/worker.py.*--model $MODEL`)
# would have matched, then confirm that it does match before relying on it.
"$PY_BIN" "$TMP/unrelated/python/worker.py" --model "$REPO/.layad/model" --device cpu &
UNRELATED_PID=$!
if pgrep -f "python/worker.py.*--model $REPO/.layad/model" >/dev/null 2>&1; then
  check "an unrelated same-model worker is visible to the old pgrep pattern" ok ok
else
  printf 'note: could not stage an unrelated worker process; skipping that check\n' >&2
fi
SMOKE_PORT=$((20000 + (RANDOM % 20000)))
LAYAD_SMOKE_PORT="$SMOKE_PORT" bash "$REPO/scripts/smoke.sh" >/dev/null 2>&1 || true
if kill -0 "$UNRELATED_PID" 2>/dev/null; then
  check "smoke leaves an unrelated same-model worker running" ok ok
else
  check "smoke leaves an unrelated same-model worker running" bad ok
fi
kill "$UNRELATED_PID" 2>/dev/null || true
wait "$UNRELATED_PID" 2>/dev/null || true

if [[ -e "$REPO/.layad/smoke.log" ]]; then
  check "smoke does not reuse a shared smoke.log" bad ok
else
  check "smoke does not reuse a shared smoke.log" ok ok
fi
if compgen -G "$REPO/.layad/smoke.*.log" >/dev/null; then
  check "smoke logs under a per-run unique name" ok ok
else
  check "smoke logs under a per-run unique name" bad ok
fi

if [[ "$failures" -ne 0 ]]; then
  printf '%d offline script test(s) failed\n' "$failures" >&2
  exit 1
fi
printf 'offline script tests passed\n' >&2
