#!/usr/bin/env bash
#
# Real end-to-end smoke test: starts the release daemon against the project-local
# checkpoint, waits for readiness within a deadline, sends two predictions
# covering choice, score and noul, verifies the answer shapes and that both came
# from the same resident worker, and cleans up on every exit path.
#
# It runs the daemon with Hugging Face offline flags set, so a passing run proves
# the warm cache created by scripts/setup.sh is genuinely sufficient.
#
# Safety: it only ever signals the daemon it started and that daemon's own
# children (resolved by parent pid, never by a model-path pattern), so a daemon
# someone else is already running with the same model is left untouched.
#
# Environment overrides:
#   LAYAD_CHECKPOINT=english|multilingual|typed-decisions  (default english)
#   LAYAD_SMOKE_PORT  port to bind                   (default 8791)
#   LAYAD_SMOKE_DEADLINE  readiness deadline, seconds (default 900)
#   LAYAD_SMOKE_HTTP_TIMEOUT  per-request curl cap, seconds (default 300)
#   LAYAD_DEVICE      cpu|mps|cuda|auto              (default cpu)
#
# Requirements: scripts/setup.sh has been run (model + cache + release binary).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BINARY="$ROOT/target/release/layad"
PYTHON="$ROOT/.venv/bin/python"
MODEL="$ROOT/.layad/model"
LAYAD_HOME="$ROOT/.layad"
PORT="${LAYAD_SMOKE_PORT:-8791}"
READY_DEADLINE="${LAYAD_SMOKE_DEADLINE:-900}"
HTTP_TIMEOUT="${LAYAD_SMOKE_HTTP_TIMEOUT:-300}"
CHECKPOINT="${LAYAD_CHECKPOINT:-english}"

log() { printf '[smoke] %s\n' "$*" >&2; }
die() {
  printf '[smoke] ERROR: %s\n' "$*" >&2
  exit 1
}

case "$CHECKPOINT" in
  english | multilingual | typed-decisions) ;;
  *) die "LAYAD_CHECKPOINT must be english, multilingual or typed-decisions (got '$CHECKPOINT')" ;;
esac
# The selected checkpoint alone decides the subfolder: a non-English
# checkpoint is a subfolder of the model directory, English is its root.
SUBFOLDER=""
if [[ "$CHECKPOINT" != "english" ]]; then
  SUBFOLDER="$CHECKPOINT"
fi

# Every request is bounded: no curl in this script may hang forever.
curl_bounded() { curl --connect-timeout 5 --max-time "$HTTP_TIMEOUT" "$@"; }

[[ -x "$BINARY" ]] || die "$BINARY is missing; run scripts/setup.sh first"
[[ -x "$PYTHON" ]] || die "$PYTHON is missing; run scripts/setup.sh first"
[[ -d "$MODEL" ]] || die "$MODEL is missing; run scripts/setup.sh first"
command -v curl >/dev/null 2>&1 || die "curl is required"

# The selected checkpoint must exist as a subfolder; warming or serving the
# English root while the operator asked for another one is a silent downgrade.
if [[ -n "$SUBFOLDER" ]]; then
  [[ -d "$MODEL/$SUBFOLDER" ]] ||
    die "checkpoint '$SUBFOLDER' is missing from $MODEL; run scripts/setup.sh first"
fi

mkdir -p "$LAYAD_HOME"
# Unique per run, so two overlapping instances never overwrite each other's log.
LOG="$LAYAD_HOME/smoke.$$.log"
DAEMON_PID=""
WORKER_PIDS=()

# Exact pids only: the daemon we launched and that daemon's own children.
owned_pids() {
  local pid
  if [[ -n "$DAEMON_PID" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] && printf '%s\n' "$pid"
    done < <(pgrep -P "$DAEMON_PID" 2>/dev/null || true)
  fi
  local recorded
  for recorded in ${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}; do
    printf '%s\n' "$recorded"
  done
}

# Stop one exact pid: TERM, bounded wait, then KILL. Non-zero only when it is
# still alive afterwards.
stop_pid() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM "$pid" 2>/dev/null || true
  local i
  for i in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -KILL "$pid" 2>/dev/null || true
  sleep 0.2
  kill -0 "$pid" 2>/dev/null && return 1
  return 0
}

cleanup() {
  local status=$?
  local pids=()
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(owned_pids)

  if [[ -n "$DAEMON_PID" ]]; then
    log "stopping daemon $DAEMON_PID"
    kill -TERM "$DAEMON_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$DAEMON_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi

  # Every worker we started (or the daemon reported) must be gone with it.
  for pid in ${pids[@]+"${pids[@]}"}; do
    if ! stop_pid "$pid"; then
      log "ERROR: worker $pid survived the daemon"
      status=1
    fi
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

BASE="http://127.0.0.1:$PORT"
# Refuse to run against someone else's server: otherwise a later /readyz would
# be a false pass on an occupied port.
if curl_bounded -fsS --connect-timeout 2 --max-time 3 "$BASE/healthz" >/dev/null 2>&1; then
  die "something is already serving $BASE/healthz; choose another LAYAD_SMOKE_PORT"
fi

log "starting the daemon with Hugging Face offline flags (log: $LOG)"
HF_HOME="$LAYAD_HOME/hf" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "$BINARY" \
  --bind "127.0.0.1:$PORT" \
  --python "$PYTHON" \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --device "${LAYAD_DEVICE:-cpu}" \
  --log info >"$LOG" 2>&1 &
DAEMON_PID=$!

log "waiting up to ${READY_DEADLINE}s for /readyz"
ready=0
for _ in $(seq 1 "$((READY_DEADLINE * 2))"); do
  if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    tail -30 "$LOG" >&2 || true
    die "daemon exited before becoming ready"
  fi
  if curl_bounded -fsS --max-time 5 "$BASE/readyz" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
[[ "$ready" -eq 1 ]] || {
  tail -30 "$LOG" >&2 || true
  die "daemon was not ready within ${READY_DEADLINE}s"
}

# Prove the server on this port is the process we launched before trusting any
# readiness answer, then remember its worker pid for exact cleanup.
health_pid="$(curl_bounded -fsS "$BASE/healthz" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["pid"])' || true)"
[[ -n "$health_pid" ]] || die "/healthz did not report a pid"
[[ "$health_pid" == "$DAEMON_PID" ]] ||
  die "/healthz pid $health_pid is not the daemon we started ($DAEMON_PID)"

ready_json="$(curl_bounded -fsS "$BASE/readyz")"
identity_before="$(printf '%s' "$ready_json" | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["worker_identity"], d["device"], sep="|")')"
worker_pid="$(printf '%s' "$ready_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["worker_pid"])')"
WORKER_PIDS+=("$worker_pid")
if ! pgrep -P "$DAEMON_PID" 2>/dev/null | grep -qx "$worker_pid"; then
  die "worker pid $worker_pid is not a child of daemon $DAEMON_PID; refusing a false pass"
fi
log "ready: $identity_before (worker $worker_pid)"

request() {
  curl_bounded -fsS -X POST "$BASE/v1/predict" -H 'content-type: application/json' --data-binary "$1"
}

prediction() {
  cat <<JSON
{"state": $1,
 "questions": {
   "intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": {"refund": "a refund", "cancel": "to cancel", "other": "something else"}},
   "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
   "human": {"type": "noul", "instructions": "Does the customer want a human agent?"}
 }}
JSON
}

log "prediction 1 (state: string)"
first="$(request "$(prediction '"I was charged twice and nobody replied for a week, fix it today"')")"
log "prediction 2 (state: object)"
second="$(request "$(prediction '{"customer": "angry", "tier": "gold", "message": "this is unacceptable"}')")"

"$PYTHON" - "$first" "$second" <<'PY'
import json
import sys

first, second = (json.loads(raw) for raw in sys.argv[1:3])
for label, body in (("first", first), ("second", second)):
    answers = body["answers"]
    assert set(answers) == {"intent", "urgency", "human"}, (label, sorted(answers))
    choice = answers["intent"]
    assert choice["type"] == "choice" and isinstance(choice["choice"], str), (label, choice)
    assert isinstance(choice["probabilities"], dict) and choice["probabilities"], (label, choice)
    assert isinstance(choice["confidence"], (int, float)), (label, choice)
    score = answers["urgency"]
    assert score["type"] == "score" and isinstance(score["score"], (int, float)), (label, score)
    assert isinstance(score["legend"], dict), (label, score)
    noul = answers["human"]
    assert noul["type"] == "noul" and isinstance(noul["noul"], (int, float)), (label, noul)
    assert "usage" in body, (label, body)
print("answer shapes ok for choice, score and noul")
PY

identity_after="$(curl_bounded -fsS "$BASE/readyz" | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(d["worker_identity"], d["device"], sep="|")')"
if [[ "$identity_before" != "$identity_after" ]]; then
  die "the worker changed between requests ($identity_before -> $identity_after); the model was not resident"
fi
log "same resident worker for both predictions: $identity_after"
log "log kept at $LOG"
log "smoke passed"
