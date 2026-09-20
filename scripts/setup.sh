#!/usr/bin/env bash
#
# One-command local setup for layad.
#
# Creates an isolated Python 3.11 environment with pinned dependencies, copies
# the chosen Laya checkpoint into .layad/model, warms the model (weights *and*
# tokenizer/encoder assets) so the daemon can start offline, and builds the
# release binary. Repeated runs are safe: existing files are reused.
#
# It never touches system Python, never installs global packages, never uses
# sudo and never registers a login service.
#
# Environment overrides:
#   LAYAD_CHECKPOINT=english|multilingual|typed-decisions   (default english)
#   LAYAD_DEVICE=cpu|mps|cuda|auto                         (default cpu)
#   LAYAD_MODEL_REPO=convaiinnovations/laya
#   LAYAD_RECOMPILE=1    recompile python/requirements.txt from the .in file
#   LAYAD_SKIP_DOWNLOAD=1  skip the checkpoint download (already present)
#   LAYAD_SKIP_WARM=1    skip the warmup/verification run
#
# Layout (the same paths the foreground daemon and scripts/service.sh use):
#   .layad/model   checkpoint files; non-English checkpoints are subfolders
#   .layad/hf      Hugging Face cache this script warms and exports as HF_HOME
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${LAYAD_CHECKPOINT:-english}"
DEVICE="${LAYAD_DEVICE:-cpu}"
MODEL_REPO="${LAYAD_MODEL_REPO:-convaiinnovations/laya}"
PY_VERSION="3.11"
VENV="$ROOT/.venv"
LAYAD_HOME="$ROOT/.layad"
MODEL_DIR="$LAYAD_HOME/model"
HF_HOME="$LAYAD_HOME/hf"
REQUIREMENTS_IN="$ROOT/python/requirements.in"
REQUIREMENTS="$ROOT/python/requirements.txt"
WORKER="$ROOT/python/worker.py"

log() { printf '[setup] %s\n' "$*" >&2; }

die() {
  printf '[setup] ERROR: %s\n' "$*" >&2
  exit 1
}

case "$CHECKPOINT" in
  english | multilingual | typed-decisions) ;;
  *) die "LAYAD_CHECKPOINT must be english, multilingual or typed-decisions (got '$CHECKPOINT')" ;;
esac

case "$DEVICE" in
  cpu | mps | cuda | auto) ;;
  *) die "LAYAD_DEVICE must be cpu, mps, cuda or auto (got '$DEVICE')" ;;
esac

# ---------------------------------------------------------------- prerequisites
command -v uv >/dev/null 2>&1 || die "uv is required (install: brew install uv, or https://docs.astral.sh/uv/)"
if ! command -v cargo >/dev/null 2>&1; then
  die "cargo is required to build the daemon (install: https://rustup.rs)"
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  log "note: this project targets macOS; continuing on $(uname -s) anyway"
fi

# ------------------------------------------------------- isolated Python 3.11 venv
if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating isolated Python $PY_VERSION environment in .venv (uv fetches the interpreter if needed)"
  uv venv --python "$PY_VERSION" "$VENV"
else
  log "reusing existing .venv"
fi
VENV_PY="$VENV/bin/python"
[[ -x "$VENV_PY" ]] || die "$VENV_PY is missing after uv venv"

actual_version="$("$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$actual_version" == "$PY_VERSION" ]] || die ".venv runs Python $actual_version, expected $PY_VERSION"

# --------------------------------------------------------- pinned dependencies
if [[ -n "${LAYAD_RECOMPILE:-}" || ! -f "$REQUIREMENTS" ]]; then
  log "compiling pinned requirements for Python $PY_VERSION (torch wheels are large)"
  compile_args=(--python-version "$PY_VERSION" --generate-hashes --output-file "$REQUIREMENTS")
  # uv only accepts its own target-triple names. sysconfig reports a platform
  # string such as "macosx-14.0-arm64", which uv rejects; map the host to a
  # supported triple, and for an unknown host pass nothing so uv uses its own
  # native default.
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64 | Darwin/aarch64) compile_args+=(--python-platform aarch64-apple-darwin) ;;
    Darwin/x86_64) compile_args+=(--python-platform x86_64-apple-darwin) ;;
    Linux/x86_64) compile_args+=(--python-platform x86_64-unknown-linux-gnu) ;;
    Linux/aarch64 | Linux/arm64) compile_args+=(--python-platform aarch64-unknown-linux-gnu) ;;
    *) log "note: unknown host $(uname -s)/$(uname -m); using uv's native platform" ;;
  esac
  uv pip compile "$REQUIREMENTS_IN" "${compile_args[@]}" ||
    die "uv pip compile failed; re-run with network access or pin python/requirements.txt manually"
else
  log "reusing pinned python/requirements.txt (LAYAD_RECOMPILE=1 recompiles)"
fi

log "installing pinned dependencies into .venv"
VIRTUAL_ENV="$VENV" uv pip install --python "$VENV_PY" -r "$REQUIREMENTS"

# --------------------------------------------------- project-local model + cache
mkdir -p "$MODEL_DIR" "$HF_HOME"
export HF_HOME

if [[ -z "${LAYAD_SKIP_DOWNLOAD:-}" ]]; then
  log "downloading checkpoint '$CHECKPOINT' from $MODEL_REPO into .layad/model (first run only, ~GBs)"
  "$VENV_PY" - "$MODEL_REPO" "$MODEL_DIR" "$CHECKPOINT" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo, target, checkpoint = sys.argv[1], sys.argv[2], sys.argv[3]
# The repository bundles the English checkpoint at the root plus optional
# subfolders. Fetch only the chosen one so a multilingual setup does not pull
# every variant.
if checkpoint == "english":
    patterns = None
    ignore = ["multilingual/*", "typed-decisions/*"]
else:
    patterns = [f"{checkpoint}/*", "README.md"]
    ignore = None
path = snapshot_download(
    repo_id=repo,
    local_dir=target,
    allow_patterns=patterns,
    ignore_patterns=ignore,
)
print(f"downloaded to {path}")
PY
else
  log "LAYAD_SKIP_DOWNLOAD set: reusing .layad/model"
fi

[[ -d "$MODEL_DIR" ]] || die "model directory $MODEL_DIR is missing"

# A non-English checkpoint is a subfolder of the model directory. Refuse to go
# on when it is absent: warming and later serving the English root while the
# operator asked for multilingual is exactly the silent fallback to avoid.
if [[ "$CHECKPOINT" != "english" ]]; then
  [[ -d "$MODEL_DIR/$CHECKPOINT" ]] ||
    die "checkpoint '$CHECKPOINT' is missing from $MODEL_DIR (found: $(ls -1 "$MODEL_DIR" 2>/dev/null | tr '\n' ' ')); re-run without LAYAD_SKIP_DOWNLOAD to fetch it"
fi

# ------------------------------------------- warm weights AND tokenizer/encoder
if [[ -z "${LAYAD_SKIP_WARM:-}" ]]; then
  log "loading and warming the model (weights, tokenizer and encoder caches are written into .layad/hf)"
  warm_args=(--model "$MODEL_DIR" --device "$DEVICE")
  if [[ "$CHECKPOINT" != "english" ]]; then
    warm_args+=(--subfolder "$CHECKPOINT")
  fi
  warm_log="$LAYAD_HOME/warmup.log"
  if ! printf '{"id": null, "method": "shutdown"}\n' |
    "$VENV_PY" "$WORKER" "${warm_args[@]}" >"$warm_log" 2>&1; then
    log "warmup failed; last lines of $warm_log:"
    tail -20 "$warm_log" >&2 || true
    die "the worker could not load the checkpoint (see above)"
  fi
  if ! grep -q '"ready": true' "$warm_log"; then
    log "warmup log:"
    tail -20 "$warm_log" >&2 || true
    die "the worker did not report readiness after loading the checkpoint"
  fi
  log "warmup ok: $(grep -o '"device": "[a-z]*"' "$warm_log" | head -1)"
  log "a full worker log is kept at .layad/warmup.log"
else
  log "LAYAD_SKIP_WARM set: skipping the warmup run"
fi

# ------------------------------------------------------------------ rust build
log "building the release daemon (locked)"
# Locked only: a silent unlocked retry would build a binary that does not match
# Cargo.lock, so a lockfile problem is a visible failure instead.
if ! cargo build --release --locked; then
  die "cargo build --release --locked failed; fix the tree/Cargo.lock instead of building unlocked"
fi

log "setup complete"
log "paths: .layad/model (checkpoint), .layad/hf (cache, exported as HF_HOME=$HF_HOME)"
log "foreground run:"
log "  ./target/release/layad --model .layad/model --checkpoint $CHECKPOINT --device $DEVICE"
if [[ "$CHECKPOINT" != "english" ]]; then
  log "worker directly: ./.venv/bin/python python/worker.py --model .layad/model --subfolder $CHECKPOINT"
else
  log "worker directly: ./.venv/bin/python python/worker.py --model .layad/model"
fi
