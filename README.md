# layad

A local HTTP daemon that keeps **one** Python Laya decision model resident, so
repeated predictions do not pay model load cost. Rust owns the HTTP surface and
the worker process lifecycle; a persistent Python child owns the model and
reuses upstream `laya` inference code unchanged.

This is a small, local, single-user tool. It is deliberately *Jev-inspired*
(typed decisions over a local endpoint) but it is **not wire-compatible with
Jev**, and it makes no claims about benchmark performance or probability
calibration.

## What it does

```
curl                                              resident Python worker
  │  POST /v1/predict  ──────────────┐                     │
  └─ Axum (loopback only)             │  one NDJSON line   │
                                      ├── stdin ───────────▶
                                      ◀── stdout ──────────┘
                                    (protocol only, single child, unbuffered)
```

* One worker child for the life of the daemon, started eagerly at boot.
* Bounded NDJSON protocol on stdin/stdout; every request has an id and a reply
  must match it.
* Fail-fast: if the worker dies, replies out of order, or misses a deadline, the
daemon tears the worker down and shuts down so `launchd` can restart it cleanly.

## Requirements

* macOS (Linux mostly works; launchd integration is macOS-only)
* `uv` — <https://docs.astral.sh/uv/> (`brew install uv`)
* Rust toolchain with cargo — <https://rustup.rs>
* ~10 GB free disk for the first setup (PyTorch wheels plus checkpoint), and
  enough RAM for the chosen checkpoint (CPU-only inference on the English
  checkpoint works in a few GB).

## Setup

```bash
scripts/setup.sh
```

This is the only setup step. It is idempotent — running it again reuses
whatever already exists. It:

1. checks prerequisites and fails with an actionable message if one is missing;
2. creates an isolated Python 3.11 environment in `.venv` with `uv` (system
   Python is never touched, nothing is installed globally, `sudo` is never used);
3. installs the pinned dependencies from `python/requirements.txt`
   (compiled from `python/requirements.in` and committed, so installs are
   reproducible);
4. downloads the chosen Laya checkpoint into `.layad/model`;
5. loads the model once and warms the weights **and** the tokenizer/encoder
   caches into `.layad/hf`;
6. builds the release binary with `cargo build --release --locked` (locked only:
a lockfile problem is a visible failure, never a silent unlocked build).

Selection is by environment variable:

| Variable | Values | Default |
| --- | --- | --- |
| `LAYAD_CHECKPOINT` | `english`, `multilingual`, `typed-decisions` | `english` |
| `LAYAD_DEVICE` | `cpu`, `mps`, `cuda`, `auto` | `cpu` |
| `LAYAD_MODEL_REPO` | any HF repo id | `convaiinnovations/laya` |
| `LAYAD_RECOMPILE=1` | recompile `requirements.txt` from `.in` | off |
| `LAYAD_SKIP_DOWNLOAD=1` | reuse `.layad/model` | off |
| `LAYAD_SKIP_WARM=1` | skip the warmup run | off |

The English checkpoint lives at the repository root; `multilingual` and
`typed-decisions` live in subfolders and are fetched as subfolders. Only the
chosen checkpoint is downloaded where the repository layout allows it, but
upstream repository structure can force extras — if the download is far larger
than expected, that is why.

A non-English checkpoint is selected by subfolder, end to end:

* `scripts/setup.sh` warms the requested subfolder (`--subfolder`, e.g.
  `.layad/model/multilingual`) and **fails** when that subfolder is missing
  instead of silently warming the English root; `LAYAD_SKIP_DOWNLOAD=1` with a
  missing subfolder is an error, not a downgrade.
* The foreground command it prints includes `--checkpoint <name>`; the daemon
  turns that into the worker's `--subfolder` and fails closed if the subfolder
  does not exist (no silent English fallback).
* `scripts/smoke.sh` and `scripts/service.sh` take `LAYAD_CHECKPOINT` the same
  way; the checkpoint name *is* the subfolder, there is no separate override.

Setup's cache and checkpoint paths are fixed: `.layad/model` for the checkpoint
(with non-English checkpoints as subfolders) and `.layad/hf` for the Hugging
Face cache. `scripts/setup.sh` exports `HF_HOME=.layad/hf` while it warms, and
the foreground daemon defaults `HF_HOME` to that same `.layad/hf` directory
unless you set `HF_HOME` yourself — so a daemon started by hand reuses exactly
the cache the setup warmed.

### Devices

`cpu` is the default and the only device that is guaranteed to work. `mps`
is Apple Metal, `cuda` is NVIDIA, `auto` prefers CUDA, then MPS, then CPU.
**Explicitly requesting an unavailable device can fall back to CPU.** The
daemon always reports both the requested and the actual device (`/readyz`,
startup logs), so trust the reported one.

### Offline startup

After `scripts/setup.sh` the daemon is expected to start **with no network**.
The smoke test proves it by running with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. If startup then fails, the warm cache is incomplete —
re-run `LAYAD_RECOMPILE=1 scripts/setup.sh` with network access. The launchd
service sets both offline flags too.

## Running it

Foreground, which is all you need on any platform:

```bash
./target/release/layad \
  --bind 127.0.0.1:8787 \
  --python ./.venv/bin/python \
  --model ./.layad/model \
  --checkpoint english \
  --device cpu
```

Useful flags: `--checkpoint english|multilingual|typed-decisions` (selects the
checkpoint subfolder inside `--model`; the daemon fails closed when that
subfolder is missing), `--startup-timeout-ms`,
`--inference-timeout-ms`, `--max-concurrent`, `--max-questions`,
`--max-body-bytes`, `--log`, `--no-fail-fast`. Every flag also reads an
environment variable (`LAYAD_*`); see `layad --help`.

The daemon binds **loopback only** and refuses any other bind address: the API
is unauthenticated, so exposing it would be unsafe. Request bodies are never
logged.

## API

### `GET /healthz`

Liveness. `200 {"status":"ok","service":"layad","version":"...","pid":N}`.
It answers while the model is still loading, which is what keeps a supervisor
from restarting a slow-but-healthy process.

### `GET /readyz`

`200` only once the worker has loaded the model and finished warmup; otherwise
`503` with `{"ready":false,"reason":"loading"|"unavailable","message":...}`.
When ready:

```json
{
  "ready": true,
  "model": ".layad/model",
  "checkpoint": "english",
  "device": "cpu",
  "requested_device": "cpu",
  "laya_version": "0.3.4",
  "worker_pid": 4321,
  "worker_identity": "4321:...",
  "warmup": ["choice", "score", "noul"]
}
```

`worker_identity` changes if the worker is ever replaced; it being stable across
requests is the signal that the model stayed resident.

### `POST /v1/predict`

Request:

```json
{
  "state": "the customer is angry about a double charge",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What does the customer want?",
      "criteria": {"refund": "a refund", "cancel": "to cancel"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "somewhat urgent", "very urgent"]
    },
    "human": {
      "type": "noul",
      "instructions": "Does the customer want a human agent?"
    }
  }
}
```

* `state` — text, an object, or a list of `{"role","content"}` messages. See
  `examples/predict.json` for a list-shaped example.
* `questions` — a nonempty object of **named** questions. Names are arbitrary
  and are echoed back in `answers`.
* Three primitive types:
  * `choice` — needs `criteria`, either an object (name → description) or a
    nonempty list. Returns the chosen name plus the full distribution.
  * `score` — needs a nonempty ordered `criteria` list. Returns an index-based
    score and the legend it was measured against.
  * `noul` — needs no criteria. Returns a number-of-utterances-like count.

Response:

```json
{
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "refund",
      "probabilities": {"refund": 0.81, "cancel": 0.19},
      "confidence": 0.81
    },
    "urgency": {
      "type": "score",
      "score": 2,
      "legend": {"0": "not urgent", "1": "somewhat urgent", "2": "very urgent"}
    },
    "human": {"type": "noul", "noul": 3}
  },
  "usage": {"id": "...", "model": "...", "device": "cpu", "elapsed_ms": 412}
}
```

Answer payloads are passthrough from upstream Laya — anything the upstream
distribution returns is preserved rather than recomputed or rounded here.

### Errors

Every error is `{"error":{"code":"...","message":"..."}}` with a status code:

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_json` | body is not JSON |
| 400 | `invalid_request` | state/questions malformed, empty questions, bad criteria |
| 404 | `not_found` | unknown path (JSON, not an empty body) |
| 405 | `method_not_allowed` | wrong method (JSON, not an empty body) |
| 413 | `body_too_large` | body over `--max-body-bytes` |
| 429 | `overloaded` | too many questions or too many concurrent requests |
| 503 | `unavailable` | worker not ready, worker died, or request canceled |
| 504 | `timeout` | inference exceeded `--inference-timeout-ms` |

### Limits

Defaults: 1 MiB body, 32 questions, 4 concurrent requests, 8 MiB worker reply
cap, 600 s startup timeout, 60 s inference timeout, 10 s shutdown timeout.
Requests beyond a limit get `429`/`413` rather than being queued
unboundedly.

## Smoke test (real model)

```bash
scripts/smoke.sh
```

Starts the release daemon inside a deadline, waits for `/readyz`, sends two
predictions that cover `choice`, `score` and `noul`, verifies the answer shapes
and that both came from the same worker, and cleans up on every exit path. It
runs with offline flags, so passing it proves the warm cache is sufficient. It
is **not** part of CI, because it needs the real checkpoint; the mock-only CI
jobs never download weights and never prove that real inference works.

Safety properties of the smoke test:

* every `curl` is bounded (`--connect-timeout`, `--max-time`), so it cannot
  hang forever;
* before any prediction it reads `/healthz` and requires the reported daemon
  pid to be the process it launched — an occupied port would otherwise be a
  false pass;
* cleanup signals only the daemon it started and that daemon's own children,
  resolved by exact pid (never by a model-path `pgrep`/`pkill`), so a daemon
  someone else is already running with the same model is left running;
* a worker that outlives the daemon fails the run;
* its log is `.layad/smoke.$$.log`, unique per run, so two overlapping runs
  never overwrite each other's evidence.

`LAYAD_CHECKPOINT` selects the subfolder, and a missing subfolder is an error
rather than a silent English run.

## Optional macOS login service

Nothing is installed or loaded unless you ask for it explicitly:

```bash
scripts/service.sh install     # write and load the LaunchAgent
scripts/service.sh status      # installed? loaded?
scripts/service.sh uninstall   # unload and remove it
scripts/service.sh plist       # print the plist, write nothing
```

The service runs the release binary with absolute paths, writes logs to
`.layad/layad.log` and `.layad/layad.err.log`, and keeps the runtime paths
identical to the foreground ones (`.layad/model`, `.layad/hf`, default
`LAYAD_HOME_DIR=<repo>/.layad`). It passes `--checkpoint $LAYAD_CHECKPOINT`
(default `english`) to the daemon, so the plist selects the same checkpoint the
foreground command would. Values are
XML-escaped, so paths containing spaces or `&` are safe. `KeepAlive` restarts
the daemon only after an unsuccessful exit, which is what the fail-fast
behavior is for.

Ownership and safety:

* `LAYAD_LABEL` must be a valid label (dotted components of letters, digits,
  `_` and `-`); anything else is refused before it reaches a path or
  `launchctl`.
* A plist is treated as ours only when it carries our generated-file marker
  **and** names this worktree's runtime home, so a service installed by a
  sibling worktree (same generic marker, different home) is never overwritten,
  unloaded or removed.
* `install` refuses to take over a label that is already loaded without a
  layad-owned plist.
* `uninstall` reports failure honestly: if `launchctl unload` fails and the
  label is still loaded, the script exits non-zero instead of claiming success.

On non-macOS hosts this script refuses to run; use the foreground command
instead. The script tests stub `launchctl` and point `HOME` at a temporary
directory, so they never touch the real launchd.

## Logs and troubleshooting

* `.layad/warmup.log` — output of the setup warmup run.
* `.layad/layad.log`, `.layad/layad.err.log` — service stdout/stderr.
* `/readyz` says `"reason":"loading"` for a long time → first load is slow,
  especially on CPU; check `--startup-timeout-ms` and the worker log.
* Daemon exits shortly after start with the worker failing → run the worker
  directly to see the real error:
  `./.venv/bin/python python/worker.py --model .layad/model --device cpu --help`.
* Setup failed mid-download → rerun it; the HF cache resumes.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
.venv/bin/python -m unittest discover -s python/tests
bash -n scripts/setup.sh scripts/service.sh scripts/smoke.sh
bash tests/service_test.sh
bash tests/scripts_offline_test.sh
```

The Rust integration tests and the Python worker tests use a fake child
process, so they need no torch and no checkpoint. `tests/service_test.sh`
stubs `launchctl` and points `HOME` at a temporary directory, so it never
registers a real login service. `tests/scripts_offline_test.sh` covers the
setup/smoke argument rendering, missing-checkpoint failure, the locked build
requirement, service ownership/unload/label safety, an occupied port and the
survival of an unrelated same-model process — all offline, with fake binaries
and a stub `launchctl`.

## Limitations

* Single model, single process, single machine. No auth, loopback only.
* One inference at a time inside the worker; concurrency above that is
  admission control, not parallel inference.
* Fail-fast by design: failures are meant to be handled by a supervisor
  restarting the process, not by hot-swapping models in place.
* Upstream Laya behavior (prompt format, sampling, calibration) is inherited
  as-is; this project does not tune or validate it.
* Not Jev-compatible: request/response shapes are inspired by Jev's typed
  decisions but are their own thing.

## Cleanup

```bash
scripts/service.sh uninstall   # first, if the service was installed
rm -rf .venv .layad target     # removes env, model/caches/logs and binaries
```

## Model and license

Weights come from the upstream [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
repository and the Python `laya==0.3.4` package; their licenses and terms apply
to the model. This repository is MIT-licensed (see `LICENSE`).
