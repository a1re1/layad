#!/usr/bin/env python3
"""Resident Laya worker for the layad daemon.

The daemon starts this process once, speaks newline-delimited JSON to it and
never starts another: the model stays loaded in memory between requests.

Protocol (one JSON object per line):

    daemon -> worker   {"id": "1", "method": "predict", "params": {"state": ..., "questions": {...}}}
    daemon -> worker   {"id": null, "method": "shutdown"}
    worker -> daemon   {"ready": true, "protocol": 1, "pid": ..., "device": ..., ...}
    worker -> daemon   {"id": "1", "ok": true, "result": {...}}
    worker -> daemon   {"id": "1", "ok": false, "error": {"code": "invalid_request", "message": "..."}}

Rules this file keeps:

* stdout carries protocol frames *only*. Anything the model stack (MLX Metal,
  PyTorch) prints — including C-level writes to fd 1 — is redirected to stderr
  before the library is imported, so a stray ``print`` can never corrupt the
  stream.
* A malformed frame or a dead-end inference error is fatal: the process exits
  non-zero and the daemon fails closed rather than answering with state it does
  not trust.
* Request-level problems (bad question structure, upstream validation errors)
  are reported with the request id so the worker stays usable.

Inference is delegated to whichever backend is selected (``--backend``, default
``mlx``) and nothing here reimplements the model:

* ``mlx`` (default): the native Apple MLX port ``laya_mlx`` (0.2.0), whose
  checkpoint is the ``aac6fef/laya-mlx`` repository. Apple silicon only.
* ``torch``: upstream ``laya`` (0.3.4) on PyTorch/transformers, the only
  backend that runs off Apple silicon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

PROTOCOL_VERSION = 1

# --------------------------------------------------------------- memory budget
#
# A resident MLX worker is memory-hungry for two reasons the inference stack
# does not cap by itself:
#
# * the Metal *buffer cache*: every intermediate MLX allocates is kept for
#   reuse and is never freed until memory pressure forces it, so a long-lived
#   worker grows without bound under a stream of differently-shaped batches;
# * the *wired* allocation pool, which lets the stack claim a growing share of
#   the machine's unified memory instead of staying inside one request's
#   working set.
#
# Left alone this reaches tens of gigabytes (reproduced here: a worker driven
# with classifier-shaped batches settled at a 12-13 GB Metal footprint, and a
# live daemon was seen at 49 GB). The budget below keeps the resident process
# inside a few gigabytes by capping both, and an operator can retune or
# disable it:
#
#   LAYAD_MEMORY_MB=3072   raise the cap
#   LAYAD_MEMORY_MB=0      leave the stack's own defaults alone
#
# The budget is a ceiling for the *whole resident process*, not just the GPU
# cache: the interpreter, the imported stack and the loaded checkpoint all live
# in the same unified memory as the cache, so their measured usage is
# subtracted from the budget before the cache gets its share.
MEMORY_BUDGET_ENV = "LAYAD_MEMORY_MB"
DEFAULT_MEMORY_BUDGET_MB = 2048
# Cache the worker keeps even when the budget is tiny. A smaller cache makes the
# allocator hand buffers back to the system between batches, which is slower but
# never incorrect, so this is a floor rather than a target.
MIN_CACHE_MB = 128
MLX_MIN_CACHE_BYTES = MIN_CACHE_MB * 1024 * 1024
# Interpreter, imported stack and allocator slack, subtracted from the budget
# when the inference stack can report live accelerator usage. It covers what
# MLX's counters leave out (see the README's memory section).
PROCESS_OVERHEAD_MB = 384

# Frames larger than this are refused instead of parsed (the daemon also caps
# bodies, this is defence in depth for a directly-driven worker).
MAX_INPUT_BYTES = 4 * 1024 * 1024

QUESTION_TYPES = ("choice", "score", "noul")

WARMUP_STATE = (
    "Customer: I was charged twice for my subscription and nobody has replied "
    "to my emails for a week. I want this fixed today."
)

WARMUP_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "a refund for the duplicate charge",
            "cancel": "to cancel the subscription",
            "other": "something else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "somewhat urgent", "very urgent"],
    },
    "human": {
        "type": "noul",
        "instructions": "Does the customer want to speak to a human agent?",
    },
}


# Inference stacks this worker can drive. ``mlx`` is the Apple silicon default;
# ``torch`` keeps the upstream path for hosts without MLX.
BACKENDS = ("mlx", "torch")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resident Laya worker for layad")
    parser.add_argument("--model", default=os.environ.get("LAYAD_MODEL", "aac6fef/laya-mlx"))
    parser.add_argument("--backend", default=os.environ.get("LAYAD_BACKEND", "mlx"))
    # "auto" means "let the stack choose" (MLX: GPU then CPU; PyTorch: CUDA,
    # MPS, then CPU), which is what both libraries treat as the default.
    parser.add_argument("--device", default=os.environ.get("LAYAD_DEVICE", "auto"))
    parser.add_argument("--subfolder", default=None)
    # Ceiling for the whole resident process, in MiB; 0 disables the cap.
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=_memory_budget_default(),
        help=(f"memory ceiling for the resident worker in MiB (env {MEMORY_BUDGET_ENV}); 0 disables it"),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="load the model but skip the warmup forward pass (debugging only)",
    )
    args = parser.parse_args(argv)
    # Validated here rather than through argparse's `choices` so a bad
    # LAYAD_BACKEND fails the same way a bad --backend does.
    if args.backend not in BACKENDS:
        parser.error(f"--backend must be one of {', '.join(BACKENDS)} (got {args.backend!r})")
    return args


def isolate_protocol_stream():
    """Give the protocol a private stream and point fd 1 at stderr.

    After this call every write to ``sys.stdout`` or to fd 1 — including from
    PyTorch's C++ layers — lands on stderr, while protocol frames go to the
    duplicate of the original stdout that only this function holds.
    """
    try:
        protocol_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(stderr_fd, 1)
        os.close(stderr_fd)
        stream = os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n", buffering=1)
    except OSError:
        # Without dup support we cannot promise a clean stream; say so loudly but
        # keep running (the daemon restarts a worker whose frames do not parse
        # rather than mis-reading them).
        sys.stderr.write("worker: cannot isolate stdout, protocol frames may interleave\n")
        sys.stderr.flush()
        return sys.stdout
    sys.stdout = sys.stderr
    return stream


def log(message: str) -> None:
    sys.stderr.write(f"[worker] {message}\n")
    sys.stderr.flush()


def _memory_budget_mb(raw: str | int) -> int:
    """Parse ``LAYAD_MEMORY_MB``: a positive integer, ``0`` to disable, garbage is fatal.

    A malformed value is refused instead of ignored: silently running without a
    cap is exactly the behaviour the cap exists to prevent.
    """
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise SystemExit(f"{MEMORY_BUDGET_ENV} must be an integer number of MiB (got {raw!r})")
    if value < 0:
        raise SystemExit(f"{MEMORY_BUDGET_ENV} must not be negative (got {raw!r})")
    return value


def _memory_budget_default() -> int:
    """Budget from the environment, for ``--memory-mb``'s default.

    An unset or empty variable means the built-in default rather than an error:
    "no opinion" and "not a number" are different things.
    """
    raw = os.environ.get(MEMORY_BUDGET_ENV, "")
    return _memory_budget_mb(raw) if raw.strip() else DEFAULT_MEMORY_BUDGET_MB


def _reported_accelerator_bytes(mx, device) -> int | None:
    """Live accelerator bytes MLX is holding, or None when it cannot report.

    ``get_active_memory`` is MLX-only. Under the torch backend the equivalent
    accounting is device-specific, so the budget falls back to subtracting
    nothing (see :func:`apply_memory_budget`).
    """
    getter = getattr(mx, "get_active_memory", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:  # pragma: no cover - a stack that raises here is unusual
        return None


def apply_memory_budget(args: argparse.Namespace) -> None:
    """Cap the resident process's memory after the checkpoint is loaded.

    Called once the weights are in memory (so their real cost is measured
    rather than guessed) and before the warmup pass.
    """
    if args.backend != "mlx":
        # Upstream laya on PyTorch owns its allocator; layad does not reach into
        # it. Say so once instead of pretending the budget applies.
        log(f"memory budget {args.memory_mb} MiB applies to the MLX backend only; torch is uncapped")
        return

    if args.memory_mb <= 0:
        log(f"{MEMORY_BUDGET_ENV}=0: leaving the MLX allocator at its own defaults")
        return

    import mlx.core as mx  # noqa: PLC0415 - imported after stdout is isolated on purpose

    budget = args.memory_mb * 1024 * 1024
    overhead = PROCESS_OVERHEAD_MB * 1024 * 1024
    measured = _reported_accelerator_bytes(mx, args.device)
    if measured is None:
        held = 0
        source = "unknown (stack reports no active-memory counter)"
    else:
        held = measured
        source = f"{held / 1024 / 1024:.0f} MiB reported by the stack"

    cache_limit = max(MLX_MIN_CACHE_BYTES, budget - overhead - held)
    wired_limit = max(held + MLX_MIN_CACHE_BYTES, budget - overhead)

    set_wired = getattr(mx, "set_wired_limit", None)
    if callable(set_wired):
        try:
            set_wired(wired_limit)
        except Exception as err:  # pragma: no cover - platform dependent
            log(f"could not set the wired-memory limit ({type(err).__name__}: {err})")
    if cache_limit + overhead + held > budget:
        effective = (cache_limit + overhead + held) / 1024 / 1024
        log(
            f"{MEMORY_BUDGET_ENV}={args.memory_mb} MiB is below the resident floor; "
            f"raising it to about {effective:.0f} MiB "
            f"(held {source})"
        )
    mx.set_cache_limit(cache_limit)
    clear = getattr(mx, "clear_cache", None)
    if callable(clear):
        # Return whatever is cached right now (the load and any pre-warm passes
        # left buffers behind), so the worker starts its life under the cap.
        clear()
    log(
        f"memory budget {args.memory_mb} MiB: MLX buffer cache capped at "
        f"{cache_limit / 1024 / 1024:.0f} MiB, wired limit "
        f"{wired_limit / 1024 / 1024:.0f} MiB (held {source})"
    )


def json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


class Worker:
    def __init__(self, stream, args: argparse.Namespace) -> None:
        self.stream = stream
        self.args = args

    def emit(self, frame: dict) -> None:
        line = json.dumps(frame, ensure_ascii=False, default=json_default)
        self.stream.write(line + "\n")
        self.stream.flush()

    def fatal(self, code: str, message: str, request_id=None) -> None:
        self.emit(
            {
                "id": request_id,
                "ok": False,
                "fatal": True,
                "error": {"code": code, "message": message},
            }
        )

    def reject(self, request_id, code: str, message: str) -> None:
        self.emit({"id": request_id, "ok": False, "error": {"code": code, "message": message}})

    def load(self):
        started = time.monotonic()
        backend = self.args.backend
        log(f"importing {backend} backend (model={self.args.model!r} device={self.args.device!r})")
        if backend == "mlx":
            # Native Apple MLX. It spells "let the library choose" as None and
            # validates the rest against gpu/metal/cpu, so "auto" is translated
            # here exactly as it is for upstream laya.
            import laya_mlx  # noqa: PLC0415 - imported after stdout is isolated on purpose

            version = getattr(laya_mlx, "__version__", "unknown")
            log(f"laya_mlx {version}; loading checkpoint")
            agent = laya_mlx.load(
                self.args.model,
                device=None if self.args.device == "auto" else self.args.device,
                subfolder=self.args.subfolder,
            )
        else:
            import laya  # noqa: PLC0415 - imported after stdout is isolated on purpose

            version = getattr(laya, "__version__", "unknown")
            log(f"laya {version}; loading checkpoint")
            agent = laya.load(
                self.args.model,
                device=None if self.args.device == "auto" else self.args.device,
                subfolder=self.args.subfolder,
            )
        log(f"checkpoint loaded in {time.monotonic() - started:.1f}s")
        return agent

    def warmup(self, agent) -> list[str]:
        if self.args.no_warmup:
            return []
        started = time.monotonic()
        # One forward pass touching all three primitives, so the encoder and the
        # typed heads are compiled/warmed before the first real request.
        agent.predict(WARMUP_STATE, WARMUP_QUESTIONS)
        log(f"warmup forward pass finished in {time.monotonic() - started:.1f}s")
        return [WARMUP_QUESTIONS[name]["type"] for name in WARMUP_QUESTIONS]

    def validate_params(self, params):
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        state = params.get("state")
        # Jev's `EntryType` allows `null`, which upstream renders as the JSON
        # literal; anything else must be a string, object or list.
        if state is not None and not isinstance(state, (str, dict, list)):
            raise ValueError("state must be a string, object or list")
        if isinstance(state, str) and not state.strip():
            raise ValueError("state must not be an empty string")
        if isinstance(state, (dict, list)) and not state:
            raise ValueError("state must not be empty")
        questions = params.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        for name, question in questions.items():
            if not isinstance(question, dict):
                raise ValueError(f"question {name!r} must be an object")
            qtype = question.get("type")
            if qtype not in QUESTION_TYPES:
                raise ValueError(
                    f"question {name!r} has unsupported type {qtype!r}; "
                    f"expected one of {', '.join(QUESTION_TYPES)}"
                )
            # Jev types `instructions` as `EntryType` (string, object, array or
            # null); upstream renders non-strings as compact JSON.
            if "instructions" not in question:
                raise ValueError(f"question {name!r} needs instructions")
        return state, questions

    def serve(self, agent) -> int:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            if len(line.encode("utf-8", errors="replace")) > MAX_INPUT_BYTES:
                self.fatal("frame_too_large", "request frame exceeds the worker limit")
                return 1
            try:
                message = json.loads(line)
            except ValueError as err:
                self.fatal("invalid_frame", f"could not parse the request frame: {err}")
                return 1
            if not isinstance(message, dict):
                self.fatal("invalid_frame", "request frame must be a JSON object")
                return 1

            method = message.get("method")
            request_id = message.get("id")
            if method == "shutdown":
                log("shutdown requested")
                return 0
            if method != "predict":
                self.reject(request_id, "invalid_request", f"unsupported method {method!r}")
                continue

            try:
                state, questions = self.validate_params(message.get("params"))
            except ValueError as err:
                self.reject(request_id, "invalid_request", str(err))
                continue

            try:
                result = agent.predict(state, questions)
            except (ValueError, KeyError) as err:
                # Request-scoped: the model is fine, the question is not.
                self.reject(request_id, "invalid_request", f"{type(err).__name__}: {err}")
                continue
            except BaseException:
                # Anything else means the resident state is suspect: exit and let
                # the daemon fail closed instead of serving doubtful answers.
                traceback.print_exc(file=sys.stderr)
                self.fatal("inference_failed", "inference failed, see worker stderr", request_id)
                return 1

            self.emit({"id": request_id, "ok": True, "result": result})
        log("input stream closed")
        return 0


def module_version(name: str) -> str:
    module = sys.modules.get(name)
    return getattr(module, "__version__", "unknown") if module else "unknown"


def backend_version(backend: str) -> str:
    """Version of the inference stack that was actually imported."""
    names = ("laya_mlx", "mlx.core") if backend == "mlx" else ("laya", "torch")
    for name in names:
        version = module_version(name)
        if version != "unknown":
            return version
    return "unknown"


def device_label(agent) -> str:
    """Actual device of a loaded agent, in a shape both stacks label plainly.

    PyTorch exposes it as a string (``torch.device("cpu").type``); MLX exposes
    a ``DeviceType`` enum member whose string form (``DeviceType.gpu``) has to
    be reduced to the device name.
    """
    value = getattr(agent, "device", None)
    if value is None:
        return "unknown"
    for attr in ("type", "device"):
        candidate = getattr(value, attr, None)
        if candidate is None:
            continue
        text = candidate if isinstance(candidate, str) else str(candidate)
        if not isinstance(candidate, str) and _looks_like_repr(text):
            continue
        if text:
            return _plain_device_name(text)
    return _plain_device_name(str(value))


def _looks_like_repr(text: str) -> bool:
    """Reject a nested object's repr so its own attributes get a chance."""
    return text.startswith("Device(")


def _plain_device_name(text: str) -> str:
    """Reduce a stack's device label to a plain name.

    ``torch.device("cpu")`` already yields ``"cpu"``. MLX yields its enum or
    repr — ``DeviceType.gpu`` / ``Device(gpu, 0)`` — so both are reduced to the
    device name and anything unparseable is passed through verbatim.
    """
    if "(" in text and text.rstrip().endswith(")"):
        text = text[text.index("(") + 1 : text.rindex(")")].split(",")[0].strip()
    elif "." in text:
        text = text.rsplit(".", 1)[-1]
    return text or "unknown"


def main(argv: list[str]) -> int:
    stream = isolate_protocol_stream()
    args = parse_args(argv)
    worker = Worker(stream, args)

    try:
        agent = worker.load()
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        worker.emit(
            {
                "ready": False,
                "protocol": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "error": {
                    "code": "load_failed",
                    "message": f"{type(err).__name__}: {err}",
                },
            }
        )
        return 1

    try:
        apply_memory_budget(args)
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        worker.emit(
            {
                "ready": False,
                "protocol": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "error": {
                    "code": "load_failed",
                    "message": f"memory budget could not be applied: {type(err).__name__}: {err}",
                },
            }
        )
        return 1

    try:
        warmup = worker.warmup(agent)
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        worker.emit(
            {
                "ready": False,
                "protocol": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "error": {"code": "warmup_failed", "message": f"warmup failed: {err}"},
            }
        )
        return 1

    device = device_label(agent)
    worker.emit(
        {
            "ready": True,
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "model": args.model,
            "checkpoint": args.subfolder or "english",
            "device": device,
            "requested_device": args.device,
            "backend": args.backend,
            "laya_version": backend_version(args.backend),
            "mlx_version": module_version("mlx.core"),
            "python_version": sys.version.split()[0],
            "warmup": warmup,
            "memory_budget_mb": args.memory_mb,
        }
    )
    log(f"ready on {device} (warmup: {', '.join(warmup) if warmup else 'skipped'})")
    return worker.serve(agent)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
