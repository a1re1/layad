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

* stdout carries protocol frames *only*. Anything Laya, PyTorch or transformers
  print (including C-level writes to fd 1) is redirected to stderr before the
  library is imported, so a stray ``print`` can never corrupt the stream.
* A malformed frame or a dead-end inference error is fatal: the process exits
  non-zero and the daemon fails closed rather than answering with state it does
  not trust.
* Request-level problems (bad question structure, upstream validation errors)
  are reported with the request id so the worker stays usable.

Inference is delegated to upstream ``laya`` (0.3.4): this file does not
reimplement the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

PROTOCOL_VERSION = 1

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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resident Laya worker for layad")
    parser.add_argument("--model", default=os.environ.get("LAYAD_MODEL", "convaiinnovations/laya"))
    parser.add_argument("--device", default=os.environ.get("LAYAD_DEVICE", "cpu"))
    parser.add_argument("--subfolder", default=None)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="load the model but skip the warmup forward pass (debugging only)",
    )
    return parser.parse_args(argv)


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
        log(f"importing laya (model={self.args.model!r} device={self.args.device!r})")
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

    device = getattr(getattr(agent, "device", None), "type", None) or "unknown"
    worker.emit(
        {
            "ready": True,
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "model": args.model,
            "checkpoint": args.subfolder or "english",
            "device": device,
            "requested_device": args.device,
            "laya_version": module_version("laya"),
            "torch_version": module_version("torch"),
            "transformers_version": module_version("transformers"),
            "python_version": sys.version.split()[0],
            "warmup": warmup,
        }
    )
    log(f"ready on {device} (warmup: {', '.join(warmup) if warmup else 'skipped'})")
    return worker.serve(agent)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
