#!/usr/bin/env python3
"""Dependency-free stand-in for ``python/worker.py``.

Rust integration tests drive the daemon against this script, so the whole
protocol surface (readiness, ids, malformed and oversized frames, timeouts,
cancellation and idle death) is exercised without torch, transformers or any
model download. It speaks the same NDJSON protocol as the real worker and, like
the real worker, keeps its stdout free of anything but protocol frames.

Usage: fake_worker.py --mode MODE [--model M] [--device D] [--subfolder S]
                      [--delay SECONDS] [--reply-id ID] [--pid-file PATH]
                      [--env-file PATH]

Modes:
  normal          announce readiness, answer every request with plausible shapes
  silent-startup  never announce readiness and never exit
  no-ready-exit   exit(3) without announcing readiness
  slow-inference  announce readiness, sleep --delay before answering (default 2s)
  die-idle        announce readiness, then exit(9) after --delay (default 1s)
  die-after-first announce readiness, answer one request, then exit(9)
  malformed       announce readiness, then write a line that is not JSON
  wrong-id        announce readiness, then answer with a frame id of --reply-id
  oversized       announce readiness, then answer with a very large single frame
  ready-then-eof  announce readiness, then close stdout
  fragmented      write the ready frame and every reply in pieces separated by a
                  real time gap (--gap seconds, default 0.25s), so the daemon
                  must reassemble frames across reads
  never-reads-stdin
                  announce readiness, then never read stdin at all (requests
                  pile up in the pipe instead of being answered)
  reject-first    announce readiness, answer the first request with ok:false,
                  then serve every later request normally
  not-ready       answer ready:false with a load_failed error (the real worker's
                  response when the requested checkpoint subfolder is missing)
                  and exit 1, so the daemon must fail the startup

--pid-file writes the process id as soon as the worker starts, so a test can
signal exactly this child (never a pattern match over other processes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def isolate_protocol_stream():
    """Point fd 1 at stderr so only protocol frames reach the daemon."""
    try:
        protocol_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(stderr_fd, 1)
        os.close(stderr_fd)
        stream = os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n", buffering=1)
    except OSError:
        return sys.stdout
    sys.stdout = sys.stderr
    return stream


def parse_args(argv):
    parser = argparse.ArgumentParser(description="fake layad worker")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--model", default="fake/model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--reply-id", default="999")
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--gap", type=float, default=0.25)
    return parser.parse_args(argv)


def answer_for(name, question):
    """Build an answer with the shape upstream laya returns for that primitive."""
    qtype = question.get("type")
    if qtype == "choice":
        criteria = question.get("criteria") or {}
        if isinstance(criteria, dict):
            keys = list(criteria.keys())
        else:
            keys = [str(item) for item in criteria]
        if not keys:
            keys = ["yes", "no"]
        probability = round(1.0 / len(keys), 4)
        return {
            "type": "choice",
            "choice": keys[0],
            "probabilities": {key: probability for key in keys},
            "confidence": 0.5,
            "action": {"act_probability": 0.1},
        }
    if qtype == "score":
        criteria = question.get("criteria") or ["low", "high"]
        probability = round(1.0 / len(criteria), 4)
        return {
            "type": "score",
            "score": 1.0,
            "legend": {str(i): level for i, level in enumerate(criteria)},
            "probabilities": {str(i): probability for i in range(len(criteria))},
            "confidence": 0.5,
            "action": {"act_probability": 0.1},
        }
    return {
        "type": "noul",
        "noul": 0.25,
        "confidence": 0.75,
        "action": {"act_probability": 0.05},
    }


def result_for(params):
    questions = params.get("questions") or {}
    return {
        "model": "fake-laya",
        "answers": {name: answer_for(name, question) for name, question in questions.items()},
        "usage": {"input_tokens": len(str(params.get("state", ""))), "output_tokens": 0},
        "worker_pid": os.getpid(),
    }


def emit_fragmented(stream, frame, pieces=4, gap=0.25):
    """Write one protocol frame in several chunks with a real gap between them."""
    payload = json.dumps(frame, ensure_ascii=False) + "\n"
    size = max(1, len(payload) // pieces)
    for start in range(0, len(payload), size):
        stream.write(payload[start : start + size])
        stream.flush()
        if start + size < len(payload):
            time.sleep(gap)


def main(argv):
    stream = isolate_protocol_stream()
    args = parse_args(argv)
    mode = args.mode
    delay = args.delay
    gap = args.gap

    def emit(frame):
        if mode == "fragmented":
            emit_fragmented(stream, frame, gap=gap)
            return
        stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
        stream.flush()

    sys.stderr.write(f"[fake-worker] mode={mode} pid={os.getpid()}\n")
    sys.stderr.flush()
    if args.pid_file:
        with open(args.pid_file, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
    if args.env_file:
        # Dump the environment the daemon actually handed this child, so a test
        # can assert the cache/checkpoint contract without reaching into daemon
        # internals.
        keys = [
            "HF_HOME",
            "LAYAD_MODEL",
            "LAYAD_DEVICE",
            "LAYAD_CHECKPOINT",
            "HF_TOKEN",
            "HOME",
        ]
        with open(args.env_file, "w", encoding="utf-8") as handle:
            for key in keys:
                handle.write(f"{key}={os.environ.get(key, '')}\n")

    if mode == "no-ready-exit":
        return 3

    if mode == "not-ready":
        emit(
            {
                "ready": False,
                "protocol": 1,
                "pid": os.getpid(),
                "model": args.model,
                "error": {
                    "code": "load_failed",
                    "message": f"checkpoint subfolder {args.subfolder!r} is missing",
                },
            }
        )
        return 1

    if mode != "silent-startup":
        emit(
            {
                "ready": True,
                "protocol": 1,
                "pid": os.getpid(),
                "model": args.model,
                "checkpoint": args.subfolder or "english",
                "device": args.device,
                "requested_device": args.device,
                "laya_version": "fake",
                "warmup": ["choice", "score", "noul"],
            }
        )
        if mode == "ready-then-eof":
            return 0

    if mode == "die-idle":
        time.sleep(delay if delay is not None else 1.0)
        return 9

    if mode == "never-reads-stdin":
        # The request never gets an answer: the daemon's deadline must bound it.
        while True:
            time.sleep(1.0)

    served = 0
    requests = 0
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        message = json.loads(line)
        if message.get("method") == "shutdown":
            return 0
        request_id = message.get("id")
        requests += 1

        if mode == "reject-first" and requests == 1:
            emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "code": "invalid_request",
                        "message": "upstream rejected the question set",
                    },
                }
            )
            continue
        if mode == "slow-inference":
            time.sleep(delay if delay is not None else 2.0)
        if mode == "malformed":
            stream.write("this is not a json frame\n")
            stream.flush()
            return 1
        if mode == "oversized":
            stream.write(json.dumps({"id": request_id, "ok": True, "result": "x" * 4096}) + "\n")
            stream.flush()
            return 0
        if mode == "wrong-id":
            emit({"id": args.reply_id, "ok": True, "result": result_for(message.get("params") or {})})
            return 0

        emit({"id": request_id, "ok": True, "result": result_for(message.get("params") or {})})
        served += 1
        if mode == "die-after-first" and served >= 1:
            return 9
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
