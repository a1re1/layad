"""Bridge tests for python/worker.py.

These use a stub ``laya`` module, so they need no model download and no torch.
They cover the parts of the protocol the daemon relies on: readiness after a
warmup, request ids, request-scoped validation errors, fatal handling of
malformed frames, and the stdout isolation that keeps protocol frames clean.
"""

import io
import json
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

WORKER_PATH = pathlib.Path(__file__).resolve().parents[1] / "worker.py"


def load_worker_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("layad_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAgent:
    """Stands in for ``laya.Agent``: records calls, returns upstream shapes."""

    class Device:
        type = "cpu"

    def __init__(self, fail_with=None, noisy=False):
        self.device = self.Device()
        self.calls = []
        self.fail_with = fail_with
        self.noisy = noisy

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.noisy:
            # A library that writes to fd 1 must not be able to corrupt the
            # protocol stream.
            print("noise from a third-party library")
        if self.fail_with is not None:
            raise self.fail_with
        return {
            "model": "laya-rl-agent",
            "answers": {name: {"type": question["type"]} for name, question in questions.items()},
            "usage": {"input_tokens": 3, "output_tokens": 0},
        }


class WorkerModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_worker_module()

    def make_worker(self, args=None):
        argv = args or ["--model", "fake/model", "--device", "cpu"]
        parsed = self.worker.parse_args(argv)
        stream = io.StringIO()
        return self.worker.Worker(stream, parsed), stream

    def frames(self, stream):
        return [json.loads(line) for line in stream.getvalue().splitlines() if line]

    def run_serve(self, worker, lines, agent=None):
        stdin = sys.stdin
        sys.stdin = io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n")
        try:
            return worker.serve(agent or FakeAgent())
        finally:
            sys.stdin = stdin

    def test_validate_params_accepts_supported_shapes(self):
        worker, _ = self.make_worker()
        state, questions = worker.validate_params(
            {
                "state": {"customer": "angry"},
                "questions": {"q": {"type": "noul", "instructions": "a human?"}},
            }
        )
        self.assertEqual(state, {"customer": "angry"})
        self.assertIn("q", questions)

    def test_validate_params_rejects_bad_input(self):
        worker, _ = self.make_worker()
        cases = [
            ("params must be an object", "not-an-object"),
            ("state must be a string, object or list", {"state": 4, "questions": {"q": {}}}),
            ("state must not be an empty string", {"state": "  ", "questions": {"q": {}}}),
            ("state must not be empty", {"state": {}, "questions": {"q": {}}}),
            ("questions must be a nonempty object", {"state": "x", "questions": {}}),
            (
                "unsupported type",
                {"state": "x", "questions": {"q": {"type": "guess", "instructions": "x"}}},
            ),
            (
                "needs instructions",
                {"state": "x", "questions": {"q": {"type": "noul", "instructions": 7}}},
            ),
        ]
        for message, params in cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    worker.validate_params(params)
                self.assertIn(message, str(caught.exception))

    def test_predict_request_answers_with_its_own_id(self):
        worker, stream = self.make_worker()
        agent = FakeAgent()
        exit_code = self.run_serve(
            worker,
            [
                {
                    "id": "7",
                    "method": "predict",
                    "params": {
                        "state": "a string state",
                        "questions": {
                            "intent": {"type": "choice", "instructions": "what?", "criteria": {"a": "b"}},
                            "urgency": {"type": "score", "instructions": "how much?", "criteria": ["low", "high"]},
                            "human": {"type": "noul", "instructions": "human?"},
                        },
                    },
                }
            ],
            agent,
        )
        self.assertEqual(exit_code, 0)
        frames = self.frames(stream)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["id"], "7")
        self.assertTrue(frames[0]["ok"])
        self.assertEqual(
            sorted(frames[0]["result"]["answers"]), ["human", "intent", "urgency"]
        )
        self.assertEqual(len(agent.calls), 1)

    def test_invalid_request_is_reported_without_killing_the_worker(self):
        worker, stream = self.make_worker()
        exit_code = self.run_serve(
            worker,
            [
                {"id": "1", "method": "predict", "params": {"state": "x", "questions": {}}},
                {
                    "id": "2",
                    "method": "unsupported",
                    "params": {},
                },
                {
                    "id": "3",
                    "method": "predict",
                    "params": {
                        "state": "x",
                        "questions": {"q": {"type": "noul", "instructions": "human?"}},
                    },
                },
            ],
        )
        self.assertEqual(exit_code, 0)
        frames = self.frames(stream)
        self.assertEqual([frame["id"] for frame in frames], ["1", "2", "3"])
        self.assertEqual(frames[0]["error"]["code"], "invalid_request")
        self.assertEqual(frames[1]["error"]["code"], "invalid_request")
        self.assertTrue(frames[2]["ok"])

    def test_upstream_validation_error_is_request_scoped(self):
        worker, stream = self.make_worker()
        agent = FakeAgent(fail_with=ValueError("question 'q' options exceed head_max_len=192"))
        exit_code = self.run_serve(
            worker,
            [
                {
                    "id": "9",
                    "method": "predict",
                    "params": {
                        "state": "x",
                        "questions": {"q": {"type": "noul", "instructions": "human?"}},
                    },
                }
            ],
            agent,
        )
        self.assertEqual(exit_code, 0, "a request-scoped error must not kill the worker")
        frame = self.frames(stream)[0]
        self.assertFalse(frame["ok"])
        self.assertEqual(frame["error"]["code"], "invalid_request")
        self.assertIn("head_max_len", frame["error"]["message"])

    def test_malformed_frame_is_fatal(self):
        worker, stream = self.make_worker()
        stdin = sys.stdin
        sys.stdin = io.StringIO("this is not json\n")
        try:
            exit_code = worker.serve(FakeAgent())
        finally:
            sys.stdin = stdin
        self.assertEqual(exit_code, 1)
        frame = self.frames(stream)[0]
        self.assertTrue(frame["fatal"])
        self.assertEqual(frame["error"]["code"], "invalid_frame")

    def test_oversized_frame_is_fatal(self):
        worker, stream = self.make_worker()
        stdin = sys.stdin
        sys.stdin = io.StringIO("x" * (self.worker.MAX_INPUT_BYTES + 16) + "\n")
        try:
            exit_code = worker.serve(FakeAgent())
        finally:
            sys.stdin = stdin
        self.assertEqual(exit_code, 1)
        self.assertEqual(self.frames(stream)[0]["error"]["code"], "frame_too_large")

    def test_unexpected_inference_error_is_fatal(self):
        worker, stream = self.make_worker()
        agent = FakeAgent(fail_with=RuntimeError("torch exploded"))
        exit_code = self.run_serve(
            worker,
            [
                {
                    "id": "4",
                    "method": "predict",
                    "params": {
                        "state": "x",
                        "questions": {"q": {"type": "noul", "instructions": "human?"}},
                    },
                }
            ],
            agent,
        )
        self.assertEqual(exit_code, 1)
        frame = self.frames(stream)[0]
        self.assertEqual(frame["id"], "4")
        self.assertTrue(frame["fatal"])

    def test_shutdown_method_exits_cleanly(self):
        worker, stream = self.make_worker()
        exit_code = self.run_serve(worker, [{"id": None, "method": "shutdown"}])
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.frames(stream), [])

    def test_json_default_serializes_tensor_like_values(self):
        class TensorLike:
            def item(self):
                return 0.25

        self.assertEqual(self.worker.json_default(TensorLike()), 0.25)
        self.assertEqual(self.worker.json_default(object()), str(object()))

    def test_warmup_requires_all_three_primitives(self):
        worker, _ = self.make_worker()
        self.assertEqual(
            sorted(
                question["type"] for question in self.worker.WARMUP_QUESTIONS.values()
            ),
            ["choice", "noul", "score"],
        )
        agent = FakeAgent()
        self.assertEqual(sorted(worker.warmup(agent)), ["choice", "noul", "score"])
        self.assertEqual(len(agent.calls), 1, "warmup must be a single forward pass")

    def test_no_warmup_skips_the_forward_pass(self):
        worker, _ = self.make_worker(["--no-warmup"])
        agent = FakeAgent()
        self.assertEqual(worker.warmup(agent), [])
        self.assertEqual(agent.calls, [])


class ProtocolIsolationTests(unittest.TestCase):
    """The protocol stream must survive a library that prints to fd 1."""

    def test_stdout_only_carries_protocol_frames(self):
        program = (
            "import importlib.util, sys;\n"
            f"spec = importlib.util.spec_from_file_location('layad_worker', {str(WORKER_PATH)!r});\n"
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module);\n"
            "stream = module.isolate_protocol_stream();\n"
            "print('library noise on fd 1');\n"
            "sys.stdout.write('more noise\\n');\n"
            "stream.write('{\"ready\": true}\\n'); stream.flush();\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, '{"ready": true}\n')
        self.assertIn("library noise", completed.stderr)


class DeviceSelectionTests(unittest.TestCase):
    """The worker's --device handling, with laya stubbed out (no torch)."""

    def test_auto_translates_to_no_device_argument(self):
        worker = load_worker_module()
        calls = []

        class StubLaya:
            def load(self, model, **kwargs):
                calls.append((model, kwargs))
                return FakeAgent()

        args = worker.parse_args(["--model", "some/repo", "--device", "auto"])
        stream = io.StringIO()
        with mock.patch.dict(sys.modules, {"laya": StubLaya()}):
            instance = worker.Worker(stream, args)
            instance.load()

        self.assertEqual(len(calls), 1)
        model, kwargs = calls[0]
        self.assertEqual(model, "some/repo")
        # None means "let upstream choose"; the string "auto" must never reach it.
        self.assertIsNone(kwargs["device"])
        self.assertIsNone(kwargs["subfolder"])

    def test_explicit_device_and_subfolder_are_passed_through(self):
        worker = load_worker_module()
        calls = []

        class StubLaya:
            def load(self, model, **kwargs):
                calls.append(kwargs)
                return FakeAgent()

        args = worker.parse_args(
            ["--model", ".layad/model", "--device", "cpu", "--subfolder", "multilingual"]
        )
        with mock.patch.dict(sys.modules, {"laya": StubLaya()}):
            worker.Worker(io.StringIO(), args).load()

        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(calls[0]["subfolder"], "multilingual")


if __name__ == "__main__":
    unittest.main()
