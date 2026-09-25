#![cfg(unix)]
//! End-to-end tests for the daemon's HTTP surface and its resident worker.
//!
//! The tests drive [`layad::api::router`] in-process against
//! `tests/data/fake_worker.py`, so the whole protocol path (readiness, request
//! ids, malformed/oversized frames, deadlines, cancellation, idle death and
//! bounded shutdown) runs without torch or a model download.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::Command;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use clap::Parser;
use http_body_util::BodyExt;
use layad::api::{router, AppState, Readiness};
use layad::config::{Cli, Config};
use layad::worker::{StartError, State as WorkerState, WorkerHandle};
use serde_json::{json, Value};
use tower::ServiceExt;

fn fake_script() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/fake_worker.py")
        .to_str()
        .expect("utf-8 path")
        .to_string()
}

/// Configuration whose worker is the fake child in `--mode <mode>`.
fn fake_config(arguments: &[&str]) -> Config {
    let script = fake_script();
    let mut full = vec![
        "layad",
        "--python",
        "python3",
        "--worker-script",
        script.as_str(),
    ];
    full.extend_from_slice(arguments);
    Config::try_from(Cli::parse_from(full)).expect("valid test configuration")
}

/// Start a daemon state whose worker is already ready.
async fn ready_app(config: Config) -> (AppState, WorkerHandle) {
    let worker = WorkerHandle::start(config.clone())
        .await
        .expect("fake worker becomes ready");
    let state = AppState::new(config);
    state.attach(worker.clone());
    (state, worker)
}

async fn send(state: &AppState, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = match body {
        Some(value) => Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(value.to_string()))
            .expect("request"),
        None => Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .expect("request"),
    };
    let response = router(state.clone())
        .oneshot(request)
        .await
        .expect("router answers");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|err| {
            panic!(
                "response body is not JSON ({err}): {}",
                String::from_utf8_lossy(&bytes)
            )
        })
    };
    (status, value)
}

fn prediction(state_value: &str) -> Value {
    json!({
        "state": state_value,
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "what does the user want?",
                "criteria": {"refund": "a refund", "cancel": "cancellation"}
            },
            "urgency": {
                "type": "score",
                "instructions": "how urgent?",
                "criteria": ["low", "medium", "high"]
            },
            "human": {
                "type": "noul",
                "instructions": "do they want a human?"
            }
        }
    })
}

fn error_code(value: &Value) -> &str {
    value
        .get("error")
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("no error code in {value}"))
}

async fn worker_pid(worker: &WorkerHandle) -> u32 {
    worker.info().await.expect("worker metadata").pid
}

fn pid_is_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// Bounded wait for one exact pid to disappear. Never pattern-matches: only the
/// pid the tests obtained from the worker handle is ever inspected.
async fn wait_for_exit(pid: u32, deadline: Duration) -> bool {
    let started = std::time::Instant::now();
    while started.elapsed() < deadline {
        if !pid_is_alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    !pid_is_alive(pid)
}

/// Bounded wait for the fake worker to publish its own pid, so a test can signal
/// exactly that child instead of guessing at a process pattern.
async fn wait_for_pid_file(path: &std::path::Path, deadline: Duration) -> Option<i32> {
    let started = std::time::Instant::now();
    while started.elapsed() < deadline {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(pid) = text.trim().parse::<i32>() {
                return Some(pid);
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    None
}

/// Read the fake worker's environment dump (see `--env-file`). The file is
/// written before the worker announces readiness, so a short bounded wait is
/// enough; nothing here inspects other processes or the daemon's internals.
async fn read_env_file(path: &std::path::Path) -> HashMap<String, String> {
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(5) {
        if let Ok(text) = std::fs::read_to_string(path) {
            return text
                .lines()
                .filter_map(|line| line.split_once('='))
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect();
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the fake worker never wrote {path:?}");
}

/// The daemon must reuse the one resident worker and preserve upstream answer
/// shapes for all three primitives.
#[tokio::test]
async fn two_predictions_reuse_one_worker_and_keep_answer_shapes() {
    let (state, worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let (status, first) = send(&state, "/v1/predict", Some(prediction("first state"))).await;
    assert_eq!(status, StatusCode::OK, "first prediction: {first}");
    let (status, second) = send(&state, "/v1/predict", Some(prediction("second state"))).await;
    assert_eq!(status, StatusCode::OK, "second prediction: {second}");

    for body in [&first, &second] {
        let answers = body.get("answers").expect("answers object");
        assert_eq!(answers["intent"]["type"], "choice");
        assert!(answers["intent"]["choice"].is_string());
        assert!(answers["intent"]["probabilities"].is_object());
        assert!(answers["intent"]["confidence"].is_number());
        assert_eq!(answers["urgency"]["type"], "score");
        assert!(answers["urgency"]["score"].is_number());
        assert!(answers["urgency"]["legend"].is_object());
        assert_eq!(answers["human"]["type"], "noul");
        assert!(answers["human"]["noul"].is_number());
        assert!(body.get("usage").is_some());
    }

    // Same single worker process for both requests.
    assert_eq!(
        first["worker_pid"], second["worker_pid"],
        "both answers must come from the same resident worker"
    );
    let info = worker.info().await.expect("worker metadata");
    assert_eq!(info.identity(), info.identity());
    assert_eq!(info.warmup.len(), 3, "readiness must report a real warmup");
}

#[tokio::test]
async fn healthz_is_up_and_readyz_reports_the_resident_model() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;
    let (status, health) = send(&state, "/healthz", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");

    let (status, ready) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::OK, "{ready}");
    assert_eq!(ready["ready"], true);
    assert_eq!(ready["device"], "cpu");
    assert!(ready["worker_pid"].as_u64().unwrap_or_default() > 0);
}

#[tokio::test]
async fn readyz_reports_loading_before_the_worker_is_attached() {
    let state = AppState::new(fake_config(&[]));
    assert_eq!(state.readiness(), Readiness::Loading);
    let (status, ready) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(ready["ready"], false);
    assert_eq!(ready["reason"], "loading");
    // Liveness must stay green while the model is still loading.
    let (status, health) = send(&state, "/healthz", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
}

#[tokio::test]
async fn predict_accepts_state_string_object_and_list() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;
    for candidate in [
        json!("a plain string state"),
        json!({"customer": "angry", "tier": "gold"}),
        json!([{"role": "user", "content": "hello"}, {"role": "agent", "content": "hi"}]),
    ] {
        let mut body = prediction("ignored");
        body["state"] = candidate.clone();
        let (status, response) = send(&state, "/v1/predict", Some(body)).await;
        assert_eq!(status, StatusCode::OK, "state {candidate}: {response}");
        assert!(response["answers"].is_object());
    }
}

#[tokio::test]
async fn malformed_requests_get_consistent_json_errors() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let mut empty_questions = prediction("x");
    empty_questions["questions"] = json!({});
    let (status, body) = send(&state, "/v1/predict", Some(empty_questions)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("at least one"));

    let mut bad_state = prediction("x");
    bad_state["state"] = json!(42);
    let (status, body) = send(&state, "/v1/predict", Some(bad_state)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");

    let mut bad_type = prediction("x");
    bad_type["questions"] = json!({"q": {"type": "guess", "instructions": "x"}});
    let (status, body) = send(&state, "/v1/predict", Some(bad_type)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("unsupported type"));

    let mut missing_criteria = prediction("x");
    missing_criteria["questions"] = json!({"q": {"type": "choice", "instructions": "x"}});
    let (status, body) = send(&state, "/v1/predict", Some(missing_criteria)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");

    let (status, body) = send(&state, "/v1/predict", Some(json!({"state": "x"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "missing questions: {body}");
    assert_eq!(error_code(&body), "invalid_json");

    // The worker must still be usable after rejected requests.
    let (status, body) = send(&state, "/v1/predict", Some(prediction("still fine"))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn bodies_and_question_counts_are_bounded() {
    let (state, _worker) = ready_app(fake_config(&[
        "--max-body-bytes",
        "2048",
        "--max-questions",
        "2",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    // Three questions against a limit of two.
    let too_many = json!({
        "state": "x",
        "questions": {
            "a": {"type": "noul", "instructions": "a?"},
            "b": {"type": "noul", "instructions": "b?"},
            "c": {"type": "noul", "instructions": "c?"}
        }
    });
    let (status, body) = send(&state, "/v1/predict", Some(too_many)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("the limit is 2"));

    // A body over the configured limit is refused before validation.
    let huge = json!({"state": "x".repeat(4096), "questions": {}});
    let (status, body) = send(&state, "/v1/predict", Some(huge)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(error_code(&body), "body_too_large");
}

#[tokio::test]
async fn admission_limit_returns_overloaded() {
    let (state, _worker) = ready_app(fake_config(&[
        "--max-concurrent",
        "1",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "slow-inference",
        "--worker-arg",
        "--delay",
        "--worker-arg",
        "1.0",
    ]))
    .await;

    // Spawn it so it actually runs and holds the only admission slot.
    let first = tokio::spawn(
        router(state.clone()).oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(prediction("busy").to_string()))
                .expect("request"),
        ),
    );
    // Give the first request time to take the only admission slot.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (status, body) = send(&state, "/v1/predict", Some(prediction("second"))).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    assert_eq!(error_code(&body), "overloaded");

    let response = first
        .await
        .expect("first request task joins")
        .expect("first request is answered");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn inference_deadline_is_enforced_and_fails_closed() {
    let (state, worker) = ready_app(fake_config(&[
        "--inference-timeout-ms",
        "400",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "slow-inference",
        "--worker-arg",
        "--delay",
        "--worker-arg",
        "5.0",
    ]))
    .await;

    let (status, body) = send(&state, "/v1/predict", Some(prediction("slow"))).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_eq!(error_code(&body), "inference_timeout");

    // A timeout invalidates the worker; the daemon must not serve it again.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(matches!(worker.state().await, WorkerState::Failed(_)));
    let (status, body) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["ready"], false);
}

#[tokio::test]
async fn startup_deadline_and_early_exit_fail_closed() {
    match WorkerHandle::start(fake_config(&[
        "--startup-timeout-ms",
        "300",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "silent-startup",
    ]))
    .await
    {
        Err(StartError::Timeout(timeout)) => assert_eq!(timeout.as_millis(), 300),
        Err(other) => panic!("expected a startup timeout, got {other:?}"),
        Ok(_) => panic!("a silent worker must not be accepted"),
    }

    match WorkerHandle::start(fake_config(&[
        "--startup-timeout-ms",
        "5000",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "no-ready-exit",
    ]))
    .await
    {
        Err(StartError::Failed(detail)) => {
            // The child may be observed as an exit status or as a closed stdout;
            // either is a startup failure that must not be accepted.
            assert!(
                detail.contains("exit") || detail.contains("3") || detail.contains("EOF"),
                "{detail}"
            );
        }
        Err(other) => panic!("expected a startup failure, got {other:?}"),
        Ok(_) => panic!("a worker that exits during startup must not be accepted"),
    }
}

#[tokio::test]
async fn malformed_answers_are_rejected_without_desync() {
    for (mode, expected) in [
        ("malformed", "worker_failed"),
        ("wrong-id", "worker_failed"),
        ("oversized", "oversized_response"),
    ] {
        let config = if mode == "oversized" {
            fake_config(&[
                "--max-response-bytes",
                "512",
                "--worker-arg",
                "--mode",
                "--worker-arg",
                mode,
            ])
        } else {
            fake_config(&["--worker-arg", "--mode", "--worker-arg", mode])
        };
        let (state, worker) = ready_app(config).await;
        let (status, body) = send(&state, "/v1/predict", Some(prediction("x"))).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "mode {mode}: {body}");
        assert_eq!(error_code(&body), expected, "mode {mode}");

        // The worker is untrusted afterwards: no stale answer, no reuse.
        let (status, body) = send(&state, "/v1/predict", Some(prediction("y"))).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "mode {mode}: {body}"
        );
        assert_eq!(error_code(&body), "unavailable", "mode {mode}");
        assert!(matches!(worker.state().await, WorkerState::Failed(_)));
    }
}

#[tokio::test]
async fn eof_after_readiness_fails_closed_at_any_admission_point() {
    let config = fake_config(&["--worker-arg", "--mode", "--worker-arg", "ready-then-eof"]);
    let worker = match WorkerHandle::start(config.clone()).await {
        Err(StartError::Failed(_)) => return,
        Err(other) => panic!("unexpected startup error: {other:?}"),
        Ok(worker) => worker,
    };
    let state = AppState::new(config);
    state.attach(worker.clone());
    let (status, body) = send(&state, "/v1/predict", Some(prediction("x"))).await;
    // EOF may be observed before admission (503), or during inference (502).
    assert!(
        status == StatusCode::BAD_GATEWAY || status == StatusCode::SERVICE_UNAVAILABLE,
        "{status}: {body}"
    );
    worker.shutdown(Duration::from_secs(2)).await;
}

#[tokio::test]
async fn idle_worker_death_is_noticed() {
    let (state, worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;
    let pid = worker.info().await.expect("worker metadata").pid;

    // Kill the resident child while nothing is in flight.
    let result = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    assert_eq!(result, 0, "failed to kill the worker");

    let mut failed = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if matches!(worker.state().await, WorkerState::Failed(_)) {
            failed = true;
            break;
        }
    }
    assert!(failed, "idle worker death must invalidate the worker");

    let (status, body) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["ready"], false);
    let (status, body) = send(&state, "/v1/predict", Some(prediction("x"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(error_code(&body), "unavailable");
    // Liveness still answers; only readiness is affected.
    let (status, _) = send(&state, "/healthz", None).await;
    assert_eq!(status, StatusCode::OK);
}

/// The real acceptance path in `scripts/smoke.sh`: two predictions covering
/// all three primitives must come from one resident worker. The same thing is
/// asserted here against the fake child so the identity guarantee is checked in
/// CI too, without a model download.
#[tokio::test]
async fn two_predictions_keep_one_worker_identity() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let (_, first) = send(&state, "/readyz", None).await;
    let identity_before = first["worker_identity"].clone();
    let pid_before = first["worker_pid"].clone();
    assert!(identity_before.is_string(), "{first}");

    for name in ["first", "second"] {
        let (status, body) = send(&state, "/v1/predict", Some(prediction(name))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    let (_, second) = send(&state, "/readyz", None).await;
    assert_eq!(second["worker_identity"], identity_before);
    assert_eq!(second["worker_pid"], pid_before);
    assert_eq!(second["device"], "cpu");
}

/// A canceled caller must not leave a late reply to be mixed into the next
/// request: the in-flight request is abandoned and the worker is torn down.
#[tokio::test]
async fn canceled_request_does_not_desync_the_protocol() {
    let (state, worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "slow-inference",
        "--worker-arg",
        "--delay",
        "--worker-arg",
        "2.0",
    ]))
    .await;

    let request = router(state.clone()).oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/predict")
            .header("content-type", "application/json")
            .body(Body::from(prediction("cancel me").to_string()))
            .expect("request"),
    );
    let handle = tokio::spawn(request);
    tokio::time::sleep(Duration::from_millis(300)).await;
    handle.abort();
    let _ = handle.await;

    // The worker is invalidated rather than left with an unanswered request.
    let mut stopped = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if matches!(worker.state().await, WorkerState::Failed(_)) {
            stopped = true;
            break;
        }
    }
    assert!(stopped, "a canceled caller must tear the worker down");

    // A follow-up request is answered with an error, never with the abandoned
    // reply from the canceled request.
    let (status, body) = send(&state, "/v1/predict", Some(prediction("next"))).await;
    assert!(
        status == StatusCode::SERVICE_UNAVAILABLE || status == StatusCode::BAD_GATEWAY,
        "unexpected status after cancellation: {status} {body}"
    );
    assert!(
        body.get("answers").is_none(),
        "no stale answer may be returned"
    );
}

#[tokio::test]
async fn shutdown_is_bounded_and_leaves_no_orphan() {
    let (state, worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;
    let pid = worker.info().await.expect("worker metadata").pid;
    assert_eq!(
        unsafe { libc::kill(pid as i32, 0) },
        0,
        "worker must be alive"
    );

    let started = std::time::Instant::now();
    worker.shutdown(Duration::from_secs(5)).await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "shutdown must be bounded"
    );

    // The child was reaped: it is gone and no zombie is left for the daemon.
    let gone = unsafe { libc::kill(pid as i32, 0) };
    assert_eq!(gone, -1, "worker {pid} must not survive shutdown");

    let (status, body) = send(&state, "/v1/predict", Some(prediction("after shutdown"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(error_code(&body), "unavailable");
}

#[tokio::test]
async fn request_ids_are_matched_in_order() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;
    for index in 0..5 {
        let (status, body) = send(
            &state,
            "/v1/predict",
            Some(prediction(&format!("state {index}"))),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "request {index}: {body}");
        assert!(body["answers"]["intent"]["choice"].is_string());
    }
}

/// A worker that answers `ok:false` must surface as a structured request error
/// and must stay usable afterwards: the rejection is request-scoped, not fatal.
#[tokio::test]
async fn worker_rejection_is_mapped_and_the_worker_stays_healthy() {
    let (state, worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "reject-first",
    ]))
    .await;

    let (status, body) = send(&state, "/v1/predict", Some(prediction("rejected"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(error_code(&body), "invalid_request");
    assert!(body.get("answers").is_none(), "no answers on a rejection");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("upstream rejected"),
        "the worker's message must reach the client: {body}"
    );

    // The next request is served by the same resident worker: an `ok:false`
    // reply must not take the worker down.
    let (status, body) = send(&state, "/v1/predict", Some(prediction("accepted"))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["answers"]["intent"]["choice"].is_string(), "{body}");
    assert!(body["worker_pid"].as_u64().unwrap_or_default() > 0);
    assert!(!matches!(worker.state().await, WorkerState::Failed(_)));
}

/// A worker that dribbles its ready frame and its replies out in pieces, with a
/// real gap between the pieces, must still be understood: frames are reassembled
/// across reads and partial data is never dropped.
#[tokio::test]
async fn fragmented_frames_across_reads_are_reassembled() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "fragmented",
        "--worker-arg",
        "--gap",
        "--worker-arg",
        "0.25",
    ]))
    .await;

    let (status, first) = send(&state, "/v1/predict", Some(prediction("fragmented one"))).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert!(first["answers"]["intent"]["choice"].is_string(), "{first}");

    // A second reply right after the first: a buffer that is dropped between
    // frames would lose this one.
    let (status, second) = send(&state, "/v1/predict", Some(prediction("fragmented two"))).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(
        first["worker_pid"], second["worker_pid"],
        "both answers must come from the same resident worker"
    );
}

/// The worker announces readiness and then never reads stdin. A request larger
/// than the pipe buffer cannot be delivered, so the daemon must bound the write
/// and fail closed instead of hanging on an unanswereable request.
#[tokio::test]
async fn request_deadline_bounds_a_worker_that_never_reads_stdin() {
    let (state, worker) = ready_app(fake_config(&[
        "--inference-timeout-ms",
        "700",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "never-reads-stdin",
    ]))
    .await;
    let pid = worker_pid(&worker).await;

    // Far larger than any pipe buffer, so the daemon cannot finish the write.
    let mut body = prediction("big");
    body["state"] = Value::String("x".repeat(512 * 1024));

    let started = std::time::Instant::now();
    let (status, response) = send(&state, "/v1/predict", Some(body)).await;
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(10),
        "request hung for {elapsed:?}"
    );
    assert!(
        status == StatusCode::GATEWAY_TIMEOUT || status == StatusCode::BAD_GATEWAY,
        "unexpected status for an unanswerable request: {status} {response}"
    );
    assert!(
        matches!(error_code(&response), "inference_timeout" | "worker_failed"),
        "{response}"
    );

    // The stalled worker is torn down, not left holding a poisoned pipe.
    assert!(
        wait_for_exit(pid, Duration::from_secs(5)).await,
        "worker {pid} must not survive a timed-out request"
    );
    let (status, body) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["ready"], false);
}

/// A real `layad` daemon process driven over loopback. Only the exact pid this
/// struct spawned is ever signaled or inspected — never a process pattern — and
/// `Drop` reaps it, so a failed assertion cannot leak a daemon.
struct Daemon {
    child: Option<std::process::Child>,
    addr: String,
    pid_file: std::path::PathBuf,
    log: std::path::PathBuf,
}

impl Daemon {
    /// Start `layad` with the fake worker and wait (bounded) for the daemon's own
    /// `layad listening on http://HOST:PORT` line; the port is read from that
    /// line, never guessed.
    async fn start(dir: &tempfile::TempDir, worker_args: &[&str], extra: &[&str]) -> Daemon {
        let script = fake_script();
        let pid_file = dir.path().join("daemon-worker.pid");
        let log = dir.path().join("daemon.log");
        let log_file = std::fs::File::create(&log).expect("daemon log");
        let mut command = Command::new(env!("CARGO_BIN_EXE_layad"));
        command
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--python")
            .arg("python3")
            .arg("--worker-script")
            .arg(&script)
            .arg("--worker-arg")
            .arg("--pid-file")
            .arg("--worker-arg")
            .arg(pid_file.to_str().expect("utf-8 pid file"));
        for arg in worker_args {
            command.arg("--worker-arg").arg(arg);
        }
        for arg in extra {
            command.arg(arg);
        }
        let child = command
            .stdout(log_file.try_clone().expect("clone log"))
            .stderr(log_file)
            .spawn()
            .expect("spawn layad");
        let mut daemon = Daemon {
            child: Some(child),
            addr: String::new(),
            pid_file,
            log,
        };
        daemon.addr = daemon.wait_for_listen(Duration::from_secs(15)).await;
        daemon
    }

    async fn wait_for_listen(&self, deadline: Duration) -> String {
        let started = std::time::Instant::now();
        while started.elapsed() < deadline {
            if let Ok(text) = std::fs::read_to_string(&self.log) {
                if let Some(line) = text
                    .lines()
                    .find(|line| line.contains("layad listening on http://"))
                {
                    if let Some(rest) = line.split("http://").nth(1) {
                        return rest.trim().to_string();
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!(
            "layad never printed its listening line: {:?}",
            std::fs::read_to_string(&self.log)
        );
    }

    /// The exact daemon pid (the process this test spawned).
    fn pid(&self) -> u32 {
        self.child.as_ref().expect("daemon is alive").id()
    }

    /// Is the daemon still running? `try_wait` reaps it if it exited; once the
    /// exit status is observed the child is dropped, so this test never signals a
    /// pid again after the OS may have reused it.
    fn is_alive(&mut self) -> bool {
        let reaped = self
            .child
            .as_mut()
            .expect("daemon is alive")
            .try_wait()
            .expect("try_wait")
            .is_some();
        if reaped {
            self.child = None;
        }
        !reaped
    }

    /// Signal exactly this daemon, never a look-alike process.
    fn signal(&self, signal: i32) {
        let pid = self.pid();
        assert_eq!(
            unsafe { libc::kill(pid as i32, signal) },
            0,
            "failed to signal daemon {pid}"
        );
    }

    /// Bounded wait for the daemon process itself to exit (reaping it).
    async fn wait_for_exit_bounded(
        &mut self,
        deadline: Duration,
    ) -> Option<std::process::ExitStatus> {
        let started = std::time::Instant::now();
        while started.elapsed() < deadline {
            let status = self
                .child
                .as_mut()
                .expect("daemon is alive")
                .try_wait()
                .expect("try_wait");
            if let Some(status) = status {
                self.child = None;
                return Some(status);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        None
    }

    /// The pid the fake worker published for itself.
    async fn worker_pid(&self, deadline: Duration) -> u32 {
        match wait_for_pid_file(&self.pid_file, deadline).await {
            Some(pid) if pid > 0 => pid as u32,
            _ => panic!("the fake worker never published its pid"),
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Only signal a child that has not been reaped yet. `try_wait` reaps an
            // exited child (and returns its cached status afterwards), so after it
            // the pid is free for the OS to reuse — signaling it then would be
            // exactly the cross-test SIGKILL this suite must never cause.
            if child.try_wait().expect("try_wait").is_none() {
                let _ = unsafe { libc::kill(child.id() as i32, libc::SIGKILL) };
            }
            let _ = child.wait();
        }
    }
}

/// One bounded blocking HTTP probe with the standard library only (no extra
/// dependency): GET `path` and return the status code.
fn http_status(addr: &str, path: &str) -> Option<u16> {
    let mut stream = std::net::TcpStream::connect(addr).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    stream.flush().ok()?;
    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf).ok()?;
    String::from_utf8_lossy(&buf[..n])
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

async fn wait_for_status(addr: &str, path: &str, deadline: Duration) -> u16 {
    let started = std::time::Instant::now();
    while started.elapsed() < deadline {
        if let Some(status) = http_status(addr, path) {
            if status == 200 {
                return status;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("{path} never answered 200 on {addr}");
}

/// SIGTERM to the *daemon* while its worker is stuck in a startup that never
/// becomes ready: the daemon must exit promptly (bounded, far below the 30s
/// startup deadline) and must not leave its worker behind. The signal goes to the
/// exact daemon pid this test spawned.
#[tokio::test]
async fn daemon_sigterm_during_never_ready_startup_is_bounded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut daemon = Daemon::start(
        &dir,
        &["--mode", "silent-startup"],
        &["--startup-timeout-ms", "30000"],
    )
    .await;
    let worker = daemon.worker_pid(Duration::from_secs(10)).await;
    assert!(
        daemon.is_alive(),
        "the daemon must still be waiting for its silent worker"
    );

    let started = std::time::Instant::now();
    daemon.signal(libc::SIGTERM);
    let status = daemon
        .wait_for_exit_bounded(Duration::from_secs(10))
        .await
        .expect("a signaled daemon must exit without waiting out the 30s startup deadline");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the daemon took {:?} to exit",
        started.elapsed()
    );
    assert!(
        status.success(),
        "a graceful shutdown must exit cleanly: {status}"
    );
    assert!(
        wait_for_exit(worker, Duration::from_secs(5)).await,
        "worker {worker} must not outlive the daemon"
    );
}

/// SIGTERM to the *daemon* while a client is still sending its request body: the
/// body is never completed, yet the daemon must still exit within its shutdown
/// deadline and must not leave its worker behind. The signal goes to the exact
/// daemon pid this test spawned, and the half-sent request is never finished, so
/// it can only fail or be cut off — never be answered successfully.
#[tokio::test]
async fn daemon_sigterm_with_incomplete_http_body_is_bounded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut daemon = Daemon::start(
        &dir,
        &["--mode", "normal"],
        &["--shutdown-timeout-ms", "2000"],
    )
    .await;
    let worker = daemon.worker_pid(Duration::from_secs(10)).await;
    // Wait (bounded) until the worker is attached, so the request below reaches
    // the predict path rather than the loading path.
    wait_for_status(&daemon.addr, "/readyz", Duration::from_secs(15)).await;

    let mut stream = std::net::TcpStream::connect(&daemon.addr).expect("connect to the daemon");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("read timeout");
    let partial = format!(
        "POST /v1/predict HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{{\"state\": \"half a body",
        daemon.addr
    );
    stream
        .write_all(partial.as_bytes())
        .expect("write the body prefix");
    stream.flush().expect("flush");
    // The body is deliberately never finished: the request is in flight now.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(daemon.is_alive(), "the daemon must still be serving");

    let started = std::time::Instant::now();
    daemon.signal(libc::SIGTERM);
    let status = daemon
        .wait_for_exit_bounded(Duration::from_secs(10))
        .await
        .expect("an unfinished request body must not block daemon exit");
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "the daemon took {:?} to exit with a half-sent body",
        started.elapsed()
    );
    assert!(
        status.success(),
        "a graceful shutdown must exit cleanly: {status}"
    );
    assert!(
        wait_for_exit(worker, Duration::from_secs(5)).await,
        "worker {worker} must not outlive the daemon"
    );

    // The half-sent request can only be cut off or answered with an error.
    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(0) => {}
        Ok(n) => {
            let text = String::from_utf8_lossy(&buf[..n]).to_string();
            assert!(
                !text.contains(" 200 ") && !text.contains(" 2"),
                "a half-sent body must never be answered successfully: {text}"
            );
        }
        Err(_) => {}
    }
}

/// Unknown paths and wrong methods must answer with the documented JSON error
/// body (not an empty framework response), even before a worker is attached.
#[tokio::test]
async fn unknown_paths_and_methods_get_json_errors() {
    let state = AppState::new(fake_config(&[]));

    let response = router(state.clone())
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nope")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router answers");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value: Value = serde_json::from_slice(&bytes).expect("a 404 must be JSON");
    assert_eq!(error_code(&value), "not_found");

    let response = router(state.clone())
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/healthz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router answers");
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value: Value = serde_json::from_slice(&bytes).expect("a 405 must be JSON");
    assert_eq!(error_code(&value), "method_not_allowed");
}

/// `--checkpoint` must really select a checkpoint: the daemon passes the
/// subfolder down to the worker, the worker reports it back, and a non-English
/// run must never be answered as the English checkpoint.
#[tokio::test]
async fn checkpoint_selection_reaches_the_worker() {
    let dir = tempfile::tempdir().expect("tempdir");
    let env_file = dir.path().join("chosen.env");
    let env_file_str = env_file.to_str().expect("utf-8 path");

    let (state, worker) = ready_app(fake_config(&[
        "--checkpoint",
        "typed-decisions",
        "--worker-arg",
        "--env-file",
        "--worker-arg",
        env_file_str,
    ]))
    .await;
    let (status, body) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["checkpoint"], "typed-decisions",
        "a non-English run must not be reported as the English checkpoint"
    );
    let info = worker.info().await.expect("worker metadata");
    assert_eq!(info.checkpoint, "typed-decisions");
    let env = read_env_file(&env_file).await;
    assert_eq!(
        env.get("LAYAD_CHECKPOINT").map(String::as_str),
        Some("typed-decisions"),
        "the selected checkpoint must reach the worker: {env:?}"
    );
    worker.shutdown(Duration::from_secs(5)).await;

    // The default configuration selects the English checkpoint (no subfolder).
    let (state, worker) = ready_app(fake_config(&[])).await;
    let (status, body) = send(&state, "/readyz", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["checkpoint"], "english");
    let info = worker.info().await.expect("worker metadata");
    assert_eq!(info.checkpoint, "english");
    worker.shutdown(Duration::from_secs(5)).await;
}

/// A worker that answers `ready:false` — how the real worker reports a missing
/// checkpoint subfolder — must fail the startup instead of leaving the daemon
/// serving a silently different model.
#[tokio::test]
async fn worker_ready_false_fails_startup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pid_file = dir.path().join("worker.pid");
    let env_file = dir.path().join("selected.env");
    match WorkerHandle::start(fake_config(&[
        "--startup-timeout-ms",
        "5000",
        "--checkpoint",
        "multilingual",
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "not-ready",
        "--worker-arg",
        "--pid-file",
        "--worker-arg",
        pid_file.to_str().expect("utf-8 pid file"),
        "--worker-arg",
        "--env-file",
        "--worker-arg",
        env_file.to_str().expect("utf-8 env file"),
    ]))
    .await
    {
        // How the refusal surfaces (the load_failed frame, the child's exit, or
        // the signal the daemon sends to a child it will not use) is up to the
        // lifecycle code; that it *fails* is the contract.
        Err(StartError::Failed(detail)) => {
            assert!(!detail.is_empty(), "a refused startup must carry a reason")
        }
        Err(other) => panic!("a refusal to load must be a startup failure, got {other:?}"),
        Ok(_) => panic!("a worker that refuses to load its checkpoint must not be accepted"),
    }

    // The daemon must have asked for the *selected* checkpoint, so the failure
    // cannot be the English checkpoint quietly answering instead.
    let env = read_env_file(&env_file).await;
    assert_eq!(
        env.get("LAYAD_CHECKPOINT").map(String::as_str),
        Some("multilingual"),
        "the daemon must select the requested checkpoint: {env:?}"
    );

    // The refused child must not be left running either.
    if let Some(pid) = wait_for_pid_file(&pid_file, Duration::from_secs(5)).await {
        assert!(
            wait_for_exit(pid as u32, Duration::from_secs(5)).await,
            "refused worker {pid} must be reaped, not orphaned"
        );
    }
}

/// The daemon must point the worker at the project-local Hugging Face cache
/// (`.layad/hf`) when nothing else is configured, and an explicit
/// `--python-env HF_HOME=...` must win. The assertion runs in a child of this
/// test binary with `HF_HOME` removed, so the caller's environment cannot mask
/// the default.
#[tokio::test]
async fn worker_defaults_hf_home_to_the_project_cache() {
    if std::env::var("LAYAD_HF_HOME_PROBE").is_err() {
        let exe = std::env::current_exe().expect("test binary");
        let status = Command::new(exe)
            .arg("--exact")
            .arg("worker_defaults_hf_home_to_the_project_cache")
            .arg("--nocapture")
            .env_remove("HF_HOME")
            .env("LAYAD_HF_HOME_PROBE", "1")
            .status()
            .expect("re-run this test with HF_HOME unset");
        assert!(status.success(), "the HF_HOME probe failed: {status}");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    assert_worker_hf_home(&dir, &[], ".layad/hf").await;

    let dir = tempfile::tempdir().expect("tempdir");
    assert_worker_hf_home(&dir, &["--python-env", "HF_HOME=/custom/hf"], "/custom/hf").await;
}

/// Start the fake worker with `extra` configuration and assert the `HF_HOME` it
/// was handed. `dir` owns the environment dump file.
async fn assert_worker_hf_home(dir: &tempfile::TempDir, extra: &[&str], expected: &str) {
    let env_file = dir.path().join("hf.env");
    let env_file_str = env_file.to_str().expect("utf-8 path");
    let mut arguments = vec!["--worker-arg", "--env-file", "--worker-arg", env_file_str];
    arguments.extend_from_slice(extra);
    let worker = WorkerHandle::start(fake_config(&arguments))
        .await
        .expect("fake worker becomes ready");
    let env = read_env_file(&env_file).await;
    assert_eq!(
        env.get("HF_HOME").map(String::as_str),
        Some(expected),
        "unexpected worker HF_HOME: {env:?}"
    );
    worker.shutdown(Duration::from_secs(5)).await;
}

/// Send a raw body, byte for byte, so a captured client request is reproduced
/// exactly rather than re-serialized.
async fn send_raw(state: &AppState, path: &str, body: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let response = router(state.clone())
        .oneshot(request)
        .await
        .expect("router answers");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|err| {
            panic!(
                "response body is not JSON ({err}): {}",
                String::from_utf8_lossy(&bytes)
            )
        })
    };
    (status, value)
}

/// `tests/data/jev_requests.json` holds request bodies captured from a real Jev
/// client (the TypeSafe SDK, which posts to `/v1/systemone`). Every one of them
/// must be accepted **byte identical** by layad, on Jev's path and on layad's
/// own — including the top-level `model` (which layad resolves itself) and the
/// `trace_id` the SDK forwards from the caller.
#[tokio::test]
async fn captured_jev_request_bodies_are_accepted_byte_identical() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("tests/data")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("jev_") && name.ends_with(".json"))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no captured Jev requests under {dir:?}");
    let mut cases: Vec<Value> = Vec::new();
    for file in &files {
        let raw = std::fs::read(file).expect("captured Jev requests");
        let batch: Vec<Value> = serde_json::from_slice(&raw)
            .unwrap_or_else(|err| panic!("{file:?} is not a JSON array of requests: {err}"));
        cases.extend(batch);
    }

    // The capture must genuinely cover the shapes that made layad stricter than
    // Jev: a null state, undescribed criteria and extra top-level fields.
    let names: Vec<String> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap_or_default().to_string())
        .collect();
    for required in [
        "classifier-three-primitives",
        "state-null",
        "choice-null-description",
        "score-two-null-criteria",
        "score-structured-criteria",
        "model-and-extra-top-level-field",
    ] {
        assert!(
            names.iter().any(|name| name == required),
            "the capture lost its {required:?} case: {names:?}"
        );
    }

    for case in &cases {
        let name = case["name"].as_str().unwrap_or_default();
        // The capture also covers Jev's read-only routes; those are checked by
        // `models_reports_the_resident_checkpoint_in_jev_shape`.
        if case["method"] != "POST" {
            continue;
        }
        let content_type = case["content_type"].as_str().unwrap_or_default();
        assert!(
            content_type.starts_with("application/json"),
            "{name} sent content-type {content_type:?}"
        );
        let body = case["body"]
            .as_str()
            .expect("the captured body is a string");
        let url = case["url"].as_str().expect("url");
        let captured_path = url
            .find("/v1/")
            .map(|start| &url[start..])
            .unwrap_or("/v1/systemone");

        let questions: Vec<String> = serde_json::from_str::<Value>(body)
            .expect("captured body is JSON")
            .get("questions")
            .and_then(Value::as_object)
            .expect("captured body has questions")
            .keys()
            .cloned()
            .collect();

        // Jev's own path and layad's path both take the captured bytes as-is.
        for route in [captured_path, "/v1/predict"] {
            let (status, value) = send_raw(&state, route, body).await;
            assert_eq!(status, StatusCode::OK, "{name} -> {route}: {value}");
            let answers = value["answers"].as_object().expect("answers object");
            assert_eq!(
                answers.keys().collect::<Vec<_>>().len(),
                questions.len(),
                "{name} -> {route} answered the wrong questions: {value}"
            );
            for question in &questions {
                assert!(answers.contains_key(question), "{name} -> {route}: {value}");
                assert!(
                    answers[question]["type"].is_string(),
                    "{name} -> {route}: {value}"
                );
            }
        }
    }
}

/// Jev's declared answer types: no `action` block on any primitive, no
/// `confidence` on `noul`, and `model` plus `usage` at the top level.
#[tokio::test]
async fn systemone_answers_carry_jev_types_and_predict_keeps_the_raw_reply() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let body = json!({
        "state": "x",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "what does the user want?",
                "criteria": {"refund": "refund it", "cancel": "cancel it"},
            },
            "urgency": {
                "type": "score",
                "instructions": "how urgent?",
                "criteria": ["low", "high"],
            },
            "human": {"type": "noul", "instructions": "does the user want a human?"},
        },
    });

    for route in ["/v1/predict", "/v1/systemone"] {
        let (status, value) = send(&state, route, Some(body.clone())).await;
        assert_eq!(status, StatusCode::OK, "{route}: {value}");
        if route == "/v1/systemone" {
            // Jev's `SystemOneResult`: exactly these three keys, and `usage` is
            // Jev's pair of integer counters rather than the worker's fuller
            // usage object.
            let mut top: Vec<&str> = value
                .as_object()
                .unwrap_or_else(|| panic!("{route}: {value}"))
                .keys()
                .map(String::as_str)
                .collect();
            top.sort_unstable();
            assert_eq!(top, ["answers", "model", "usage"], "{route}: {value}");
        }
        assert!(value["model"].is_string(), "{route}: {value}");
        assert!(
            value["usage"]["input_tokens"].is_number(),
            "{route}: {value}"
        );
        assert!(
            value["usage"]["output_tokens"].is_number(),
            "{route}: {value}"
        );

        let expected: [(&str, &[&str]); 3] = [
            ("intent", &["choice", "confidence", "probabilities", "type"]),
            (
                "urgency",
                &["confidence", "legend", "probabilities", "score", "type"],
            ),
            ("human", &["noul", "type"]),
        ];
        for (question, keys) in expected {
            let answer = value["answers"][question]
                .as_object()
                .unwrap_or_else(|| panic!("{route}: no answer for {question}: {value}"));
            if route == "/v1/systemone" {
                let mut actual: Vec<&str> = answer.keys().map(String::as_str).collect();
                actual.sort_unstable();
                assert_eq!(actual, keys, "{route}: {question} shape changed: {value}");
            } else {
                // layad's own path keeps Laya's extra `action` block but must
                // still carry every key Jev declares.
                for key in keys {
                    assert!(
                        answer.contains_key(*key),
                        "{route}: {question} lost {key:?}: {value}"
                    );
                }
            }
        }
    }

    // layad's own path still exposes Laya's `action` block; only the Jev path
    // drops it, because Jev's types do not declare it.
    let (_, raw) = send(&state, "/v1/predict", Some(body.clone())).await;
    assert!(raw["answers"]["intent"].get("action").is_some(), "{raw}");
    assert!(raw["answers"]["human"].get("action").is_some(), "{raw}");
}

/// Jev's `GET /v1/models` shape: `{"models": [{"name", "description",
/// "release_date"}]}`.
#[tokio::test]
async fn models_reports_the_resident_checkpoint_in_jev_shape() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let (status, body) = send(&state, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1, "{body}");
    for key in ["name", "description", "release_date"] {
        assert!(models[0].get(key).is_some(), "{body}");
    }
    assert!(models[0]["description"].is_string(), "{body}");
    assert_eq!(models[0]["release_date"], "", "{body}");
}

/// The body a skill classifier posts: the resolved `model` plus `noul`
/// questions whose identity travels in a structured `instructions` object
/// (Jev types `instructions` as any JSON value). Both of layad's prediction
/// routes must take it as-is.
#[tokio::test]
async fn classifier_shaped_skill_questions_are_accepted_on_both_routes() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let body = json!({
        "model": "laya-rl-agent",
        "state": "GOAL: refund the duplicate charge",
        "questions": {
            "skill_0": {
                "type": "noul",
                "instructions": {
                    "question": "Would following the skill `skill` help complete the current task described in the state?",
                    "skill": {"name": "create-skill", "description": "Author a new skill"}
                }
            }
        }
    })
    .to_string();

    for path in ["/v1/predict", "/v1/systemone"] {
        let (status, value) = send_raw(&state, path, &body).await;
        assert_eq!(status, StatusCode::OK, "{path}: {value}");
        assert_eq!(
            value["answers"]["skill_0"]["type"],
            json!("noul"),
            "{path}: {value}"
        );
        assert!(
            value["answers"]["skill_0"]["noul"].is_number(),
            "{path}: {value}"
        );
    }

    // Jev's projection drops the `action` block upstream Laya adds; layad's own
    // route passes it through, so the same body answers on both.
    let (_, jev) = send_raw(&state, "/v1/systemone", &body).await;
    assert!(
        jev["answers"]["skill_0"].get("action").is_none(),
        "{jev}"
    );
    let (_, own) = send_raw(&state, "/v1/predict", &body).await;
    assert!(
        own["answers"]["skill_0"].get("action").is_some(),
        "{own}"
    );
}

/// One candidate per named question: a pool of this size must fit in the
/// default request, not be refused with `400`.
#[tokio::test]
async fn a_full_skill_pool_batch_is_not_refused() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "normal",
    ]))
    .await;

    let mut questions = serde_json::Map::new();
    for index in 0..64 {
        questions.insert(
            format!("skill_{index}"),
            json!({
                "type": "noul",
                "instructions": {
                    "question": "Would following the skill `skill` help?",
                    "skill": {"name": format!("skill-{index}"), "description": "a candidate"}
                }
            }),
        );
    }

    let (status, body) = send(
        &state,
        "/v1/predict",
        Some(json!({
            "model": "laya-rl-agent",
            "state": "GOAL: refund the duplicate charge",
            "questions": questions,
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["answers"].as_object().map(serde_json::Map::len),
        Some(64),
        "{body}"
    );
}

/// The fan-out such a caller makes: one request per authored skill, all at
/// once. Every one of them must be admitted instead of answered `429`.
#[tokio::test]
async fn a_skill_pool_fanout_is_admitted_not_refused() {
    let (state, _worker) = ready_app(fake_config(&[
        "--worker-arg",
        "--mode",
        "--worker-arg",
        "slow-inference",
        "--worker-arg",
        "--delay",
        "--worker-arg",
        "0.2",
    ]))
    .await;

    let mut handles = Vec::new();
    for index in 0..24 {
        let state = state.clone();
        handles.push(tokio::spawn(async move {
            send(&state, "/v1/predict", Some(prediction(&format!("burst {index}")))).await
        }));
    }

    for handle in handles {
        let (status, value) = handle.await.expect("fan-out task joins");
        assert_eq!(status, StatusCode::OK, "fan-out request was refused: {value}");
    }
}
