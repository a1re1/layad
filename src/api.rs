//! HTTP surface: routes, request validation, admission and the JSON error shape.
//!
//! Error bodies are always `{"error":{"code":"...","message":"..."}}` so
//! callers can branch on `code` without parsing prose. Request contents (state
//! text, questions) are never logged.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::config::Config;
use crate::worker::{WorkerError, WorkerHandle};

/// Readiness as seen by the HTTP layer.
///
/// The daemon binds its port before the model is loaded so `/healthz` can report
/// liveness during a slow (`~minutes`) cold start; readiness is tracked here and
/// reports accurately while the worker is still loading or has failed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Readiness {
    /// The resident model has not finished loading yet.
    #[default]
    Loading,
    /// The worker loaded the model, warmed up and is attached.
    Ready,
    /// Startup failed or the worker died; the daemon is failing closed.
    Failed(String),
    /// Graceful shutdown in progress.
    Stopping,
}

/// Shared application state. Cloning is cheap and the slot is filled in once the
/// worker reports readiness.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppSlot>,
}

struct AppSlot {
    readiness: std::sync::RwLock<Readiness>,
    worker: std::sync::RwLock<Option<WorkerHandle>>,
    config: Config,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        Self {
            inner: Arc::new(AppSlot {
                readiness: std::sync::RwLock::new(Readiness::Loading),
                worker: std::sync::RwLock::new(None),
                config,
            }),
        }
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    /// The resident worker, once it is ready.
    pub fn worker(&self) -> Option<WorkerHandle> {
        self.inner
            .worker
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn readiness(&self) -> Readiness {
        self.inner
            .readiness
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Publish a ready worker. Called exactly once, by `main`.
    pub fn attach(&self, worker: WorkerHandle) {
        *self
            .inner
            .worker
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(worker);
        self.set_readiness(Readiness::Ready);
    }

    pub fn set_readiness(&self, readiness: Readiness) {
        *self
            .inner
            .readiness
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = readiness;
    }
}

/// Build the router. Body limits, error shape and route set are all defined
/// here so `main` and the tests exercise the same surface.
pub fn router(state: AppState) -> Router {
    let max_body = state.config().max_body_bytes;
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/v1/predict", post(predict))
        .fallback(|| async { error_response(StatusCode::NOT_FOUND, "not_found", "unknown path") })
        .method_not_allowed_fallback(|| async {
            error_response(
                StatusCode::METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "wrong method",
            )
        })
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state)
}

/// Liveness: the daemon itself is up. Available during model loading.
async fn healthz() -> Response {
    json_response(
        StatusCode::OK,
        json!({
            "status": "ok",
            "service": "layad",
            "version": env!("CARGO_PKG_VERSION"),
            "pid": std::process::id(),
        }),
    )
}

/// Readiness: the resident worker loaded the model and warmed up.
async fn readyz(State(state): State<AppState>) -> Response {
    let worker = state.worker();
    let worker_state = match &worker {
        Some(worker) => Some(worker.state().await),
        None => None,
    };
    let info = match &worker {
        Some(worker) => worker.info().await,
        None => None,
    };
    match (worker_state, info) {
        (Some(crate::worker::State::Ready), None) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "ready": false,
                "reason": "loading",
                "message": "worker metadata unavailable",
            }),
        ),
        (Some(crate::worker::State::Ready), Some(info)) => json_response(
            StatusCode::OK,
            json!({
                "ready": true,
                "model": info.model,
                "checkpoint": info.checkpoint,
                "device": info.device,
                "requested_device": info.requested_device,
                "laya_version": info.laya_version,
                "worker_pid": info.pid,
                "worker_identity": info.identity(),
                "warmup": info.warmup,
            }),
        ),
        (None, _) | (Some(crate::worker::State::Starting), _) => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "ready": false,
                "reason": "loading",
                "message": "the resident model is still loading",
                "elapsed_state": readiness_label(&state.readiness()),
            }),
        ),
        (Some(worker_state), _) => {
            let reason = match worker_state {
                crate::worker::State::Failed(reason) => reason,
                crate::worker::State::ShuttingDown => "daemon is shutting down".to_string(),
                crate::worker::State::Pending(reason) => reason,
                _ => "worker not ready".to_string(),
            };
            json_response(
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"ready": false, "reason": "unavailable", "message": reason}),
            )
        }
    }
}

fn readiness_label(readiness: &Readiness) -> &'static str {
    match readiness {
        Readiness::Loading => "loading",
        Readiness::Ready => "ready",
        Readiness::Failed(_) => "failed",
        Readiness::Stopping => "stopping",
    }
}

/// Prediction request: `{"state": <string|object|array>, "questions": {...}}`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
}

/// Question types supported by the pinned upstream release.
const QUESTION_TYPES: [&str; 3] = ["choice", "score", "noul"];

async fn predict(State(state): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    let bytes = match body {
        Ok(bytes) => bytes,
        Err(rejection) => {
            let status = rejection.status();
            let code = if status == StatusCode::PAYLOAD_TOO_LARGE {
                "body_too_large"
            } else {
                "invalid_body"
            };
            let message = if code == "body_too_large" {
                format!(
                    "request body exceeds the {} byte limit",
                    state.config().max_body_bytes
                )
            } else {
                "could not read the request body".to_string()
            };
            return error_response(status, code, message);
        }
    };

    let request: PredictRequest = match serde_json::from_slice(&bytes) {
        Ok(request) => request,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                format!("request body is not a valid prediction request: {err}"),
            )
        }
    };

    if let Err(err) = validate_state(&request.state) {
        return error_response(StatusCode::BAD_REQUEST, "invalid_request", err);
    }
    if request.questions.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "questions must contain at least one named question".to_string(),
        );
    }
    if request.questions.len() > state.config().max_questions {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "questions contains {} entries, the limit is {}",
                request.questions.len(),
                state.config().max_questions
            ),
        );
    }
    for (name, question) in &request.questions {
        if let Err(err) = validate_question(name, question) {
            return error_response(StatusCode::BAD_REQUEST, "invalid_request", err);
        }
    }

    let worker = match state.worker() {
        Some(worker) => worker,
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable",
                "the resident model is still loading",
            )
        }
    };

    let params = json!({"state": request.state, "questions": request.questions});
    match worker.predict(params).await {
        Ok(result) => json_response(StatusCode::OK, result),
        Err(err) => worker_error_response(err),
    }
}

fn validate_state(state: &Value) -> Result<(), String> {
    match state {
        Value::String(text) => {
            if text.trim().is_empty() {
                Err("state must not be an empty string".to_string())
            } else {
                Ok(())
            }
        }
        Value::Object(map) => {
            if map.is_empty() {
                Err("state object must not be empty".to_string())
            } else {
                Ok(())
            }
        }
        Value::Array(items) => {
            if items.is_empty() {
                Err("state list must not be empty".to_string())
            } else {
                Ok(())
            }
        }
        other => Err(format!(
            "state must be a string, object or list, got {}",
            kind_of(other)
        )),
    }
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// Validate one named question. Mirrors the upstream `_to_internal` contract so
/// malformed requests are rejected with a clear message instead of a stack
/// trace from inside PyTorch.
fn validate_question(name: &str, question: &Value) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("question names must not be empty".to_string());
    }
    let object = match question {
        Value::Object(object) => object,
        other => {
            return Err(format!(
                "question {name:?} must be an object, got {}",
                kind_of(other)
            ))
        }
    };
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("question {name:?} is missing a string \"type\""))?;
    if !QUESTION_TYPES.contains(&kind) {
        return Err(format!(
            "question {name:?} has unsupported type {kind:?}, expected one of {}",
            QUESTION_TYPES.join(", ")
        ));
    }
    match object.get("instructions") {
        Some(Value::String(text)) if !text.trim().is_empty() => {}
        Some(Value::String(_)) => return Err(format!("question {name:?} has empty instructions")),
        Some(Value::Object(map)) if !map.is_empty() => {}
        Some(Value::Object(_)) => return Err(format!("question {name:?} has empty instructions")),
        Some(other) => {
            return Err(format!(
                "question {name:?} instructions must be a string or object, got {}",
                kind_of(other)
            ))
        }
        None => return Err(format!("question {name:?} is missing \"instructions\"")),
    }
    match (kind, object.get("criteria")) {
        ("choice", Some(Value::Object(map))) if !map.is_empty() => Ok(()),
        ("choice", Some(Value::Array(items))) if !items.is_empty() => {
            if items
                .iter()
                .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
            {
                Ok(())
            } else {
                Err(format!(
                    "question {name:?} criteria entries must be nonempty strings"
                ))
            }
        }
        ("choice", Some(Value::Object(_))) | ("choice", Some(Value::Array(_))) => Err(format!(
            "question {name:?} needs at least one choice criterion"
        )),
        ("choice", Some(other)) => Err(format!(
            "question {name:?} criteria must be an object or list of option names, got {}",
            kind_of(other)
        )),
        ("choice", None) => Err(format!(
            "question {name:?} of type choice is missing \"criteria\""
        )),
        ("score", Some(Value::Array(items))) if !items.is_empty() => {
            if items
                .iter()
                .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
            {
                Ok(())
            } else {
                Err(format!(
                    "question {name:?} criteria entries must be nonempty strings"
                ))
            }
        }
        ("score", Some(Value::Array(_))) => Err(format!(
            "question {name:?} needs at least one score criterion"
        )),
        ("score", Some(other)) => Err(format!(
            "question {name:?} criteria must be a list of ordered levels, got {}",
            kind_of(other)
        )),
        ("score", None) => Err(format!(
            "question {name:?} of type score is missing \"criteria\""
        )),
        // noul is a yes/no question: upstream ignores criteria entirely.
        ("noul", _) => Ok(()),
        _ => Err(format!("question {name:?} has unsupported type {kind:?}")),
    }
}

fn worker_error_response(err: WorkerError) -> Response {
    let status = match &err {
        WorkerError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        WorkerError::Overloaded => StatusCode::TOO_MANY_REQUESTS,
        WorkerError::Unavailable(_) | WorkerError::ShuttingDown => StatusCode::SERVICE_UNAVAILABLE,
        WorkerError::StartupFailed(_) | WorkerError::StartupTimeout(_) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        WorkerError::InferenceTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
        WorkerError::WorkerFailed(_) | WorkerError::OversizedResponse(_) => StatusCode::BAD_GATEWAY,
    };
    error_response(status, err.kind(), err.message())
}

/// Canonical JSON error body.
pub fn error_response(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    json_response(
        status,
        json!({"error": {"code": code, "message": message.into()}}),
    )
}

/// Serialize a JSON value with an explicit content type. Serialization of these
/// shapes cannot fail, but a panic would be worse than a 500.
fn json_response(status: StatusCode, value: Value) -> Response {
    match serde_json::to_vec(&value) {
        Ok(body) => (
            status,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            )],
            body,
        )
            .into_response(),
        Err(err) => {
            tracing::error!(error = %err, "failed to serialize a JSON response");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                )],
                br#"{"error":{"code":"internal_error","message":"failed to serialize response"}}"#
                    .to_vec(),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_accepts_string_object_and_list() {
        assert!(validate_state(&json!("a customer is angry")).is_ok());
        assert!(validate_state(&json!({"customer": "angry"})).is_ok());
        assert!(validate_state(&json!([{"customer": "angry"}, {"tier": "gold"}])).is_ok());
        assert!(validate_state(&json!("   ")).is_err());
        assert!(validate_state(&json!({})).is_err());
        assert!(validate_state(&json!([])).is_err());
        assert!(validate_state(&json!(7)).is_err());
        assert!(validate_state(&json!(null)).is_err());
    }

    #[test]
    fn choice_question_accepts_object_and_list_criteria() {
        let object = json!({"type": "choice", "instructions": "refund?", "criteria": {"yes": "full refund", "no": "deny"}});
        assert!(validate_question("refund", &object).is_ok());
        let list = json!({"type": "choice", "instructions": "refund?", "criteria": ["yes", "no"]});
        assert!(validate_question("refund", &list).is_ok());
    }

    #[test]
    fn score_and_noul_questions_validate() {
        let score = json!({"type": "score", "instructions": "how urgent?", "criteria": ["low", "medium", "high"]});
        assert!(validate_question("urgency", &score).is_ok());
        let noul = json!({"type": "noul", "instructions": "does the user want a human?"});
        assert!(validate_question("handoff", &noul).is_ok());
    }

    #[test]
    fn malformed_questions_are_rejected() {
        let cases: Vec<(Value, &str)> = vec![
            (json!("nope"), "must be an object"),
            (json!({"instructions": "x"}), "missing a string"),
            (
                json!({"type": "guess", "instructions": "x"}),
                "unsupported type",
            ),
            (
                json!({"type": "choice", "instructions": "x"}),
                "missing \"criteria\"",
            ),
            (
                json!({"type": "choice", "instructions": "x", "criteria": []}),
                "at least one",
            ),
            (
                json!({"type": "choice", "instructions": "x", "criteria": [""]}),
                "nonempty",
            ),
            (
                json!({"type": "score", "instructions": "x", "criteria": {"a": 1}}),
                "list of ordered levels",
            ),
            (
                json!({"type": "score", "instructions": "x"}),
                "missing \"criteria\"",
            ),
            (
                json!({"type": "noul", "instructions": 12}),
                "instructions must be",
            ),
            (
                json!({"type": "choice", "instructions": "", "criteria": ["a"]}),
                "empty instructions",
            ),
        ];
        for (value, needle) in cases {
            let err = validate_question("q", &value).unwrap_err();
            assert!(err.contains(needle), "expected {needle:?} in {err:?}");
        }
    }

    #[test]
    fn error_body_shape_is_stable() {
        let response = error_response(StatusCode::BAD_REQUEST, "invalid_request", "boom");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
