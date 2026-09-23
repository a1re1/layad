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
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(models))
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

/// Prediction request: `{"state": ..., "questions": {...}}`.
///
/// Deserialization is permissive on purpose: a Jev client always sends the
/// resolved `model`, and its SDK forwards any additional top-level property the
/// caller set. Those land in `extra` instead of failing the request. `model` is
/// accepted and ignored — layad serves the single checkpoint its worker was
/// started with.
#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
    /// Top-level properties layad does not use (`model`, `trace_id`, ...).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Question types supported by the pinned upstream release.
const QUESTION_TYPES: [&str; 3] = ["choice", "score", "noul"];

async fn predict(State(state): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    predict_impl(state, body, false).await
}

/// `POST /v1/systemone` — Jev's path for the same prediction.
///
/// The request is the one Jev accepts, byte for byte; the reply is projected
/// onto the key set Jev's types declare (see [`jev_result`]).
async fn systemone(State(state): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    predict_impl(state, body, true).await
}

async fn predict_impl(
    state: AppState,
    body: Result<Bytes, BytesRejection>,
    jev_shape: bool,
) -> Response {
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

    let kinds = question_kinds(&request.questions);
    let params = json!({"state": request.state, "questions": request.questions});
    match worker.predict(params).await {
        Ok(result) => json_response(
            StatusCode::OK,
            if jev_shape {
                jev_result(&kinds, result, state.config().model.clone())
            } else {
                result
            },
        ),
        Err(err) => worker_error_response(err),
    }
}

fn validate_state(state: &Value) -> Result<(), String> {
    match state {
        // Jev's `EntryType` includes `null`, and upstream `serialize_state`
        // renders it as the JSON literal `null`.
        Value::Null => Ok(()),
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
    // `instructions` must be present, but any JSON value is legal: Jev types it
    // as `EntryType` (string, object, array or null) and upstream renders
    // strings verbatim and everything else as compact JSON.
    if !object.contains_key("instructions") {
        return Err(format!("question {name:?} is missing \"instructions\""));
    }
    match (kind, object.get("criteria")) {
        // Criterion values are `EntryType`: a label may be undescribed (`null`
        // or `""`) or structured JSON, and upstream renders every value.
        ("choice", Some(Value::Object(map))) if !map.is_empty() => Ok(()),
        ("choice", Some(Value::Array(items))) if !items.is_empty() => Ok(()),
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
        ("score", Some(Value::Array(items))) if !items.is_empty() => Ok(()),
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
/// Question name -> declared type, captured before the request is forwarded, so
/// the reply can be projected onto Jev's per-question answer types.
fn question_kinds(questions: &Map<String, Value>) -> Vec<(String, String)> {
    questions
        .iter()
        .map(|(name, question)| {
            (
                name.clone(),
                question
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect()
}

/// Project a worker result onto Jev's `SystemOneResult` shape: exactly
/// `{model, answers, usage}`, in that key set, with `usage` carrying Jev's two
/// token counters.
///
/// The daemon's own `/v1/predict` reply is untouched — it keeps Laya's `action`
/// blocks and the worker's fuller `usage` — so anything the worker adds beyond
/// Jev's types is dropped here and only here. `model` falls back to the
/// checkpoint this daemon was started with, so the key is never missing.
fn jev_result(kinds: &[(String, String)], result: Value, fallback_model: String) -> Value {
    let object = result.as_object();
    let answers = object
        .and_then(|map| map.get("answers"))
        .and_then(Value::as_object);
    let model = object
        .and_then(|map| map.get("model"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or(fallback_model);
    let usage = object
        .and_then(|map| map.get("usage"))
        .and_then(Value::as_object);
    let counter = |key: &str| {
        usage
            .and_then(|map| map.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let projected = kinds
        .iter()
        .map(|(name, kind)| {
            let answer = answers
                .and_then(|map| map.get(name))
                .cloned()
                .unwrap_or(Value::Null);
            (name.clone(), jev_answer(kind, &answer))
        })
        .collect();
    json!({
        "model": model,
        "answers": Value::Object(projected),
        "usage": {
            "input_tokens": counter("input_tokens"),
            "output_tokens": counter("output_tokens"),
        },
    })
}

/// The exact answer key set Jev declares for one question type. Upstream Laya
/// adds an `action` block (and a `confidence` on `noul`) that Jev's types do not
/// have, so those are dropped. A key the worker omitted becomes `null` rather
/// than disappearing, so the shape does not depend on the worker's health.
fn jev_answer(kind: &str, answer: &Value) -> Value {
    let keys: &[&str] = match kind {
        "noul" => &["type", "noul"],
        "choice" => &["type", "choice", "confidence", "probabilities"],
        "score" => &["type", "score", "confidence", "legend", "probabilities"],
        _ => return answer.clone(),
    };
    let mut object = Map::new();
    for key in keys {
        object.insert(
            (*key).to_string(),
            answer.get(*key).cloned().unwrap_or(Value::Null),
        );
    }
    Value::Object(object)
}

/// `GET /v1/models` — Jev's model list, with the one checkpoint this daemon
/// keeps resident.
async fn models(State(state): State<AppState>) -> Response {
    let model = match state.worker() {
        Some(worker) => match worker.info().await {
            Some(info) => info.model,
            None => state.config().model.clone(),
        },
        None => state.config().model.clone(),
    };
    json_response(
        StatusCode::OK,
        json!({
            "models": [{
                "name": model,
                "description": "the checkpoint resident in this layad daemon",
                // A local checkpoint path has no release date to report.
                "release_date": "",
            }],
        }),
    )
}

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
        // Jev's `EntryType` allows a null state; upstream renders it as "null".
        assert!(validate_state(&json!(null)).is_ok());
    }

    #[test]
    fn choice_question_accepts_object_and_list_criteria() {
        let object = json!({"type": "choice", "instructions": "refund?", "criteria": {"yes": "full refund", "no": "deny"}});
        assert!(validate_question("refund", &object).is_ok());
        let list = json!({"type": "choice", "instructions": "refund?", "criteria": ["yes", "no"]});
        assert!(validate_question("refund", &list).is_ok());
        // Jev descriptions are `EntryType`: `null` leaves a label undescribed
        // and structured values are rendered as JSON by upstream.
        let undescribed = json!({"type": "choice", "instructions": "refund?", "criteria": {"refund": null, "cancel": "cancel it"}});
        assert!(validate_question("refund", &undescribed).is_ok());
        let structured = json!({"type": "choice", "instructions": "refund?", "criteria": [null, {"level": "high"}]});
        assert!(validate_question("refund", &structured).is_ok());
    }

    #[test]
    fn score_and_noul_questions_validate() {
        let score = json!({"type": "score", "instructions": "how urgent?", "criteria": ["low", "medium", "high"]});
        assert!(validate_question("urgency", &score).is_ok());
        let noul = json!({"type": "noul", "instructions": "does the user want a human?"});
        assert!(validate_question("handoff", &noul).is_ok());
        // Jev allows `instructions` to be any `EntryType` (here a list) and a
        // `noul` to carry `{true, false}` descriptions with null values.
        let structured = json!({"type": "score", "instructions": ["how urgent?"], "criteria": [null, {"level": "high"}]});
        assert!(validate_question("urgency", &structured).is_ok());
        let described = json!({"type": "noul", "instructions": {"q": "human?"}, "criteria": {"true": "yes", "false": null}});
        assert!(validate_question("handoff", &described).is_ok());
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
                json!({"type": "score", "instructions": "x", "criteria": []}),
                "at least one",
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
                json!({"type": "noul", "criteria": {"true": "yes"}}),
                "missing \"instructions\"",
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
