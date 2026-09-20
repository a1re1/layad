//! The resident Python child process and its bounded NDJSON protocol.
//!
//! Protocol (one JSON object per line, UTF-8, no embedded newlines):
//!
//! ```text
//! daemon -> worker  {"id":"1","method":"warmup","params":{...}}
//! daemon -> worker  {"id":"2","method":"predict","params":{"state":...,"questions":{...}}}
//! daemon -> worker  {"id":null,"method":"shutdown"}
//! worker -> daemon  {"id":"1","ok":true,"result":{...}}
//! worker -> daemon  {"id":"2","ok":false,"error":{"code":"invalid_request","message":"..."}}
//! ```
//!
//! Frames are read with `tokio::io::AsyncBufRead::fill_buf` plus an explicit
//! length cap, so a worker that floods stdout is killed rather than OOM-ing the
//! daemon. Every state change happens on one supervision task, so late replies
//! can never be mixed with a subsequent request: replies are matched by id and
//! anything else invalidates (and kills) the worker.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Notify;
use tokio::sync::{mpsc, oneshot, Mutex, Semaphore};

use crate::config::Config;

/// How often an idle session re-checks the child process.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Errors surfaced to the API layer. They map one-to-one onto JSON error codes.
#[derive(Debug, Clone)]
pub enum WorkerError {
    /// The worker has no live model (dead, starting, restarting, shutting down).
    Unavailable(String),
    // NOTE: variants are kept explicit rather than collapsed so `/readyz` and the
    // error mapping can distinguish a dead worker from an overloaded one.
    /// Startup deadline expired while loading the model.
    StartupTimeout(Duration),
    /// The worker answered the ready handshake but could not actually warm up.
    StartupFailed(String),
    /// A single prediction exceeded its deadline.
    InferenceTimeout(Duration),
    /// The request never reached the model because the worker broke first.
    WorkerFailed(String),
    /// The worker rejected the request itself.
    InvalidRequest(String),
    /// Response frame larger than the configured cap.
    OversizedResponse(usize),
    /// The daemon is shutting down and refuses new work.
    ShuttingDown,
    /// All admission slots are busy.
    Overloaded,
}

impl WorkerError {
    /// Stable machine-readable code used in JSON error bodies.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Unavailable(_) | Self::ShuttingDown => "unavailable",
            Self::StartupTimeout(_) | Self::StartupFailed(_) => "startup_failed",
            Self::InferenceTimeout(_) => "inference_timeout",
            Self::WorkerFailed(_) => "worker_failed",
            Self::InvalidRequest(_) => "invalid_request",
            Self::OversizedResponse(_) => "oversized_response",
            Self::Overloaded => "overloaded",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::Unavailable(d) => format!("worker unavailable: {d}"),
            Self::StartupTimeout(d) => {
                format!(
                    "worker did not finish loading the model within {}s",
                    d.as_secs()
                )
            }
            Self::StartupFailed(d) => format!("worker failed to start: {d}"),
            Self::InferenceTimeout(d) => {
                format!("inference exceeded the {}s deadline", d.as_secs())
            }
            Self::WorkerFailed(d) => format!("worker failed: {d}"),
            Self::InvalidRequest(d) => d.clone(),
            Self::OversizedResponse(n) => {
                format!("worker response exceeded {n} bytes")
            }
            Self::ShuttingDown => "daemon is shutting down".to_string(),
            Self::Overloaded => "too many predictions in flight, retry shortly".to_string(),
        }
    }

    /// Whether the failure means the resident worker is no longer trustworthy.
    pub fn invalidates_worker(&self) -> bool {
        matches!(
            self,
            Self::StartupTimeout(_)
                | Self::StartupFailed(_)
                | Self::InferenceTimeout(_)
                | Self::WorkerFailed(_)
                | Self::OversizedResponse(_)
        )
    }
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

/// Readiness and metadata reported by the worker's ready frame.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    pub pid: u32,
    pub model: String,
    pub checkpoint: String,
    pub device: String,
    pub requested_device: String,
    pub laya_version: String,
    pub warmup: Vec<String>,
}

impl Default for WorkerInfo {
    fn default() -> Self {
        Self {
            pid: std::process::id(),
            model: String::new(),
            checkpoint: "english".to_string(),
            device: "unknown".to_string(),
            requested_device: "unknown".to_string(),
            laya_version: String::new(),
            warmup: Vec::new(),
        }
    }
}

impl WorkerInfo {
    /// Parse the free-form ready frame, tolerating a worker that omits fields.
    pub fn from_ready(value: &Value) -> Self {
        let s = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let warmup = value
            .get("warmup")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Self {
            pid: value.get("pid").and_then(Value::as_u64).unwrap_or_default() as u32,
            model: s("model"),
            checkpoint: if s("checkpoint").is_empty() {
                "english".to_string()
            } else {
                s("checkpoint")
            },
            device: {
                let d = s("device");
                if d.is_empty() {
                    "unknown".to_string()
                } else {
                    d
                }
            },
            requested_device: s("requested_device"),
            laya_version: s("laya_version"),
            warmup,
        }
    }

    /// Stable identity of the resident model process, used by the smoke test.
    pub fn identity(&self) -> String {
        format!("{}:{}:{}", self.pid, self.device, self.checkpoint)
    }
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    code: Option<String>,
    message: Option<String>,
    /// The worker marked this failure as fatal: its resident state can no longer
    /// be trusted, so the daemon tears it down instead of trusting what follows
    /// on the stream.
    #[serde(default)]
    fatal: bool,
}

#[derive(Debug)]
enum Reply {
    Ok(Value),
    Err(ErrorBody),
}

/// Why the initial model load did not complete.
#[derive(Debug, Clone)]
pub enum StartError {
    Timeout(Duration),
    Failed(String),
}

impl StartError {
    fn into_worker_error(self) -> WorkerError {
        match self {
            Self::Timeout(d) => WorkerError::StartupTimeout(d),
            Self::Failed(d) => WorkerError::StartupFailed(d),
        }
    }
}

/// One-shot handshake channel that can be settled exactly once, from any task.
struct ReadySlot(std::sync::Mutex<Option<oneshot::Sender<Result<WorkerInfo, StartError>>>>);

impl ReadySlot {
    fn new(sender: oneshot::Sender<Result<WorkerInfo, StartError>>) -> Self {
        Self(std::sync::Mutex::new(Some(sender)))
    }

    /// Deliver the outcome to `WorkerHandle::start`, exactly once.
    fn settle(&self, result: Result<WorkerInfo, StartError>) {
        let sender = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(sender) = sender {
            let _ = sender.send(result);
        }
    }

    fn is_canceled(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .is_some_and(oneshot::Sender::is_closed)
    }

    /// Whether the daemon is still waiting for the ready handshake.
    fn is_pending(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }
}

/// Handle held by the HTTP layer. Cheap to clone and safe to share.
#[derive(Debug, Clone)]
pub struct WorkerHandle {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    config: Config,
    tx: mpsc::Sender<Job>,
    state: Mutex<State>,
    admission: Arc<Semaphore>,
    info: Mutex<Option<WorkerInfo>>,
    pid: Mutex<Option<u32>>,
    fatal_tx: mpsc::UnboundedSender<String>,
    fatal_rx: Mutex<Option<mpsc::UnboundedReceiver<String>>>,
    done: Arc<Notify>,
    force_stop: Notify,
}

/// Lifecycle of the resident worker, as reported by `/readyz`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    Starting,
    Pending(String),
    Ready,
    Failed(String),
    ShuttingDown,
}

#[derive(Debug)]
struct Job {
    params: Value,
    reply: oneshot::Sender<Result<Value, WorkerError>>,
}

impl WorkerHandle {
    /// Spawn the supervision task and wait for the worker's ready frame.
    pub async fn start(config: Config) -> Result<Self, StartError> {
        let (_, ready) = Self::spawn(config);
        ready.await
    }

    /// Return a handle immediately so startup can be interrupted and reaped.
    pub fn spawn(
        config: Config,
    ) -> (
        Self,
        impl std::future::Future<Output = Result<Self, StartError>>,
    ) {
        let (tx, rx) = mpsc::channel::<Job>(64);
        let (fatal_tx, fatal_rx) = mpsc::unbounded_channel::<String>();
        let admission = Arc::new(Semaphore::new(config.max_concurrent));
        let handle = Self {
            inner: Arc::new(Inner {
                config: config.clone(),
                tx,
                state: Mutex::new(State::Starting),
                admission,
                info: Mutex::new(None),
                pid: Mutex::new(None),
                fatal_tx,
                fatal_rx: Mutex::new(Some(fatal_rx)),
                done: Arc::new(Notify::new()),
                force_stop: Notify::new(),
            }),
        };
        let (ready_tx, ready_rx) = oneshot::channel::<Result<WorkerInfo, StartError>>();
        let supervisor = handle.clone();
        tokio::spawn(async move {
            supervisor.supervise(rx, ready_tx).await;
            supervisor.inner.done.notify_one();
        });
        let waiting = handle.clone();
        let ready = async move {
            let result = match tokio::time::timeout(config.startup_timeout, ready_rx).await {
                Ok(Ok(Ok(info))) => {
                    tracing::debug!(identity = %info.identity(), "worker ready handshake complete");
                    return Ok(waiting);
                }
                Ok(Ok(Err(err))) => err,
                Ok(Err(_)) => {
                    StartError::Failed("worker supervision task ended before readiness".to_string())
                }
                Err(_) => StartError::Timeout(config.startup_timeout),
            };
            waiting.shutdown(config.shutdown_timeout).await;
            Err(result)
        };
        (handle, ready)
    }

    /// Current readiness for `/readyz`.
    pub async fn state(&self) -> State {
        self.inner.state.lock().await.clone()
    }

    pub fn max_concurrent(&self) -> usize {
        self.inner.config.max_concurrent
    }

    /// Metadata reported by the worker's ready frame, once it has loaded.
    pub async fn info(&self) -> Option<WorkerInfo> {
        self.inner.info.lock().await.clone()
    }

    /// Receive fatal worker faults (used by `main` to fail fast).
    pub async fn take_fatal_receiver(&self) -> Option<mpsc::UnboundedReceiver<String>> {
        self.inner.fatal_rx.lock().await.take()
    }

    /// Stop admitting work, ask the child to exit, and wait for the supervisor
    /// (bounded by `grace`).
    pub async fn shutdown(&self, grace: Duration) {
        *self.inner.state.lock().await = State::ShuttingDown;
        let (reply, _rx) = oneshot::channel();
        let _ = self.inner.tx.try_send(Job {
            params: Value::Null,
            reply,
        });
        if tokio::time::timeout(grace, self.inner.done.notified())
            .await
            .is_err()
        {
            tracing::warn!(
                grace_ms = grace.as_millis() as u64,
                "worker did not stop within the shutdown deadline, killing it"
            );
            self.kill_now().await;
        }
    }

    /// SIGKILL the child immediately. Used on shutdown deadlines and before a
    /// fail-fast exit so no orphan worker is ever left behind.
    pub async fn kill_now(&self) {
        let pid = *self.inner.pid.lock().await;
        if let Some(pid) = pid {
            tracing::warn!(pid, "killing resident python worker");
            // Only the supervisor owns the Child and may signal it. A cached PID
            // could be reused between observing it here and delivering a signal.
            self.inner.force_stop.notify_one();
            // Wait for the supervisor to reap rather than guessing with a sleep.
            let _ = tokio::time::timeout(
                self.inner.config.shutdown_timeout,
                self.inner.done.notified(),
            )
            .await;
        }
    }

    /// Queue one prediction and wait for its reply.
    pub async fn predict(&self, params: Value) -> Result<Value, WorkerError> {
        let _permit = self
            .inner
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| WorkerError::Overloaded)?;

        match self.state().await {
            State::Ready => {}
            State::ShuttingDown => return Err(WorkerError::ShuttingDown),
            State::Failed(reason) => return Err(WorkerError::Unavailable(reason)),
            State::Starting => {
                return Err(WorkerError::Unavailable("worker is still loading".into()))
            }
            State::Pending(reason) => return Err(WorkerError::Unavailable(reason)),
        }

        let (reply_tx, reply_rx) = oneshot::channel();
        self.inner
            .tx
            .send(Job {
                params,
                reply: reply_tx,
            })
            .await
            .map_err(|_| WorkerError::ShuttingDown)?;

        // Cancelling this future (client disconnect) drops the receiver, which the
        // supervisor notices via `reply.is_closed()` and treats as worker-invalidating.
        reply_rx.await.map_err(|_| WorkerError::ShuttingDown)?
    }

    /// Fail closed: mark the worker untrusted and clear admission.
    pub async fn mark_failed(&self, reason: &str) {
        let mut state = self.inner.state.lock().await;
        if *state != State::ShuttingDown {
            *state = State::Failed(reason.to_string());
        }
        drop(state);
        *self.inner.info.lock().await = None;
        *self.inner.pid.lock().await = None;
        let _ = self.inner.fatal_tx.send(reason.to_string());
    }

    async fn supervise(
        &self,
        mut rx: mpsc::Receiver<Job>,
        ready: oneshot::Sender<Result<WorkerInfo, StartError>>,
    ) {
        let ready = ReadySlot::new(ready);
        let startup_deadline = tokio::time::Instant::now() + self.inner.config.startup_timeout;
        let mut job = None;

        // Exactly one worker lifetime: every path through the body below either
        // shuts the child down or fails closed, so this is not a retry loop.
        let starting = matches!(
            *self.inner.state.lock().await,
            State::Starting | State::Pending(_)
        );
        if starting {
            let spawned = spawn_child(&self.inner.config).await;
            let (mut child, stdin, stdout) = match spawned {
                Ok(parts) => parts,
                Err(err) => {
                    let fatal = StartError::Failed(err);
                    *self.inner.state.lock().await =
                        State::Failed(fatal.clone().into_worker_error().message());
                    ready.settle(Err(fatal));
                    return;
                }
            };

            let pid = child.id();
            *self.inner.pid.lock().await = pid;
            tracing::info!(pid, "python worker started");
            let outcome = self
                .run_session(Session {
                    child: &mut child,
                    stdin,
                    stdout,
                    ready: &ready,
                    startup_deadline,
                    pending_job: &mut job,
                    rx: &mut rx,
                })
                .await;

            match outcome {
                SessionOutcome::Shutdown | SessionOutcome::Kill => {
                    if matches!(outcome, SessionOutcome::Kill) {
                        let _ = child.start_kill();
                    }
                    ready.settle(Err(StartError::Failed("worker stopped".into())));
                    let _ = shutdown_child(&mut child, self.inner.config.shutdown_timeout).await;
                    *self.inner.pid.lock().await = None;
                    *self.inner.info.lock().await = None;
                    tracing::info!("python worker stopped");
                }
                SessionOutcome::Restart(reason) => {
                    tracing::warn!(reason = %reason, "python worker failed, killing and reaping it");
                    let status =
                        shutdown_child(&mut child, self.inner.config.shutdown_timeout).await;
                    let detail = match status {
                        Ok(status) => format!("worker {status}"),
                        Err(err) => err,
                    };
                    if ready.is_pending() {
                        // Startup failed: fail closed, never retry a broken model.
                        ready.settle(Err(StartError::Failed(detail.clone())));
                    } else {
                        tracing::error!(detail = %detail, "resident worker lost; failing closed");
                    }
                    self.mark_failed(&detail).await;
                    // Nothing will ever answer these; do not leave callers hanging.
                    while let Ok(job) = rx.try_recv() {
                        let _ = job
                            .reply
                            .send(Err(WorkerError::WorkerFailed(detail.clone())));
                    }
                }
            }
        }
    }

    /// One worker lifetime. Returns when the worker must be stopped (daemon
    /// shutdown) or has been invalidated.
    ///
    /// Any restart reason is published as a failed state before returning, so a
    /// request arriving while the supervisor is still killing the child is
    /// answered `unavailable` instead of being queued behind a dying worker.
    async fn run_session(&self, session: Session<'_>) -> SessionOutcome {
        let outcome = tokio::select! {
            outcome = self.run_session_inner(session) => outcome,
            _ = self.inner.force_stop.notified() => SessionOutcome::Kill,
        };
        if let SessionOutcome::Restart(reason) = &outcome {
            let mut state = self.inner.state.lock().await;
            if !matches!(*state, State::Failed(_) | State::ShuttingDown) {
                *state = State::Failed(reason.clone());
            }
        }
        outcome
    }

    async fn run_session_inner(&self, session: Session<'_>) -> SessionOutcome {
        let Session {
            child,
            stdin,
            stdout,
            ready,
            startup_deadline,
            pending_job,
            rx,
        } = session;
        let mut stdin = stdin;
        let max_response_bytes = self.inner.config.max_response_bytes;
        let mut lines = tokio::io::BufReader::new(stdout);
        let mut frame_buffer = Vec::new();
        let mut inflight: Option<(String, oneshot::Sender<Result<Value, WorkerError>>)> = None;
        let mut next_id: u64 = 0;
        let mut inference_deadline = None;

        loop {
            if ready.is_canceled() || matches!(*self.inner.state.lock().await, State::ShuttingDown)
            {
                return SessionOutcome::Shutdown;
            }
            // 0. Re-check the startup deadline every iteration so a worker that
            //    never announces readiness cannot hold the daemon open.
            if ready.is_pending() && tokio::time::Instant::now() >= startup_deadline {
                let timeout = self.inner.config.startup_timeout;
                ready.settle(Err(StartError::Timeout(timeout)));
                return SessionOutcome::Restart(format!(
                    "worker did not become ready within {}s",
                    timeout.as_secs()
                ));
            }

            // 1. Take the next job: an already-queued one, or the channel. While
            //    idle we watch the child too, so an idle worker death is noticed
            //    immediately instead of on the next request.
            if inflight.is_none() {
                let job = match pending_job.take() {
                    Some(job) => Some(job),
                    None => tokio::select! {
                        job = rx.recv() => match job {
                            Some(job) => Some(job),
                            // A closed channel means the daemon is going away.
                            None => return SessionOutcome::Shutdown,
                        },
                        status = child.wait() => {
                            let detail = match status {
                                Ok(status) => format!("worker exited with {status}"),
                                Err(err) => format!("cannot wait on worker: {err}"),
                            };
                            if ready.is_pending() {
                                ready.settle(Err(StartError::Failed(detail.clone())));
                            }
                            return SessionOutcome::Restart(detail);
                        }
                        // While the ready handshake is outstanding the session
                        // must read frames even though no request is queued;
                        // otherwise a worker that announces readiness before the
                        // first prediction would never be recognised.
                        frame = read_frame_buffered(&mut lines, max_response_bytes, &mut frame_buffer) => {
                            match frame {
                                Ok(Frame::Line(line)) => {
                                    if !ready.is_pending() {
                                        return SessionOutcome::Restart("unsolicited worker output".into());
                                    }
                                    let value: Value = match serde_json::from_str(&line) {
                                        Ok(value) => value,
                                        Err(err) => return SessionOutcome::Restart(format!("worker sent invalid JSON: {err}")),
                                    };
                                    if value.get("id").map(|id| !id.is_null()).unwrap_or(false) {
                                        return SessionOutcome::Restart(
                                            "worker replied although no request was outstanding".to_string(),
                                        );
                                    }
                                    match parse_ready(&value) {
                                        Ok(info) => {
                                            tracing::info!(
                                                pid = info.pid,
                                                device = %info.device,
                                                requested = %info.requested_device,
                                                checkpoint = %info.checkpoint,
                                                laya = %info.laya_version,
                                                "worker ready"
                                            );
                                            *self.inner.state.lock().await = State::Ready;
                                            *self.inner.info.lock().await = Some(info.clone());
                                            ready.settle(Ok(info));
                                            continue;
                                        }
                                        Err(err) => return SessionOutcome::Restart(err),
                                    }
                                }
                                Ok(Frame::Eof) => {
                                    if ready.is_pending() {
                                        ready.settle(Err(StartError::Failed(
                                            "worker closed stdout (EOF)".to_string(),
                                        )));
                                    }
                                    return SessionOutcome::Restart(
                                        "worker closed stdout (EOF)".to_string(),
                                    );
                                }
                                Ok(Frame::Oversized(bytes)) => {
                                    return SessionOutcome::Restart(format!(
                                        "worker response exceeded {max_response_bytes} bytes (at least {bytes} bytes read)"
                                    ));
                                }
                                Err(err) => {
                                    return SessionOutcome::Restart(format!(
                                        "worker read error: {err}"
                                    ));
                                }
                            }
                        }
                        _ = tokio::time::sleep(POLL_INTERVAL) => continue,
                    },
                };
                if let Some(job) = job {
                    if matches!(*self.inner.state.lock().await, State::ShuttingDown) {
                        let _ = job.reply.send(Err(WorkerError::ShuttingDown));
                        return SessionOutcome::Shutdown;
                    }
                    next_id += 1;
                    let id = next_id.to_string();
                    let frame = json!({"id": id, "method": "predict", "params": job.params});
                    let deadline =
                        tokio::time::Instant::now() + self.inner.config.inference_timeout;
                    inference_deadline = Some(deadline);
                    match tokio::time::timeout_at(deadline, write_frame(&mut stdin, &frame)).await {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            let _ = job
                                .reply
                                .send(Err(WorkerError::WorkerFailed("worker write failed".into())));
                            return SessionOutcome::Restart(format!(
                                "cannot write to worker: {err}"
                            ));
                        }
                        Err(_) => {
                            let _ = job.reply.send(Err(WorkerError::InferenceTimeout(
                                self.inner.config.inference_timeout,
                            )));
                            return SessionOutcome::Restart(
                                "worker input deadline exceeded".into(),
                            );
                        }
                    }
                    inflight = Some((id, job.reply));
                }
            }

            // 2. Wait for one frame, a canceled caller, or the inference deadline.
            let waited = match inference_deadline {
                Some(deadline) => {
                    let mut poll = tokio::time::interval(POLL_INTERVAL);
                    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    loop {
                        tokio::select! {
                            frame = read_frame_buffered(&mut lines, max_response_bytes, &mut frame_buffer) => break frame,
                            _ = poll.tick() => {
                                if matches!(*self.inner.state.lock().await, State::ShuttingDown) {
                                    return SessionOutcome::Shutdown;
                                }
                                // A caller that went away mid-request leaves a
                                // reply channel nobody can receive on. The request
                                // is still in flight, so the worker is torn down
                                // rather than left desynchronised.
                                if inflight
                                    .as_ref()
                                    .is_some_and(|(_, reply)| reply.is_closed())
                                {
                                    let id = inflight
                                        .as_ref()
                                        .map(|(id, _)| id.clone())
                                        .unwrap_or_default();
                                    tracing::warn!(
                                        request_id = %id,
                                        "caller canceled; killing the worker to keep the protocol in sync"
                                    );
                                    return SessionOutcome::Restart(format!(
                                        "request {id} was canceled by the caller"
                                    ));
                                }
                            }
                            _ = tokio::time::sleep_until(deadline) => {
                                let timeout = self.inner.config.inference_timeout;
                                if let Some((_, reply)) = inflight.take() {
                                    let _ = reply.send(Err(WorkerError::InferenceTimeout(timeout)));
                                }
                                return SessionOutcome::Restart(format!(
                                    "worker did not answer within {}s",
                                    timeout.as_secs()
                                ));
                            }
                        }
                    }
                }
                None => {
                    read_frame_buffered(&mut lines, max_response_bytes, &mut frame_buffer).await
                }
            };

            match waited {
                Ok(Frame::Line(line)) => {
                    let value: Value = match serde_json::from_str(&line) {
                        Ok(value) => value,
                        Err(err) => {
                            self.fail_inflight(
                                &mut inflight,
                                &format!("worker sent invalid JSON: {err}"),
                            );
                            return SessionOutcome::Restart(format!(
                                "malformed worker frame: {err}"
                            ));
                        }
                    };
                    let id = value.get("id").and_then(Value::as_str).map(str::to_string);
                    if ready.is_pending() && id.is_none() {
                        match parse_ready(&value) {
                            Ok(info) => {
                                tracing::info!(
                                    pid = info.pid,
                                    device = %info.device,
                                    requested = %info.requested_device,
                                    checkpoint = %info.checkpoint,
                                    laya = %info.laya_version,
                                    "worker ready"
                                );
                                *self.inner.state.lock().await = State::Ready;
                                *self.inner.info.lock().await = Some(info.clone());
                                ready.settle(Ok(info));
                            }
                            Err(err) => return SessionOutcome::Restart(err),
                        }
                        continue;
                    }
                    let Some((expected, reply)) = inflight.take() else {
                        return SessionOutcome::Restart(
                            "worker replied although no request was outstanding".to_string(),
                        );
                    };
                    let parsed = match parse_reply(&value) {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            let _ = reply.send(Err(WorkerError::WorkerFailed(err.clone())));
                            return SessionOutcome::Restart(err);
                        }
                    };
                    // A worker that marks its own error fatal is unusable: the
                    // daemon fails closed instead of trusting the stream further.
                    let parsed_fatal = matches!(&parsed.body, Reply::Err(body) if body.fatal);
                    // The protocol is strictly single-flight: a reply for any
                    // other id means the stream is desynchronised, so the worker
                    // is invalidated rather than trusted.
                    if parsed.id != expected {
                        let got = parsed.id;
                        let _ = reply.send(Err(WorkerError::WorkerFailed(format!(
                            "response id {got} does not match request id {expected}"
                        ))));
                        return SessionOutcome::Restart(format!(
                            "worker response id {got} does not match request id {expected}"
                        ));
                    }
                    match parsed.body {
                        Reply::Ok(result) => {
                            let _ = reply.send(Ok(result));
                        }
                        Reply::Err(body) => {
                            let message = body
                                .message
                                .clone()
                                .unwrap_or_else(|| "worker rejected the request".to_string());
                            let error = match body.code.as_deref() {
                                Some("invalid_request") | Some("validation") => {
                                    WorkerError::InvalidRequest(message)
                                }
                                _ => WorkerError::WorkerFailed(message),
                            };
                            let _ = reply.send(Err(error));
                        }
                    }
                    if parsed_fatal {
                        return SessionOutcome::Restart(
                            "worker reported a fatal error".to_string(),
                        );
                    }
                }
                Ok(Frame::Eof) => {
                    self.fail_inflight(&mut inflight, "worker closed its output stream");
                    return SessionOutcome::Restart("worker closed stdout (EOF)".to_string());
                }
                Ok(Frame::Oversized(bytes)) => {
                    let cap = self.inner.config.max_response_bytes;
                    // The caller gets the dedicated oversized code, not a generic
                    // worker failure, so a runaway frame is diagnosable.
                    if let Some((_, reply)) = inflight.take() {
                        let _ = reply.send(Err(WorkerError::OversizedResponse(cap)));
                    }
                    return SessionOutcome::Restart(format!(
                        "worker response exceeded {cap} bytes (at least {bytes} bytes read)"
                    ));
                }
                Err(err) => {
                    self.fail_inflight(&mut inflight, &format!("worker read error: {err}"));
                    return SessionOutcome::Restart(format!("worker read error: {err}"));
                }
            }

            // 3. Detect a cancelled caller: if the HTTP task went away, the reply
            //    channel is closed. The request is still in flight, so the whole
            //    worker is torn down rather than desynchronised.
            // Cancellation is detected while the reply is still in flight (see
            // the wait above), so there is nothing left to handle here.
        }
    }

    fn fail_inflight(
        &self,
        inflight: &mut Option<(String, oneshot::Sender<Result<Value, WorkerError>>)>,
        reason: &str,
    ) {
        if let Some((_, reply)) = inflight.take() {
            let _ = reply.send(Err(WorkerError::WorkerFailed(reason.to_string())));
        }
    }
}

/// Everything one worker lifetime needs, bundled so the session functions keep
/// a readable signature instead of taking eight positional arguments.
struct Session<'a> {
    child: &'a mut Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    ready: &'a ReadySlot,
    startup_deadline: tokio::time::Instant,
    pending_job: &'a mut Option<Job>,
    rx: &'a mut mpsc::Receiver<Job>,
}

#[derive(Debug)]
enum SessionOutcome {
    Restart(String),
    Shutdown,
    Kill,
}

/// Ready frame validation shared by the supervisor and the Python tests.
pub fn parse_ready(value: &Value) -> Result<WorkerInfo, String> {
    match value.get("ready").and_then(Value::as_bool) {
        Some(true) => {}
        Some(false) => {
            let detail = value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("worker reported that it could not start");
            return Err(detail.to_string());
        }
        None => return Err("worker frame is missing ready=true".to_string()),
    }
    if value.get("id").map(|id| !id.is_null()).unwrap_or(false) {
        return Err("ready frame must not carry a request id".to_string());
    }
    if !value
        .get("pid")
        .and_then(Value::as_u64)
        .is_some_and(|pid| pid > 0 && pid <= u32::MAX as u64)
    {
        return Err("ready frame is missing a valid worker pid".to_string());
    }
    let info = WorkerInfo::from_ready(value);
    if info.device.is_empty() || info.device == "unknown" {
        return Err("ready frame is missing the actual device".to_string());
    }
    Ok(info)
}

/// Validate one reply frame without running a child process.
fn parse_reply(value: &Value) -> Result<ParsedReply, String> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| "reply frame has no string id".to_string())?;
    match value.get("ok").and_then(Value::as_bool) {
        Some(true) => {
            let result = value
                .get("result")
                .cloned()
                .ok_or_else(|| "ok reply has no result".to_string())?;
            Ok(ParsedReply {
                id: id.to_string(),
                body: Reply::Ok(result),
            })
        }
        Some(false) => {
            let mut body = value
                .get("error")
                .cloned()
                .map(serde_json::from_value::<ErrorBody>)
                .transpose()
                .map_err(|e| format!("malformed error object: {e}"))?
                .unwrap_or(ErrorBody {
                    code: None,
                    message: None,
                    fatal: false,
                });
            body.fatal |= value.get("fatal").and_then(Value::as_bool).unwrap_or(false);
            Ok(ParsedReply {
                id: id.to_string(),
                body: Reply::Err(body),
            })
        }
        None => Err("reply frame has no boolean ok field".to_string()),
    }
}

#[derive(Debug)]
struct ParsedReply {
    id: String,
    body: Reply,
}

enum Frame {
    Line(String),
    Eof,
    Oversized(usize),
}

/// Read one newline-terminated frame, refusing anything longer than the cap.
#[cfg(test)]
async fn read_frame<R: tokio::io::AsyncBufRead + Unpin>(
    lines: &mut R,
    max_bytes: usize,
) -> std::io::Result<Frame> {
    read_frame_buffered(lines, max_bytes, &mut Vec::new()).await
}

// The caller retains consumed bytes when select! cancels a pending read.
async fn read_frame_buffered<R: tokio::io::AsyncBufRead + Unpin>(
    lines: &mut R,
    max_bytes: usize,
    buf: &mut Vec<u8>,
) -> std::io::Result<Frame> {
    loop {
        let available = match lines.fill_buf().await {
            Ok(available) => available,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        };
        if available.is_empty() {
            if buf.is_empty() {
                return Ok(Frame::Eof);
            }
            break;
        }
        let newline = available.iter().position(|b| *b == b'\n');
        let take = newline.map(|pos| pos + 1).unwrap_or(available.len());
        if buf.len() + take > max_bytes {
            return Ok(Frame::Oversized(buf.len() + take));
        }
        buf.extend_from_slice(&available[..take]);
        lines.consume(take);
        if newline.is_some() {
            break;
        }
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
    }
    if buf.last() == Some(&b'\r') {
        buf.pop();
    }
    let line = String::from_utf8(std::mem::take(buf))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(Frame::Line(line))
}

async fn write_frame(stdin: &mut ChildStdin, value: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    line.push(b'\n');
    stdin.write_all(&line).await?;
    stdin.flush().await
}

/// Environment the worker inherits. `PATH` and `HOME` are needed to find the
/// interpreter and the Hugging Face cache even under launchd.
fn worker_env(config: &Config) -> HashMap<String, String> {
    let mut env = HashMap::new();
    env.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());
    env.insert("PYTHONIOENCODING".to_string(), "utf-8".to_string());
    env.insert("LAYAD_MODEL".to_string(), config.model.clone());
    env.insert(
        "LAYAD_DEVICE".to_string(),
        config.device.as_str().to_string(),
    );
    env.insert(
        "LAYAD_CHECKPOINT".to_string(),
        config.checkpoint.as_str().to_string(),
    );
    // Other variables (including PATH and offline flags) are already inherited.
    // Re-setting PATH makes std::process fall back from posix_spawn to fork for
    // bare executable names; macOS frameworks are not safe after threaded fork.
    env.insert(
        "HF_HOME".to_string(),
        std::env::var("HF_HOME").unwrap_or_else(|_| ".layad/hf".to_string()),
    );
    for (key, value) in &config.python_env {
        env.insert(key.clone(), value.clone());
    }
    env
}

async fn spawn_child(config: &Config) -> Result<(Child, ChildStdin, ChildStdout), String> {
    let (program, args) = config.worker_command();
    let mut command = Command::new(&program);
    command.args(&args);
    command.envs(worker_env(config));
    command.stdin(std::process::Stdio::piped());
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::inherit());
    // The child is killed explicitly by the supervisor so the daemon can report
    // bounded shutdown; `kill_on_drop` is the last-resort net.
    command.kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot start worker {}: {e}", program.display()))?;
    let stdin = child.stdin.take().ok_or("worker stdin unavailable")?;
    let stdout = child.stdout.take().ok_or("worker stdout unavailable")?;
    Ok((child, stdin, stdout))
}

/// Ask the child to exit, then escalate to SIGKILL, and always reap it.
/// Returns the exit status description when it was observed.
async fn shutdown_child(child: &mut Child, grace: Duration) -> Result<String, String> {
    let pid = child.id();
    if let Some(pid) = pid {
        // SIGTERM lets the worker run its own cleanup (closing the model).
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }

    match tokio::time::timeout(grace, child.wait()).await {
        Ok(Ok(status)) => return Ok(status.to_string()),
        Ok(Err(err)) => return Err(format!("cannot reap worker: {err}")),
        Err(_) => {}
    }

    tracing::warn!(
        grace_ms = grace.as_millis() as u64,
        "worker did not exit after SIGTERM, sending SIGKILL"
    );
    let _ = child.start_kill();
    match tokio::time::timeout(grace.max(Duration::from_millis(200)), child.wait()).await {
        Ok(Ok(status)) => Ok(status.to_string()),
        Ok(Err(err)) => Err(format!("cannot reap worker: {err}")),
        Err(_) => Err("worker did not exit after SIGKILL".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;
    use tokio::io::BufReader;

    #[tokio::test]
    async fn read_frame_reads_one_line() {
        let data = b"{\"a\":1}\n{\"b\":2}\n";
        let mut reader = BufReader::new(&data[..]);
        match read_frame(&mut reader, 1024).await.unwrap() {
            Frame::Line(line) => assert_eq!(line, "{\"a\":1}"),
            _ => panic!("expected a line"),
        }
        match read_frame(&mut reader, 1024).await.unwrap() {
            Frame::Line(line) => assert_eq!(line, "{\"b\":2}"),
            _ => panic!("expected a line"),
        }
        assert!(matches!(
            read_frame(&mut reader, 1024).await.unwrap(),
            Frame::Eof
        ));
    }

    #[tokio::test]
    async fn read_frame_rejects_oversized_frames() {
        let data = vec![b'x'; 256];
        let mut reader = BufReader::new(&data[..]);
        match read_frame(&mut reader, 128).await.unwrap() {
            Frame::Oversized(bytes) => assert!(bytes > 128),
            _ => panic!("expected an oversized frame"),
        }
    }

    #[test]
    fn parse_ready_requires_ready_device() {
        let ok = json!({"ready": true, "pid": 7, "device": "cpu", "checkpoint": "english"});
        let info = parse_ready(&ok).unwrap();
        assert_eq!(info.pid, 7);
        assert_eq!(info.identity(), "7:cpu:english");
        // Missing the actual device: the worker must not be trusted as ready.
        assert!(parse_ready(&json!({"ready": true})).is_err());
        assert!(parse_ready(&json!({"ready": true, "device": "unknown"})).is_err());
        // A worker that reports a failed load is surfaced with its message.
        let failed =
            parse_ready(&json!({"ready": false, "error": {"message": "no weights"}})).unwrap_err();
        assert!(failed.contains("no weights"), "{failed}");
        assert!(parse_ready(&json!({"ready": false, "device": "cpu"})).is_err());
        assert!(parse_ready(&json!({"ready": true, "device": "cpu", "id": "1"})).is_err());
    }

    #[test]
    fn parse_reply_accepts_and_rejects() {
        let ok = json!({"id": "3", "ok": true, "result": {"answers": {}}});
        assert_eq!(parse_reply(&ok).unwrap().id, "3");
        let err =
            json!({"id": "4", "ok": false, "error": {"code": "invalid_request", "message": "bad"}});
        let parsed = parse_reply(&err).unwrap();
        assert_eq!(parsed.id, "4");
        match parsed.body {
            Reply::Err(body) => assert_eq!(body.code.as_deref(), Some("invalid_request")),
            Reply::Ok(_) => panic!("expected an error frame"),
        }
        assert!(parse_reply(&json!({"id": "5", "ok": false})).unwrap().id == "5");
        assert!(parse_reply(&json!({"ok": true, "result": 1})).is_err());
        assert!(parse_reply(&json!({"id": "1", "result": 1})).is_err());
        assert!(parse_reply(&json!({"id": "1", "ok": true})).is_err());
    }

    #[test]
    fn error_codes_are_stable() {
        assert_eq!(WorkerError::Unavailable("x".into()).kind(), "unavailable");
        assert_eq!(
            WorkerError::StartupTimeout(Duration::from_secs(1)).kind(),
            "startup_failed"
        );
        assert_eq!(
            WorkerError::InferenceTimeout(Duration::from_secs(1)).kind(),
            "inference_timeout"
        );
        assert!(WorkerError::WorkerFailed("x".into()).invalidates_worker());
        assert!(!WorkerError::InvalidRequest("x".into()).invalidates_worker());
    }

    #[test]
    fn worker_env_maps_config() {
        let cli = crate::config::Cli::parse_from([
            "layad",
            "--python-env",
            "HF_HUB_OFFLINE=1",
            "--device",
            "mps",
        ]);
        let config = crate::config::Config::try_from(cli).unwrap();
        let env = worker_env(&config);
        assert_eq!(env.get("HF_HUB_OFFLINE").map(String::as_str), Some("1"));
        assert_eq!(env.get("LAYAD_DEVICE").map(String::as_str), Some("mps"));
        assert_eq!(env.get("PYTHONUNBUFFERED").map(String::as_str), Some("1"));
    }

    #[tokio::test]
    async fn start_reports_timeout_when_worker_never_announces_ready() {
        let script = std::env::current_dir()
            .unwrap()
            .join("tests/data/fake_worker.py");
        let cli = crate::config::Cli::parse_from([
            "layad",
            "--python",
            "python3",
            "--worker-script",
            script.to_str().unwrap(),
            "--worker-arg",
            "--mode",
            "--worker-arg",
            "silent-startup",
            "--startup-timeout-ms",
            "300",
        ]);
        let config = crate::config::Config::try_from(cli).unwrap();
        match WorkerHandle::start(config).await {
            Err(StartError::Timeout(d)) => assert_eq!(d.as_millis(), 300),
            other => panic!("expected a startup timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn canceled_reply_channel_is_detected() {
        // Dropping the receiver simulates an HTTP client that went away.
        let (tx, rx) = oneshot::channel::<Result<Value, WorkerError>>();
        drop(rx);
        assert!(tx.is_closed());
    }
}
