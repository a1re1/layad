//! layad — a local HTTP daemon that keeps one Laya decision model resident.
//!
//! The crate is split into four pieces:
//!
//! * [`config`] — CLI flags and the validated runtime configuration.
//! * [`worker`] — the resident Python child process and its NDJSON protocol.
//! * [`api`] — the Axum router, request validation and JSON error shape.
//! * `main` — process wiring, signal handling and fail-fast exit policy.
//!
//! The service is deliberately unauthenticated: it refuses to bind anywhere but
//! loopback, so only local callers can reach it.

pub mod api;
pub mod config;
pub mod worker;

/// Exit code used when the daemon fails closed and wants a supervisor
/// (launchd, or a human) to restart it.
pub const FAIL_FAST_EXIT_CODE: i32 = 70;

/// Grace period between noticing a fatal worker fault and exiting, long enough
/// for the in-flight response and the tracing output to reach the caller.
pub const FAIL_FAST_GRACE: std::time::Duration = std::time::Duration::from_millis(250);

/// Line the daemon logs once it is bound; scripts and tests read the port from
/// it (it also carries the `--bind` port `0` case).
pub fn listening_line(addr: std::net::SocketAddr) -> String {
    format!("layad listening on http://{addr}")
}

/// Schedule the fail-fast exit without blocking the current task.
pub fn schedule_fail_fast(reason: &str) {
    let reason = reason.to_string();
    tokio::spawn(async move {
        tokio::time::sleep(FAIL_FAST_GRACE).await;
        tracing::error!(reason = %reason, "fatal worker fault, exiting so a supervisor can restart layad");
        // Flush tracing before leaving.
        std::process::exit(FAIL_FAST_EXIT_CODE);
    });
}
