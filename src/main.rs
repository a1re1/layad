//! Process entry point: configuration, tracing, worker startup, signals and the
//! fail-fast policy.

use std::process::ExitCode;

use clap::Parser;
use layad::api::{router, AppState};
use layad::config::{Cli, Config};
use layad::worker::{StartError, WorkerHandle};
use layad::{listening_line, schedule_fail_fast, FAIL_FAST_EXIT_CODE};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let config = match Config::try_from(cli) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("layad: {err}");
            return ExitCode::from(2);
        }
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&config.log)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    tracing::info!(
        model = %config.model,
        checkpoint = %config.checkpoint,
        device = %config.device,
        bind = %config.bind,
        "layad starting"
    );

    // Bind first so `/healthz` answers while the (slow) model load runs; `/readyz`
    // reports readiness accurately until the worker is attached.
    let listener = match tokio::net::TcpListener::bind(config.bind).await {
        Ok(listener) => listener,
        Err(err) => {
            tracing::error!(bind = %config.bind, error = %err, "cannot bind");
            return ExitCode::from(FAIL_FAST_EXIT_CODE as u8);
        }
    };
    let local_addr = listener.local_addr().unwrap_or(config.bind);
    println!("{}", listening_line(local_addr));

    let state = AppState::new(config.clone());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = axum::serve(listener, router(state.clone())).with_graceful_shutdown(async {
        let _ = stop_rx.await;
    });
    let mut server = tokio::spawn(async move { server.await });
    let signal = shutdown_signal();
    tokio::pin!(signal);

    let (loading, startup) = WorkerHandle::spawn(config.clone());
    let started = tokio::select! {
        result = startup => result,
        _ = &mut signal => {
            server.abort();
            loading.shutdown(config.shutdown_timeout).await;
            return ExitCode::SUCCESS;
        }
    };
    let worker = match started {
        Ok(worker) => worker,
        Err(StartError::Timeout(timeout)) => {
            tracing::error!(
                timeout_s = timeout.as_secs(),
                "worker did not finish loading the model; failing closed"
            );
            state.set_readiness(layad::api::Readiness::Failed(
                "model load timed out".to_string(),
            ));
            server.abort();
            return ExitCode::from(FAIL_FAST_EXIT_CODE as u8);
        }
        Err(StartError::Failed(detail)) => {
            tracing::error!(detail = %detail, "worker failed to start; failing closed");
            state.set_readiness(layad::api::Readiness::Failed(detail));
            server.abort();
            return ExitCode::from(FAIL_FAST_EXIT_CODE as u8);
        }
    };
    state.attach(worker.clone());

    // Fail closed: if the resident worker dies, stop serving so launchd (or the
    // operator) can restart with a fresh model instead of answering 5xx forever.
    let fail_fast = config.fail_fast;
    let watcher = worker.clone();
    let signal_state = state.clone();
    let watch = tokio::spawn(async move {
        if let Some(mut fatal) = watcher.take_fatal_receiver().await {
            if let Some(reason) = fatal.recv().await {
                state.set_readiness(layad::api::Readiness::Failed(reason.clone()));
                if fail_fast {
                    tracing::error!(reason = %reason, "resident worker died; exiting for a supervisor to restart");
                    watcher.kill_now().await;
                    schedule_fail_fast(&reason);
                } else {
                    tracing::error!(reason = %reason, "resident worker died; running degraded (fail-fast disabled)");
                }
            }
        }
    });

    let server_result = tokio::select! {
        result = &mut server => Some(result),
        _ = &mut signal => None,
    };
    watch.abort();
    signal_state.set_readiness(layad::api::Readiness::Stopping);
    let _ = stop_tx.send(());
    worker.shutdown(config.shutdown_timeout).await;
    let failed = match server_result {
        Some(Ok(Ok(()))) => false,
        Some(result) => {
            tracing::error!(?result, "http server stopped unexpectedly");
            true
        }
        None => {
            // A client that never finishes its HTTP body must not block exit.
            if tokio::time::timeout(config.shutdown_timeout, &mut server)
                .await
                .is_err()
            {
                server.abort();
            }
            false
        }
    };
    tracing::info!("layad stopped cleanly");
    if failed {
        ExitCode::from(FAIL_FAST_EXIT_CODE as u8)
    } else {
        ExitCode::SUCCESS
    }
}

/// SIGINT/SIGTERM handling. Both platforms get SIGINT; SIGTERM is only wired up
/// where it exists.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %err, "cannot listen for SIGINT");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(err) => {
                tracing::error!(error = %err, "cannot listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}
