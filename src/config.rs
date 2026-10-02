//! CLI surface and validated runtime configuration.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use clap::{Parser, ValueEnum};

/// Default Hugging Face repository: the native Apple MLX port of the Laya
/// checkpoint (`laya-mlx`). Upstream `convaiinnovations/laya` is still a valid
/// `--model`, but only the `torch` backend can load it.
pub const DEFAULT_MODEL: &str = "aac6fef/laya-mlx";

/// Which checkpoint inside the model repository to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Checkpoint {
    /// Repository root: the English checkpoint (default).
    English,
    /// `multilingual/` subfolder.
    Multilingual,
    /// `typed-decisions/` subfolder.
    TypedDecisions,
}

impl Checkpoint {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }

    /// Subfolder passed to `laya.load`, or `None` for the repository root.
    pub fn subfolder(self) -> Option<&'static str> {
        match self {
            Self::English => None,
            Self::Multilingual => Some("multilingual"),
            Self::TypedDecisions => Some("typed-decisions"),
        }
    }

    /// Directory name used by `scripts/setup.sh` for the project-local copy of
    /// the chosen checkpoint (`.layad/model/<name>`).
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }
}

impl std::fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Device requested for the resident model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Device {
    /// CPU inference (reliable everywhere, slowest).
    Cpu,
    /// Apple silicon GPU under the MLX backend (`--backend mlx`).
    Gpu,
    /// Apple Metal under PyTorch (`--backend torch`); PyTorch/mps aborts on
    /// this checkpoint's graph shapes, so MLX is what actually reaches the GPU.
    Mps,
    /// NVIDIA GPU (CUDA build of PyTorch required).
    Cuda,
    /// Let the backend pick (MLX: GPU then CPU; PyTorch: CUDA, then MPS, CPU).
    Auto,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Mps => "mps",
            Self::Cuda => "cuda",
            Self::Auto => "auto",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Inference backend for the resident model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    /// Native Apple MLX via `laya-mlx`. The default; Apple silicon only.
    Mlx,
    /// Upstream `laya` on PyTorch/transformers, for hosts without MLX.
    Torch,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Torch => "torch",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default ceiling for the resident worker's memory, in MiB. The worker caps
/// its Metal buffer cache and wired pool to stay inside it (see README
/// "Memory"); `0` leaves the inference stack's own limits alone.
pub const DEFAULT_MEMORY_MB: usize = 2048;

/// Environment variable the daemon passes to the worker for the budget.
pub const MEMORY_MB_ENV: &str = "LAYAD_MEMORY_MB";

/// Smallest budget that still fits the resident checkpoint and the stack.
const MIN_MEMORY_MB: usize = 512;

/// Default for `--memory-mb`, honouring the environment fallback.
///
/// This is the `default_value_t` of `--memory-mb`, so clap evaluates it for
/// every run. It cannot be paired with clap's `env` on the same argument, or
/// clap would parse the variable itself, and `LAYAD_MEMORY_MB=` (an empty
/// export, as launchd jobs and shells routinely produce) would abort startup
/// with "cannot parse integer from empty string". An unset, empty or
/// non-numeric value means the default here instead.
fn default_memory_mb() -> usize {
    std::env::var(MEMORY_MB_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MEMORY_MB)
}

fn default_python() -> PathBuf {
    let local = PathBuf::from(".venv/bin/python");
    if local.is_file() {
        local
    } else {
        PathBuf::from("python3")
    }
}

/// Default model: the project-local copy prepared by `scripts/setup.sh` when it
/// exists (so the daemon starts entirely offline), otherwise the upstream repo.
fn default_model() -> String {
    let local = PathBuf::from(LOCAL_MODEL_ROOT);
    if local.is_dir() {
        local.to_string_lossy().into_owned()
    } else {
        DEFAULT_MODEL.to_string()
    }
}

/// Root of the project-local model tree maintained by `scripts/setup.sh`.
pub const LOCAL_MODEL_ROOT: &str = ".layad/model";

/// Command line interface. Every flag also has a `LAYAD_*` environment
/// variable so launchd jobs and scripts can be declarative.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "layad",
    version,
    about = "Local HTTP daemon keeping one Laya decision model resident",
    long_about = "Serves Jev-inspired typed decisions (choice, score, noul) from a single \
                  resident Laya checkpoint. Binds to loopback only: the API is \
                  unauthenticated and must never be exposed to a network."
)]
pub struct Cli {
    /// Address to bind. Must be a loopback address.
    #[arg(long, env = "LAYAD_BIND", default_value = "127.0.0.1:8787")]
    pub bind: String,

    /// Python interpreter (or executable) that runs the worker.
    #[arg(long, env = "LAYAD_PYTHON", default_value_os_t = default_python())]
    pub python: PathBuf,

    /// Worker script passed to the interpreter. Empty string runs the program
    /// directly (used by the test fake worker).
    #[arg(long, env = "LAYAD_WORKER_SCRIPT", default_value = "python/worker.py")]
    pub worker_script: String,

    /// Extra argument appended to the worker command line (repeatable).
    /// Values may start with `-`, so worker flags can be passed as
    /// `--worker-arg --no-warmup`.
    #[arg(
        long = "worker-arg",
        env = "LAYAD_WORKER_ARG",
        allow_hyphen_values = true
    )]
    pub worker_args: Vec<String>,

    /// Hugging Face model id, or a local directory prepared by setup.sh.
    #[arg(long, env = "LAYAD_MODEL", default_value_t = default_model())]
    pub model: String,

    /// Checkpoint inside the model repository.
    #[arg(long, value_enum, env = "LAYAD_CHECKPOINT", default_value_t = Checkpoint::English)]
    pub checkpoint: Checkpoint,

    /// Inference backend for the resident model: `mlx` (native Apple MLX,
    /// the default) or `torch` (upstream laya on PyTorch).
    #[arg(long, value_enum, env = "LAYAD_BACKEND", default_value_t = Backend::Mlx)]
    pub backend: Backend,

    /// Device for the resident model (actual device is reported by /readyz).
    #[arg(long, value_enum, env = "LAYAD_DEVICE", default_value_t = Device::Auto)]
    pub device: Device,

    /// Memory ceiling for the resident worker, in MiB.
    ///
    /// The worker holds a checkpoint and a Metal buffer cache for the life of
    /// the daemon, and the stack caps neither by itself; this budget bounds
    /// both so a busy classifier cannot grow the process to tens of gigabytes.
    /// `0` disables layad's cap and leaves the stack's own limits alone.
    ///
    /// The `LAYAD_MEMORY_MB` fallback is read inside `default_memory_mb()`
    /// rather than through clap's `env`, so an unset, empty or non-numeric
    /// export means the default instead of a startup failure.
    #[arg(
        long,
        default_value_t = default_memory_mb(),
        value_name = "MIB"
    )]
    pub memory_mb: usize,

    /// Environment variable for the worker, `KEY=VALUE` (repeatable).
    #[arg(
        long = "python-env",
        env = "LAYAD_PYTHON_ENV",
        value_name = "KEY=VALUE"
    )]
    pub python_env: Vec<String>,

    /// Maximum request body size in bytes.
    #[arg(long, env = "LAYAD_MAX_BODY_BYTES", default_value_t = 1_048_576)]
    pub max_body_bytes: usize,

    /// Maximum number of named questions per request.
    ///
    /// A skill-classifier caller posts one question per candidate skill in a
    /// single request, so the default leaves room for a pool of a few dozen.
    #[arg(long, env = "LAYAD_MAX_QUESTIONS", default_value_t = 256)]
    pub max_questions: usize,

    /// Maximum number of predictions executing at once.
    ///
    /// This is admission control, not parallel inference: the worker answers
    /// one request at a time and the rest queue behind it. The default admits
    /// the one-request-per-authored-skill fan-out such a caller makes; a
    /// request beyond the limit is refused with `429` rather than queueing
    /// without bound.
    #[arg(long, env = "LAYAD_MAX_CONCURRENT", default_value_t = 64)]
    pub max_concurrent: usize,

    /// Maximum worker response frame size in bytes.
    #[arg(long, env = "LAYAD_MAX_RESPONSE_BYTES", default_value_t = 8_388_608)]
    pub max_response_bytes: usize,

    /// Deadline for the worker to load the model and warm up.
    #[arg(long, env = "LAYAD_STARTUP_TIMEOUT_MS", default_value_t = 600_000)]
    pub startup_timeout_ms: u64,

    /// Deadline for one prediction round trip.
    #[arg(long, env = "LAYAD_INFERENCE_TIMEOUT_MS", default_value_t = 60_000)]
    pub inference_timeout_ms: u64,

    /// Deadline for graceful worker shutdown before it is signalled.
    #[arg(long, env = "LAYAD_SHUTDOWN_TIMEOUT_MS", default_value_t = 10_000)]
    pub shutdown_timeout_ms: u64,

    /// Keep running (unready, failing closed) when the worker dies instead of
    /// exiting for a supervisor to restart.
    #[arg(long, env = "LAYAD_NO_FAIL_FAST")]
    pub no_fail_fast: bool,

    /// Tracing filter (for example `info`, `layad=debug`).
    #[arg(long, env = "LAYAD_LOG", default_value = "info")]
    pub log: String,
}

/// Validated configuration used by the rest of the daemon.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind: SocketAddr,
    pub python: PathBuf,
    pub worker_script: String,
    pub worker_args: Vec<String>,
    pub model: String,
    pub checkpoint: Checkpoint,
    pub backend: Backend,
    pub device: Device,
    pub memory_mb: usize,
    pub python_env: Vec<(String, String)>,
    pub max_body_bytes: usize,
    pub max_questions: usize,
    pub max_concurrent: usize,
    pub max_response_bytes: usize,
    pub startup_timeout: Duration,
    pub inference_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub fail_fast: bool,
    pub log: String,
}

impl Config {
    /// Reject configurations that would break the daemon's safety properties.
    pub fn validate(&self) -> Result<(), String> {
        if !self.bind.ip().is_loopback() {
            return Err(format!(
                "refusing to bind {}: layad is an unauthenticated local service and only serves loopback addresses",
                self.bind
            ));
        }
        if self.max_body_bytes == 0 {
            return Err("--max-body-bytes must be greater than zero".to_string());
        }
        if self.max_questions == 0 {
            return Err("--max-questions must be greater than zero".to_string());
        }
        if self.max_concurrent == 0 {
            return Err("--max-concurrent must be greater than zero".to_string());
        }
        if self.max_response_bytes == 0 {
            return Err("--max-response-bytes must be greater than zero".to_string());
        }
        // The budget is a ceiling, so an absurdly small one is a mistake: a
        // resident checkpoint alone does not fit in a few hundred MiB and the
        // worker would refuse to start. (0 is the documented "disabled".)
        if self.memory_mb != 0 && self.memory_mb < MIN_MEMORY_MB {
            return Err(format!(
                "--memory-mb {value} is below the {min} MiB a resident checkpoint needs; \
                 raise it or pass 0 to disable layad's memory cap",
                value = self.memory_mb,
                min = MIN_MEMORY_MB
            ));
        }
        for (name, value) in &self.python_env {
            if name.is_empty() || name.contains('=') {
                return Err(format!("--python-env expects KEY=VALUE, got {value:?}"));
            }
        }
        if self.python.as_os_str().is_empty() {
            return Err("--python must not be empty".to_string());
        }
        // A device name only means something under its own backend. Refusing the
        // pair here turns a typo into a startup error instead of a worker that
        // dies inside the inference stack after the model was downloaded.
        match (self.backend, self.device) {
            (Backend::Mlx, Device::Mps) | (Backend::Mlx, Device::Cuda) => {
                return Err(format!(
                    "--device {} is a PyTorch device and only works with --backend torch",
                    self.device
                ));
            }
            (Backend::Torch, Device::Gpu) => {
                return Err(
                    "--device gpu is the MLX device name and only works with --backend mlx \
                     (use --device cuda or --device mps for PyTorch)"
                        .to_string(),
                );
            }
            _ => {}
        }
        Ok(())
    }

    /// Whether `--model` points at a directory (project-local checkpoint tree or
    /// any other local path) instead of a Hugging Face repository id.
    pub fn model_is_local(&self) -> bool {
        std::path::Path::new(&self.model).is_dir()
    }

    /// Subfolder passed to `laya.load`, resolved for local and remote models.
    ///
    /// * remote repository: upstream layout (`multilingual/`, `typed-decisions/`,
    ///   English at the root);
    /// * project-local tree from `scripts/setup.sh`: `.layad/model/<checkpoint>`.
    pub fn subfolder(&self) -> Option<String> {
        self.checkpoint.subfolder().map(str::to_string)
    }

    /// Program and arguments for the resident worker child.
    pub fn worker_command(&self) -> (PathBuf, Vec<String>) {
        let mut args = Vec::new();
        if !self.worker_script.is_empty() {
            args.push(self.worker_script.clone());
        }
        args.push("--model".to_string());
        args.push(self.model.clone());
        args.push("--device".to_string());
        args.push(self.device.as_str().to_string());
        args.push("--backend".to_string());
        args.push(self.backend.as_str().to_string());
        if let Some(subfolder) = self.subfolder() {
            args.push("--subfolder".to_string());
            args.push(subfolder);
        }
        args.extend(self.worker_args.iter().cloned());
        (self.python.clone(), args)
    }
}

impl TryFrom<Cli> for Config {
    type Error = String;

    fn try_from(cli: Cli) -> Result<Self, Self::Error> {
        let bind = SocketAddr::from_str(&cli.bind)
            .map_err(|e| format!("invalid --bind {:?}: {e}", cli.bind))?;

        let mut python_env = Vec::with_capacity(cli.python_env.len());
        for item in &cli.python_env {
            let (name, value) = item
                .split_once('=')
                .ok_or_else(|| format!("--python-env expects KEY=VALUE, got {item:?}"))?;
            python_env.push((name.trim().to_string(), value.to_string()));
        }

        let config = Config {
            bind,
            python: cli.python,
            worker_script: cli.worker_script,
            worker_args: cli.worker_args,
            model: cli.model,
            checkpoint: cli.checkpoint,
            backend: cli.backend,
            device: cli.device,
            memory_mb: cli.memory_mb,
            python_env,
            max_body_bytes: cli.max_body_bytes,
            max_questions: cli.max_questions,
            max_concurrent: cli.max_concurrent,
            max_response_bytes: cli.max_response_bytes,
            startup_timeout: Duration::from_millis(cli.startup_timeout_ms),
            inference_timeout: Duration::from_millis(cli.inference_timeout_ms),
            shutdown_timeout: Duration::from_millis(cli.shutdown_timeout_ms),
            fail_fast: !cli.no_fail_fast,
            log: cli.log,
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(args: &[&str]) -> Cli {
        let mut full = vec!["layad"];
        full.extend_from_slice(args);
        Cli::parse_from(full)
    }

    #[test]
    fn defaults_are_loopback_mlx_english() {
        let config = Config::try_from(cli(&[])).expect("defaults valid");
        assert_eq!(config.bind.ip().to_string(), "127.0.0.1");
        assert_eq!(config.checkpoint, Checkpoint::English);
        assert_eq!(config.checkpoint.subfolder(), None);
        // Apple MLX is the default backend and picks the GPU itself.
        assert_eq!(config.backend, Backend::Mlx);
        assert_eq!(config.device, Device::Auto);
        // A resident checkpoint plus its cache fits in a couple of gigabytes.
        assert_eq!(config.memory_mb, DEFAULT_MEMORY_MB);
        assert_eq!(DEFAULT_MODEL, "aac6fef/laya-mlx");
        assert!(config.fail_fast);
        let (_, args) = config.worker_command();
        let idx = args
            .iter()
            .position(|a| a == "--backend")
            .expect("backend flag");
        assert_eq!(args[idx + 1], "mlx");
        // A classifier pool (one question per candidate, one request per
        // authored skill) must fit the defaults without raising flags.
        assert!(config.max_questions >= 64, "{}", config.max_questions);
        assert!(config.max_concurrent >= 24, "{}", config.max_concurrent);
        let (program, args) = config.worker_command();
        assert!(program.to_string_lossy().contains("python"));
        assert_eq!(args[0], "python/worker.py");
        assert!(args.contains(&"--model".to_string()));
        assert!(!args.contains(&"--subfolder".to_string()));
    }

    #[test]
    fn non_loopback_bind_is_rejected() {
        let err = Config::try_from(cli(&["--bind", "0.0.0.0:8787"])).unwrap_err();
        assert!(err.contains("loopback"), "{err}");
        let err = Config::try_from(cli(&["--bind", "192.168.1.10:8787"])).unwrap_err();
        assert!(err.contains("loopback"), "{err}");
    }

    #[test]
    fn multilingual_uses_subfolder() {
        // An explicit remote repository id keeps this test independent of whether
        // a project-local `.layad/model` tree happens to exist in the cwd.
        let config = Config::try_from(cli(&[
            "--checkpoint",
            "multilingual",
            "--model",
            "convaiinnovations/laya",
        ]))
        .unwrap();
        assert_eq!(config.checkpoint.subfolder(), Some("multilingual"));
        let (_, args) = config.worker_command();
        let idx = args
            .iter()
            .position(|a| a == "--subfolder")
            .expect("subfolder flag");
        assert_eq!(args[idx + 1], "multilingual");
    }

    #[test]
    fn torch_backend_is_selectable() {
        let config = Config::try_from(cli(&[
            "--backend",
            "torch",
            "--device",
            "cuda",
            "--model",
            "convaiinnovations/laya",
        ]))
        .unwrap();
        assert_eq!(config.backend, Backend::Torch);
        assert_eq!(config.device.as_str(), "cuda");
        let (_, args) = config.worker_command();
        let idx = args
            .iter()
            .position(|a| a == "--backend")
            .expect("backend flag");
        assert_eq!(args[idx + 1], "torch");
    }

    #[test]
    fn mismatched_backend_and_device_are_rejected() {
        // `mps`/`cuda` are torch devices and `gpu` is MLX's; each pair is refused
        // at startup rather than at model load.
        for args in [
            vec!["--backend", "mlx", "--device", "cuda"],
            vec!["--backend", "mlx", "--device", "mps"],
            vec!["--backend", "torch", "--device", "gpu"],
        ] {
            let err = Config::try_from(cli(&args)).unwrap_err();
            assert!(
                err.contains("--device") && err.contains("backend"),
                "{args:?}: {err}"
            );
        }
        // The per-backend device sets themselves stay accepted.
        for args in [
            vec!["--backend", "mlx", "--device", "gpu"],
            vec!["--backend", "mlx", "--device", "cpu"],
            vec!["--backend", "mlx", "--device", "auto"],
            vec!["--backend", "torch", "--device", "cuda"],
            vec!["--backend", "torch", "--device", "mps"],
            vec!["--backend", "torch", "--device", "auto"],
        ] {
            Config::try_from(cli(&args)).expect("valid backend/device pairing");
        }
    }

    #[test]
    fn memory_budget_is_configurable_and_validated() {
        let config = Config::try_from(cli(&["--memory-mb", "3072"])).unwrap();
        assert_eq!(config.memory_mb, 3072);
        // 0 is the documented "no cap".
        let config = Config::try_from(cli(&["--memory-mb", "0"])).unwrap();
        assert_eq!(config.memory_mb, 0);
        // A budget no resident checkpoint could fit in is refused.
        let err = Config::try_from(cli(&["--memory-mb", "64"])).unwrap_err();
        assert!(err.contains("--memory-mb"), "{err}");
        assert!(err.contains("0"), "{err}");
    }

    #[test]
    fn memory_default_ignores_an_empty_or_garbage_environment() {
        // `default_memory_mb` is what `--memory-mb` uses, so an exported but
        // empty (or non-numeric) `LAYAD_MEMORY_MB` must mean "the default", not
        // "abort startup". The end-to-end version of this lives in
        // `tests/integration.rs`, which spawns the real binary.
        assert_eq!(default_memory_mb(), DEFAULT_MEMORY_MB);
        // The default is a plain 2048 MiB ceiling the flag can override.
        assert_eq!(DEFAULT_MEMORY_MB, 2048);
        assert!(MIN_MEMORY_MB < DEFAULT_MEMORY_MB);
    }

    #[test]
    fn python_env_must_be_key_value() {
        let err = Config::try_from(cli(&["--python-env", "justaname"])).unwrap_err();
        assert!(err.contains("KEY=VALUE"), "{err}");
        let config = Config::try_from(cli(&[
            "--python-env",
            "HF_HUB_OFFLINE=1",
            "--python-env",
            "TRANSFORMERS_OFFLINE=1",
        ]))
        .unwrap();
        assert_eq!(config.python_env.len(), 2);
        assert_eq!(config.python_env[0].0, "HF_HUB_OFFLINE");
    }

    #[test]
    fn local_model_tree_uses_checkpoint_subdirectory() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("multilingual")).unwrap();
        let config = Config::try_from(cli(&[
            "--model",
            temp.path().to_str().unwrap(),
            "--checkpoint",
            "multilingual",
        ]))
        .unwrap();
        assert!(config.model_is_local());
        assert_eq!(config.subfolder().as_deref(), Some("multilingual"));
        let (_, args) = config.worker_command();
        let idx = args
            .iter()
            .position(|a| a == "--subfolder")
            .expect("subfolder flag");
        assert_eq!(args[idx + 1], "multilingual");

        // A local directory without the checkpoint subdirectory is used as-is.
        let plain = Config::try_from(cli(&[
            "--model",
            temp.path().to_str().unwrap(),
            "--checkpoint",
            "english",
        ]))
        .unwrap();
        assert_eq!(plain.subfolder(), None);
    }

    #[test]
    fn remote_model_keeps_upstream_layout() {
        let config = Config::try_from(cli(&[
            "--checkpoint",
            "typed-decisions",
            "--model",
            "convaiinnovations/laya",
        ]))
        .unwrap();
        assert!(!config.model_is_local());
        assert_eq!(config.subfolder().as_deref(), Some("typed-decisions"));
    }

    #[test]
    fn empty_worker_script_runs_program_directly() {
        let config = Config::try_from(cli(&[
            "--worker-script",
            "",
            "--worker-arg",
            "--mode",
            "--worker-arg",
            "normal",
        ]))
        .unwrap();
        let (_, args) = config.worker_command();
        assert_eq!(args[0], "--model");
        assert!(args.contains(&"--mode".to_string()));
    }
}
