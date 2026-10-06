//! Command tree of 06 §7 (flags owned by other sections are accepted and passed through).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "kiln",
    version,
    about = "kiln: accelerator-system simulator",
    propagate_version = true
)]
pub struct Cli {
    #[command(flatten)]
    pub global: Global,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Args, Debug, Clone)]
pub struct Global {
    /// Calibration set id or path.
    #[arg(long, global = true)]
    pub calib: Option<String>,
    #[arg(long, global = true)]
    pub cache_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    pub no_cache: bool,
    #[arg(long, global = true)]
    pub threads: Option<usize>,
    #[arg(long, global = true, value_enum, default_value_t = Format::Text)]
    pub format: Format,
    #[arg(long, global = true, value_enum, default_value_t = LogLevel::Warn)]
    pub log_level: LogLevel,
    /// Output path (JSON documents go here instead of stdout).
    #[arg(long, short = 'o', global = true)]
    pub out: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    Text,
    Json,
    Jsonl,
    Llm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

/// Flags owned by another spec section, accepted until that command is implemented.
#[derive(Args, Debug, Default)]
pub struct Rest {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub rest: Vec<String>,
}

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Design files (.json5/.json), JSON text, or reference names (a100_40gb, tpu_v5e, ...).
    pub designs: Vec<String>,
    /// `<preset>:<scenario>`, a suite (standard, legacy, smoke) or a workload file; default `standard`.
    #[arg(long)]
    pub workload: Option<String>,
    /// Scenario of `--workload` (a preset name or a workload file with several scenarios).
    #[arg(long)]
    pub scenario: Option<String>,
    #[arg(long, value_parser = ["A", "B", "a", "b", "cascade", "validate"])]
    pub tier: Option<String>,
    /// 01 §18.3 validation profile applied at S0.
    #[arg(long, default_value = "full", value_parser = ["full", "reference", "search", "stream_compat"])]
    pub profile: String,
    #[arg(long)]
    pub fitness: Option<String>,
    #[arg(long)]
    pub baseline: Option<String>,
    #[arg(long)]
    pub seeds: Option<String>,
    #[arg(long)]
    pub timeout: Option<f64>,
    #[arg(long, value_parser = ["none", "summary", "ops", "full"])]
    pub trace: Option<String>,
    /// Software stack both sides of `score` run under: a built-in id or recipe file (default `kiln_ideal`,
    /// hardware-only), or `own` (each design under its execution model's default). The realistic score
    /// (each side under its own default) is always reported next to it.
    #[arg(long)]
    pub stack: Option<String>,
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Evaluate one design.
    Eval(EvalArgs),
    /// Relative table vs the first design.
    Compare(EvalArgs),
    /// IR, workload, calibration or measurement validation only.
    Validate {
        file: PathBuf,
        #[arg(long, value_enum, default_value_t = Kind::Auto)]
        kind: Kind,
        #[arg(long, value_parser = ["full", "reference", "search", "stream_compat"])]
        profile: Option<String>,
        #[arg(long)]
        report: bool,
        #[arg(long)]
        allow_implausible: Vec<String>,
        #[arg(long)]
        deny_warnings: bool,
        #[arg(long)]
        strict_convert: bool,
    },
    /// Open the visualizer (05).
    Viz(Box<VizArgs>),
    /// LLM-readable summary of a result.
    Explain {
        result: PathBuf,
        #[arg(long)]
        max_items: Option<usize>,
    },
    #[command(subcommand)]
    Calibrate(CalibrateCmd),
    #[command(subcommand)]
    Bench(BenchCmd),
    /// L3 differential testing.
    DiffTest {
        #[arg(long, value_parser = ["stream", "zigzag"])]
        oracle: Option<String>,
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long)]
        oracle_python: Option<PathBuf>,
    },
    /// L4 agreement report.
    Agree {
        #[arg(long)]
        corpus: Option<PathBuf>,
    },
    #[command(subcommand)]
    Trust(TrustCmd),
    #[command(subcommand)]
    Corpus(CorpusCmd),
    /// Throughput and latency benchmark.
    Perf {
        #[arg(long)]
        corpus: Option<String>,
        #[arg(long)]
        workers: Option<String>,
        #[arg(long)]
        repeat: Option<u32>,
    },
    #[command(subcommand)]
    Trace(TraceCmd),
    /// Canonical form (01 §14.6).
    Fmt(Rest),
    /// Expansion report (01 §8.2).
    Expand(Rest),
    /// Design hash (01 §17).
    Hash(Rest),
    /// Schema migration (01 §19).
    Migrate(Rest),
    #[command(subcommand)]
    Phys(PhysCmd),
    /// Print a JSON Schema.
    Schema {
        #[arg(value_parser = ["hardware", "workload", "result", "calibration", "options", "measurement"])]
        kind: String,
    },
    #[command(subcommand)]
    Import(ImportCmd),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Kind {
    #[value(alias = "hw")]
    Hardware,
    Workload,
    Calibration,
    Measurement,
    /// From the `schema` / `kiln_workload` field, or `.json5` for hardware.
    Auto,
}

/// `kiln viz` (05 §8).
#[derive(Args, Debug)]
#[command(args_conflicts_with_subcommands = true)]
pub struct VizArgs {
    #[command(subcommand)]
    pub sub: Option<VizCmd>,
    /// A `.kiln` run (or `kiln.result/1` JSON with `sim`, or a design file for a structure-only floorplan).
    pub input: Option<PathBuf>,
    #[command(flatten)]
    pub data: VizData,
    /// Start on this view (floorplan, noc, timeline, roofline, bottleneck, compare, evolution, calibration).
    #[arg(long)]
    pub view: Option<String>,
    /// View state string (05 §4.2; JSON of the view spec).
    #[arg(long)]
    pub state: Option<String>,
    /// Reload the archive when its files change.
    #[arg(long)]
    pub watch: bool,
    /// Export the run to Perfetto and open it in ui.perfetto.dev via a localhost server.
    #[arg(long)]
    pub perfetto: bool,
}

/// Data sources shared by `kiln viz` and `kiln viz render`.
#[derive(Args, Debug, Default, Clone)]
pub struct VizData {
    /// Runs to compare (the first is the reference A).
    #[arg(long, num_args = 2.., value_name = "RUN")]
    pub compare: Vec<PathBuf>,
    /// Evolution archive directory (05 §3.9).
    #[arg(long)]
    pub archive: Option<PathBuf>,
    /// Calibration report: `calibration_report.arrow`, a directory holding one, or the JSON of
    /// `kiln calibrate report --format json`.
    #[arg(long)]
    pub calibration: Option<PathBuf>,
    /// Workload for a design input (default `llama3_8b:decode_b1`).
    #[arg(long)]
    pub workload: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum VizCmd {
    /// Headless PNG/SVG (05 §7.2).
    Render(RenderArgs),
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    pub input: Option<PathBuf>,
    #[command(flatten)]
    pub data: VizData,
    /// Views to render; several produce `<out stem>-<view>.<ext>` each.
    #[arg(long = "view", default_value = "floorplan")]
    pub views: Vec<String>,
    /// `WxH` in logical pixels.
    #[arg(long, default_value = "1600x1000")]
    pub size: String,
    /// Device pixels per logical pixel (PNG only).
    #[arg(long, default_value_t = 1.0)]
    pub scale: f32,
    #[arg(long, value_parser = ["light", "dark"], default_value = "light")]
    pub theme: String,
    /// Phase id (default: all phases).
    #[arg(long)]
    pub phase: Option<String>,
    /// Floorplan color mode: utilization, idle, energy, bytes, kind.
    #[arg(long)]
    pub color: Option<String>,
    /// Floorplan subtree (resource path).
    #[arg(long)]
    pub root: Option<String>,
    /// Roofline: one point per op family across layers.
    #[arg(long)]
    pub aggregate: bool,
    /// Calibration / roofline measured points: only this device.
    #[arg(long)]
    pub device: Option<String>,
    /// Evolution grid axes: descriptor indices `X,Y`.
    #[arg(long)]
    pub axes: Option<String>,
    /// View state string (JSON of the view spec); flags above override it.
    #[arg(long)]
    pub state: Option<String>,
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    pub file: PathBuf,
    /// Perfetto protobuf trace (05 §3.11).
    #[arg(long, group = "fmt")]
    pub perfetto: bool,
    /// Legacy Chrome JSON trace.
    #[arg(long, group = "fmt")]
    pub chrome_json: bool,
    /// One table as CSV.
    #[arg(long, group = "fmt", value_name = "TABLE")]
    pub csv: Option<String>,
    /// Export even above the Chrome JSON event limit.
    #[arg(long)]
    pub force: bool,
}

#[derive(Subcommand, Debug)]
pub enum CalibrateCmd {
    /// Create a deterministic split.
    Split {
        #[arg(long)]
        measurements: Option<String>,
        #[arg(long)]
        salt: Option<String>,
        #[arg(long)]
        holdout_phase: Option<String>,
    },
    /// Fit calibration sets on calib-micro records (stages P, L, D, U) and write them.
    Fit {
        /// `platform:a100_40gb`, `platform:tpu_v6e`, `generic-v1`, `generic-v2` or `all` (default).
        #[arg(long)]
        scope: Option<String>,
        #[arg(long)]
        measurements: Option<String>,
        #[arg(long)]
        split: Option<PathBuf>,
        /// Comma-separated stages to run (default `P,L,D,U`); skipped stages keep their priors.
        #[arg(long)]
        stages: Option<String>,
        /// Bootstrap resamples per stage (default 200).
        #[arg(long)]
        bootstrap: Option<usize>,
    },
    /// Re-derive the generic range policy (ranges for designs outside the fit devices) of generic sets from
    /// measured evidence, without refitting.
    Policy {
        /// `generic-v1`, `generic-v2` or `all` (default).
        #[arg(long)]
        set: Option<String>,
    },
    /// Test-split acceptance report; logs the test access in the set file.
    Report {
        #[arg(long)]
        set: Option<String>,
        #[arg(long)]
        devices: Option<String>,
        #[arg(long, value_parser = ["A", "B"])]
        tier: Option<String>,
        #[arg(long)]
        arrow: Option<PathBuf>,
        /// Do not append to the set's test_access_log.
        #[arg(long)]
        no_log: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum BenchCmd {
    /// Workload IR -> bench manifest.
    Export {
        /// One workload `<preset>:<scenario>`, e.g. `llama3_8b:decode_b1`.
        #[arg(long, required_unless_present = "suite", conflicts_with = "suite")]
        workload: Option<String>,
        #[arg(long, value_parser = ["gpu", "tpu", "nccl"])]
        target: Option<String>,
        #[arg(long)]
        sequence: bool,
        #[arg(long, value_parser = ["step", "layer"])]
        scope: Option<String>,
        /// `legacy` (the harness op list), `standard`, `smoke`, or a workload `<preset>:<scenario>`.
        #[arg(long)]
        suite: Option<String>,
    },
    /// Runner output -> measurement session (kiln.meas/1).
    Import(ImportArgs),
}

#[derive(Args, Debug)]
pub struct ImportArgs {
    pub file: PathBuf,
    /// Convert pre-kiln gpu_bench.py / tpu_bench.py output.
    #[arg(long)]
    pub legacy: bool,
    #[arg(long, value_parser = ["cuda", "jax", "nccl"])]
    pub runner: Option<String>,
    /// Op list the session ran with (default: ../oplist.json next to a gpu_bench file, if present).
    #[arg(long)]
    pub oplist: Option<PathBuf>,
    /// Also store into an append-only measurement tree (e.g. calibration/measurements) and its index.json.
    #[arg(long)]
    pub store: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum TrustCmd {
    Run(TrustArgs),
    Report(TrustArgs),
}

#[derive(Args, Debug)]
pub struct TrustArgs {
    #[arg(long)]
    pub release: bool,
    #[arg(long)]
    pub claim: Option<String>,
    #[arg(long)]
    pub parts: Option<String>,
    #[arg(long)]
    pub baseline: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum CorpusCmd {
    /// Golden regression.
    Run {
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long)]
        golden: Option<PathBuf>,
        #[arg(long)]
        tier: Option<String>,
        #[arg(long)]
        update: bool,
        #[arg(long)]
        reason: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum TraceCmd {
    /// Manifest and headline of a `.kiln` trace or a result JSON.
    Info {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Consistency checks; non-zero exit on violation.
    Validate {
        file: PathBuf,
    },
    Export(ExportArgs),
    Pack(Rest),
    Unpack(Rest),
    Upgrade(Rest),
    Recover(Rest),
}

#[derive(Subcommand, Debug)]
pub enum PhysCmd {
    ExportDef(Rest),
    ImportDef(Rest),
}

#[derive(Subcommand, Debug)]
pub enum ImportCmd {
    /// Legacy harness design -> hardware IR.
    HarnessDesign { file: PathBuf },
}

impl Cmd {
    /// Command path for messages, e.g. `trace export`.
    pub fn name(&self) -> &'static str {
        match self {
            Cmd::Eval(_) => "eval",
            Cmd::Compare(_) => "compare",
            Cmd::Validate { .. } => "validate",
            Cmd::Viz(v) if matches!(v.sub, Some(VizCmd::Render(_))) => "viz render",
            Cmd::Viz(_) => "viz",
            Cmd::Explain { .. } => "explain",
            Cmd::Calibrate(CalibrateCmd::Split { .. }) => "calibrate split",
            Cmd::Calibrate(CalibrateCmd::Fit { .. }) => "calibrate fit",
            Cmd::Calibrate(CalibrateCmd::Policy { .. }) => "calibrate policy",
            Cmd::Calibrate(CalibrateCmd::Report { .. }) => "calibrate report",
            Cmd::Bench(BenchCmd::Export { .. }) => "bench export",
            Cmd::Bench(BenchCmd::Import(_)) => "bench import",
            Cmd::DiffTest { .. } => "diff-test",
            Cmd::Agree { .. } => "agree",
            Cmd::Trust(TrustCmd::Run(_)) => "trust run",
            Cmd::Trust(TrustCmd::Report(_)) => "trust report",
            Cmd::Corpus(_) => "corpus run",
            Cmd::Perf { .. } => "perf",
            Cmd::Trace(t) => match t {
                TraceCmd::Info { .. } => "trace info",
                TraceCmd::Validate { .. } => "trace validate",
                TraceCmd::Export(_) => "trace export",
                TraceCmd::Pack(_) => "trace pack",
                TraceCmd::Unpack(_) => "trace unpack",
                TraceCmd::Upgrade(_) => "trace upgrade",
                TraceCmd::Recover(_) => "trace recover",
            },
            Cmd::Fmt(_) => "fmt",
            Cmd::Expand(_) => "expand",
            Cmd::Hash(_) => "hash",
            Cmd::Migrate(_) => "migrate",
            Cmd::Phys(PhysCmd::ExportDef(_)) => "phys export-def",
            Cmd::Phys(PhysCmd::ImportDef(_)) => "phys import-def",
            Cmd::Schema { .. } => "schema",
            Cmd::Import(_) => "import harness-design",
        }
    }

    /// Milestone (06 §10) that delivers the command.
    pub fn milestone(&self) -> &'static str {
        match self {
            Cmd::Eval(_)
            | Cmd::Compare(_)
            | Cmd::Validate { .. }
            | Cmd::Explain { .. }
            | Cmd::Schema { .. } => "M1",
            Cmd::Fmt(_)
            | Cmd::Expand(_)
            | Cmd::Hash(_)
            | Cmd::Migrate(_)
            | Cmd::Import(_)
            | Cmd::Bench(_) => "M0",
            Cmd::Calibrate(_) => "M2",
            Cmd::DiffTest { .. } | Cmd::Perf { .. } | Cmd::Corpus(_) => "M1",
            Cmd::Phys(_) => "M3",
            Cmd::Agree { .. } | Cmd::Trust(_) => "M4",
            Cmd::Viz(_) | Cmd::Trace(_) => "M4-M5",
        }
    }
}
