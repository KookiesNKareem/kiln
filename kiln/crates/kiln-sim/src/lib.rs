//! Engines (03 §4-§10). M1: Tier A (analytical) over the task graph shared with Tier B, whole-step mode,
//! parameter-corner intervals, invariants, `explain_run`, and the `evaluate` entry point.

pub mod calib;
pub mod engine;
pub mod evaluate;
pub mod explain;
pub mod invariants;
pub mod params;
pub mod result;
pub mod run;
pub mod stack;

pub use evaluate::{Prepared, check_priced, evaluate, evaluate_prepared, simulate_bench_op, simulate_member};
pub use explain::explain_run;
pub use calib::CalibSet;
pub use params::{ParamSet, SimParams};
pub use run::{PhaseRun, SimOptions, TIMEOUT_CODE, recost, simulate};
