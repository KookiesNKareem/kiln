//! Result and trace data model (spec 03 §10, 05 §3, 06 §4.5, §6.3).

pub mod analysis;
pub mod archive;
pub mod arrowx;
pub mod build;
pub mod calib_report;
pub mod check;
pub mod container;
pub mod interval;
pub mod layout;
pub mod meas;
pub mod perfetto;
pub mod provenance;
pub mod result;
pub mod sim;
pub mod trace;

pub use interval::{Corner, Interval, IntervalMethod};
pub use provenance::{Provenance, Tier, TraceLevel, TrustLevel};
pub use result::EvalResult;
pub use sim::SimResult;

pub const KILN_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const RESULT_SCHEMA: &str = "kiln.result/1";
pub const SIM_SCHEMA: &str = "kiln.sim/1";

/// Relative tolerance for sums that the spec calls exact (03 §10 attribution, 05 §3.8 energy).
pub const SUM_REL_TOL: f64 = 1e-9;
