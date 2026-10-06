//! kiln evaluation API (06 §6): the Rust core shared by the `kiln` Python module (feature `python`) and `kiln eval`.

pub mod cache;
pub mod engine;
pub mod explain;
pub mod features;
pub mod fitness;
pub mod inputs;
pub mod options;
pub mod session;

#[cfg(feature = "python")]
mod python;
#[cfg(test)]
mod testutil;

pub use engine::{Engine, EngineRequest};
pub use inputs::{DesignInput, WorkloadInput, WorkloadSet};
pub use options::Options;
pub use session::{Session, SessionConfig};

/// Git hash of the engine crates at build time (`-dirty<fnv>` with local changes); part of every cache key.
pub const GIT_HASH: &str = env!("KILN_GIT_HASH");

/// `kiln.bench_export(suite)`: the bench manifest (`kiln.bench/1`) of a suite or one workload.
pub fn bench_export(suite: &str) -> Result<serde_json::Value, Vec<kiln_ir::common::Diagnostic>> {
    kiln_wl::bench::manifest(suite).map(|m| serde_json::to_value(&m).expect("manifest serializes"))
}
