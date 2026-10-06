//! Mapper (03 §3): serializable [`Mapping`], single-chip partitioning, tensor placement, transfer
//! derivation and routing, lowering to the shared task graph, and the seeded search interface.

pub mod cost;
pub mod geom;
pub mod heuristic;
pub mod hwview;
pub mod lower;
pub mod mapping;
pub mod program;
pub mod search;

pub use cost::{KilnCost, NestCost, NestQuery, RooflineCost, UnitCostModel};
pub use heuristic::{MapOptions, MapReport, heuristic};
pub use hwview::{HwView, Pool};
pub use lower::{Amount, TaskGraph, lower};
pub use mapping::Mapping;
pub use program::Program;
pub use search::{BeamSearch, MappingSearch, Move, Score};

pub type MapError = kiln_ir::common::Diagnostic;
