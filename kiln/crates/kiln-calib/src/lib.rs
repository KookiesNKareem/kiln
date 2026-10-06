//! Single-chip calibration (06 §3, M2): measurement records and splits, staged bounded fits of registered
//! mechanism parameters on calib-micro only, versioned calibration sets, and the held-out evaluation report.

pub mod campaign;
pub mod fit;
pub mod policy;
pub mod predict;
pub mod records;
pub mod report;
pub mod sets;
pub mod solve;
pub mod split;
