//! UNCALIBRATED placeholder energies used until kiln-phys (04) supplies per-access and per-MAC energies.
//! Values are order-of-magnitude assumptions for a 5-7 nm node, not fitted to anything; every result built
//! from them carries `EnergySource::Uncalibrated`. Access COUNTS never depend on this table.

use kiln_ir::hw::compute::MemKind;
use kiln_ir::precision::Precision;

pub const PJ: f64 = 1e-12;

/// Energy per MAC (J) by input precision (assumed).
pub fn e_mac(p: Precision) -> f64 {
    use Precision::*;
    PJ * match p.compute() {
        Fp64 => 3.0,
        Fp32 | Int32 => 0.9,
        Tf32 => 0.4,
        Bf16 | Fp16 | Int16 => 0.25,
        Fp8E4m3 | Fp8E5m2 | Mxfp8E4m3 | Mxfp8E5m2 | Mxint8 => 0.12,
        Int8 | Uint8 => 0.06,
        Fp6E3m2 | Fp6E2m3 | Mxfp6E3m2 | Mxfp6E2m3 => 0.09,
        Fp4E2m1 | Mxfp4 | Nvfp4 => 0.05,
        Int4 | Uint4 => 0.03,
        _ => 0.25,
    }
}

/// Energy per vector-lane op (conversion, MX scale application), J (assumed).
pub const E_VECTOR_OP: f64 = 0.5 * PJ;

/// Read energy per byte (J) of an on-chip memory by kind and per-instance capacity (assumed).
pub fn e_read_onchip(kind: Option<MemKind>, capacity: u64) -> f64 {
    let pj = match kind {
        None | Some(MemKind::RegisterFile) | Some(MemKind::Fifo) => 0.1,
        Some(MemKind::Scratchpad) if capacity <= 256 << 10 => 0.5,
        Some(MemKind::Scratchpad) => 1.5,
        Some(MemKind::Cache) => 2.0,
    };
    pj * PJ
}

/// Off-chip DRAM energy per byte (J): ~4 pJ/bit HBM-class (assumed).
pub const E_DRAM_PER_BYTE: f64 = 32.0 * PJ;

/// Writes cost this much more than reads (assumed).
pub const WRITE_RATIO: f64 = 1.1;

/// Clock-gated padding MAC energy as a fraction of `e_mac` (03 §2.6 default, assumed).
pub const IDLE_MAC_RATIO: f64 = 0.1;
