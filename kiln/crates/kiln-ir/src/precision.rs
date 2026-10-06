//! The single precision registry (01 §6.1, 00 decision 4), shared by the hardware and workload IRs.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::common::Diagnostic;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    Fp64,
    Fp32,
    Tf32,
    Bf16,
    Fp16,
    Fp8E4m3,
    Fp8E5m2,
    Fp6E3m2,
    Fp6E2m3,
    Fp4E2m1,
    Mxfp8E4m3,
    Mxfp8E5m2,
    Mxfp6E3m2,
    Mxfp6E2m3,
    Mxfp4,
    Mxint8,
    Nvfp4,
    Int32,
    Int16,
    Int8,
    Int4,
    Uint8,
    Uint4,
    Int64,
    Bool,
    E8m0,
    Fp8E4m3Pt,
    Int8Pc,
    Int4G128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionKind {
    Float,
    Mx,
    BlockFloat,
    Int,
    /// Index/mask tensors only (`int64`, `bool`).
    IndexOrMask,
    ScaleOnly,
    StorageVariant,
}

/// How scale data is carried alongside elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    None,
    PerTensor { scale: Precision },
    PerAxis { scale: Precision },
    Block { block: u32, scale_bits: u32, zero_point_bits: u32 },
}

impl Precision {
    pub const ALL: [Precision; 29] = [
        Self::Fp64,
        Self::Fp32,
        Self::Tf32,
        Self::Bf16,
        Self::Fp16,
        Self::Fp8E4m3,
        Self::Fp8E5m2,
        Self::Fp6E3m2,
        Self::Fp6E2m3,
        Self::Fp4E2m1,
        Self::Mxfp8E4m3,
        Self::Mxfp8E5m2,
        Self::Mxfp6E3m2,
        Self::Mxfp6E2m3,
        Self::Mxfp4,
        Self::Mxint8,
        Self::Nvfp4,
        Self::Int32,
        Self::Int16,
        Self::Int8,
        Self::Int4,
        Self::Uint8,
        Self::Uint4,
        Self::Int64,
        Self::Bool,
        Self::E8m0,
        Self::Fp8E4m3Pt,
        Self::Int8Pc,
        Self::Int4G128,
    ];

    pub const fn name(self) -> &'static str {
        use Precision::*;
        match self {
            Fp64 => "fp64",
            Fp32 => "fp32",
            Tf32 => "tf32",
            Bf16 => "bf16",
            Fp16 => "fp16",
            Fp8E4m3 => "fp8_e4m3",
            Fp8E5m2 => "fp8_e5m2",
            Fp6E3m2 => "fp6_e3m2",
            Fp6E2m3 => "fp6_e2m3",
            Fp4E2m1 => "fp4_e2m1",
            Mxfp8E4m3 => "mxfp8_e4m3",
            Mxfp8E5m2 => "mxfp8_e5m2",
            Mxfp6E3m2 => "mxfp6_e3m2",
            Mxfp6E2m3 => "mxfp6_e2m3",
            Mxfp4 => "mxfp4",
            Mxint8 => "mxint8",
            Nvfp4 => "nvfp4",
            Int32 => "int32",
            Int16 => "int16",
            Int8 => "int8",
            Int4 => "int4",
            Uint8 => "uint8",
            Uint4 => "uint4",
            Int64 => "int64",
            Bool => "bool",
            E8m0 => "e8m0",
            Fp8E4m3Pt => "fp8_e4m3_pt",
            Int8Pc => "int8_pc",
            Int4G128 => "int4_g128",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    pub const fn kind(self) -> PrecisionKind {
        use Precision::*;
        match self {
            Fp64 | Fp32 | Tf32 | Bf16 | Fp16 | Fp8E4m3 | Fp8E5m2 | Fp6E3m2 | Fp6E2m3 | Fp4E2m1 => PrecisionKind::Float,
            Mxfp8E4m3 | Mxfp8E5m2 | Mxfp6E3m2 | Mxfp6E2m3 | Mxfp4 | Mxint8 => PrecisionKind::Mx,
            Nvfp4 => PrecisionKind::BlockFloat,
            Int32 | Int16 | Int8 | Int4 | Uint8 | Uint4 => PrecisionKind::Int,
            Int64 | Bool => PrecisionKind::IndexOrMask,
            E8m0 => PrecisionKind::ScaleOnly,
            Fp8E4m3Pt | Int8Pc | Int4G128 => PrecisionKind::StorageVariant,
        }
    }

    /// Bits of one element, excluding any scale overhead.
    pub const fn element_bits(self) -> u32 {
        use Precision::*;
        match self {
            Fp64 | Int64 => 64,
            Fp32 | Tf32 | Int32 => 32,
            Bf16 | Fp16 | Int16 => 16,
            Fp8E4m3 | Fp8E5m2 | Mxfp8E4m3 | Mxfp8E5m2 | Mxint8 | Int8 | Uint8 | Bool | E8m0 | Fp8E4m3Pt | Int8Pc => 8,
            Fp6E3m2 | Fp6E2m3 | Mxfp6E3m2 | Mxfp6E2m3 => 6,
            Fp4E2m1 | Mxfp4 | Nvfp4 | Int4 | Uint4 | Int4G128 => 4,
        }
    }

    /// The compute precision a storage variant runs as (identity for every other name).
    pub const fn compute(self) -> Precision {
        match self {
            Self::Fp8E4m3Pt => Self::Fp8E4m3,
            Self::Int8Pc => Self::Int8,
            Self::Int4G128 => Self::Int4,
            p => p,
        }
    }

    pub const fn default_block(self) -> Option<u32> {
        match self.kind() {
            PrecisionKind::Mx => Some(32),
            PrecisionKind::BlockFloat => Some(16),
            _ if matches!(self, Self::Int4G128) => Some(128),
            _ => None,
        }
    }

    pub const fn scale_layout(self, block: u32) -> ScaleLayout {
        match self {
            Self::Nvfp4 => ScaleLayout::Block { block, scale_bits: 8, zero_point_bits: 0 },
            Self::Int4G128 => ScaleLayout::Block { block, scale_bits: 16, zero_point_bits: 4 },
            Self::Fp8E4m3Pt => ScaleLayout::PerTensor { scale: Self::Fp32 },
            Self::Int8Pc => ScaleLayout::PerAxis { scale: Self::Fp32 },
            p if matches!(p.kind(), PrecisionKind::Mx) => ScaleLayout::Block { block, scale_bits: 8, zero_point_bits: 0 },
            _ => ScaleLayout::None,
        }
    }

    /// Storage bits per element including amortized block scales (01 §6.1); per-tensor and per-row scales are
    /// excluded because they do not scale with element count.
    pub fn storage_bits(self) -> f64 {
        PrecisionSpec::new(self).storage_bits()
    }

    pub const fn is_float(self) -> bool {
        matches!(self.compute().kind(), PrecisionKind::Float | PrecisionKind::BlockFloat)
            || matches!(self, Self::Mxfp8E4m3 | Self::Mxfp8E5m2 | Self::Mxfp6E3m2 | Self::Mxfp6E2m3 | Self::Mxfp4)
    }

    pub const fn is_integer(self) -> bool {
        matches!(self.compute(), Self::Int32 | Self::Int16 | Self::Int8 | Self::Int4 | Self::Uint8 | Self::Uint4)
            || matches!(self, Self::Mxint8 | Self::Int64)
    }

    /// Exponent bits for float formats (accumulator range check, E-IR-0303); MX elements count their element
    /// exponent since the shared scale is applied outside the datapath.
    pub const fn exponent_bits(self) -> Option<u32> {
        use Precision::*;
        match self.compute() {
            Fp64 => Some(11),
            Fp32 | Tf32 | Bf16 => Some(8),
            Fp16 | Fp8E5m2 | Mxfp8E5m2 => Some(5),
            Fp8E4m3 | Mxfp8E4m3 => Some(4),
            Fp6E3m2 | Mxfp6E3m2 => Some(3),
            Fp6E2m3 | Mxfp6E2m3 | Fp4E2m1 | Mxfp4 | Nvfp4 => Some(2),
            _ => None,
        }
    }

    /// True if this name may appear as an operand of a hardware `PrecisionMode` (compute names only).
    pub const fn is_compute_name(self) -> bool {
        !matches!(self.kind(), PrecisionKind::StorageVariant | PrecisionKind::ScaleOnly)
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Precision {
    type Err = Diagnostic;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| unknown_precision(s))
    }
}

fn unknown_precision(s: &str) -> Diagnostic {
    let names: Vec<&str> = Precision::ALL.iter().map(|p| p.name()).collect();
    Diagnostic::error("E-IR-0302", format!("unknown precision {s:?}"))
        .hint(format!("use one of: {}; non-default MX block sizes are written \"mxfp4/16\"", names.join(", ")))
}

impl Serialize for Precision {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Precision {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(|e: Diagnostic| serde::de::Error::custom(e))
    }
}

/// A registry name plus an optional non-default block size (`"mxfp4/16"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecisionSpec {
    pub precision: Precision,
    pub block: Option<u32>,
}

impl PrecisionSpec {
    pub const fn new(precision: Precision) -> Self {
        Self { precision, block: None }
    }

    pub fn block_size(self) -> Option<u32> {
        self.block.or(self.precision.default_block())
    }

    pub fn storage_bits(self) -> f64 {
        let elem = f64::from(self.precision.element_bits());
        match self.precision.scale_layout(self.block_size().unwrap_or(1)) {
            ScaleLayout::Block { block, scale_bits, zero_point_bits } => {
                elem + f64::from(scale_bits + zero_point_bits) / f64::from(block)
            }
            _ => elem,
        }
    }
}

impl From<Precision> for PrecisionSpec {
    fn from(p: Precision) -> Self {
        Self::new(p)
    }
}

impl fmt::Display for PrecisionSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.block {
            Some(b) => write!(f, "{}/{b}", self.precision),
            None => write!(f, "{}", self.precision),
        }
    }
}

impl FromStr for PrecisionSpec {
    type Err = Diagnostic;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((name, block)) = s.split_once('/') else {
            return Ok(Self::new(s.parse()?));
        };
        let precision: Precision = name.parse()?;
        let block: u32 = block.parse().ok().filter(|b| *b > 0).ok_or_else(|| {
            Diagnostic::error("E-IR-0302", format!("invalid block size in precision {s:?}"))
                .hint("write a positive integer block size, e.g. \"mxfp4/16\"")
        })?;
        if precision.default_block().is_none() {
            return Err(Diagnostic::error("E-IR-0302", format!("precision {name:?} has no block size"))
                .hint("only block-scaled formats (mx*, nvfp4, int4_g128) accept \"/<block>\""));
        }
        let block = (Some(block) != precision.default_block()).then_some(block);
        Ok(Self { precision, block })
    }
}

impl Serialize for PrecisionSpec {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for PrecisionSpec {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(|e: Diagnostic| serde::de::Error::custom(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for p in Precision::ALL {
            assert_eq!(p.name().parse::<Precision>().unwrap(), p);
            assert_eq!(serde_json::to_string(&p).unwrap(), format!("\"{}\"", p.name()));
        }
        assert_eq!("fp7".parse::<Precision>().unwrap_err().code, "E-IR-0302");
    }

    #[test]
    fn storage_bits_match_registry_table() {
        let cases = [
            ("fp32", 32.0),
            ("tf32", 32.0),
            ("bf16", 16.0),
            ("fp8_e4m3", 8.0),
            ("mxfp8_e4m3", 8.25),
            ("mxfp6_e2m3", 6.25),
            ("mxfp4", 4.25),
            ("mxint8", 8.25),
            ("nvfp4", 4.5),
            ("int4", 4.0),
            ("bool", 8.0),
            ("fp8_e4m3_pt", 8.0),
            ("int8_pc", 8.0),
            ("mxfp4/16", 4.5),
        ];
        for (name, bits) in cases {
            assert_eq!(name.parse::<PrecisionSpec>().unwrap().storage_bits(), bits, "{name}");
        }
        assert!((Precision::Int4G128.storage_bits() - 4.156).abs() < 1e-3);
    }

    #[test]
    fn block_sizes() {
        assert_eq!("mxfp4/32".parse::<PrecisionSpec>().unwrap(), PrecisionSpec::new(Precision::Mxfp4));
        assert_eq!("mxfp4/16".parse::<PrecisionSpec>().unwrap().to_string(), "mxfp4/16");
        assert!("bf16/16".parse::<PrecisionSpec>().is_err());
    }

    #[test]
    fn storage_variants_compute_as_base() {
        assert_eq!(Precision::Int4G128.compute(), Precision::Int4);
        assert!(!Precision::Int8Pc.is_compute_name());
        assert!(Precision::Int8Pc.is_integer());
        assert!(Precision::Mxfp4.is_float() && !Precision::Mxint8.is_float());
    }
}
