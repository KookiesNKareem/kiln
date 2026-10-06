//! Element types (02 §3.2): a scalar precision from the shared registry (01 §6.1) plus scaling.

use serde::{Deserialize, Deserializer, Serialize};

use crate::common::Diagnostic;
use crate::precision::{Precision, PrecisionKind, PrecisionSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Scaling {
    None,
    PerTensor {
        scale: Precision,
    },
    PerAxis {
        axis: i32,
        scale: Precision,
    },
    Block {
        axis: i32,
        block: u32,
        scale: Precision,
        #[serde(default)]
        zero_point: Option<Precision>,
        #[serde(default)]
        tensor_scale: Option<Precision>,
    },
}

/// Canonical form is always the expanded struct; deserialization also accepts a registry shorthand string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElemType {
    pub scalar: Precision,
    pub scaling: Scaling,
}

impl ElemType {
    pub const fn plain(scalar: Precision) -> Self {
        Self {
            scalar,
            scaling: Scaling::None,
        }
    }

    pub const BF16: Self = Self::plain(Precision::Bf16);
    pub const FP32: Self = Self::plain(Precision::Fp32);
    pub const INT32: Self = Self::plain(Precision::Int32);

    /// Expansion of a registry shorthand (02 §3.2 table).
    pub fn from_spec(spec: PrecisionSpec) -> Self {
        use Precision::*;
        let p = spec.precision;
        let block = |scale, zero_point, tensor_scale| Scaling::Block {
            axis: -1,
            block: spec.block_size().expect("block formats have a block size"),
            scale,
            zero_point,
            tensor_scale,
        };
        let (scalar, scaling) = match p {
            Mxfp8E4m3 => (Fp8E4m3, block(E8m0, None, None)),
            Mxfp8E5m2 => (Fp8E5m2, block(E8m0, None, None)),
            Mxfp6E3m2 => (Fp6E3m2, block(E8m0, None, None)),
            Mxfp6E2m3 => (Fp6E2m3, block(E8m0, None, None)),
            Mxfp4 => (Fp4E2m1, block(E8m0, None, None)),
            Mxint8 => (Int8, block(E8m0, None, None)),
            Nvfp4 => (Fp4E2m1, block(Fp8E4m3, None, Some(Fp32))),
            Int4G128 => (Int4, block(Bf16, Some(Int4), None)),
            Fp8E4m3Pt => (Fp8E4m3, Scaling::PerTensor { scale: Fp32 }),
            Int8Pc => (
                Int8,
                Scaling::PerAxis {
                    axis: 0,
                    scale: Fp32,
                },
            ),
            p => (p, Scaling::None),
        };
        Self { scalar, scaling }
    }

    /// The registry name this type is the expansion of, if any.
    pub fn shorthand(&self) -> Option<PrecisionSpec> {
        let candidates = Precision::ALL.into_iter().map(PrecisionSpec::new);
        let blocks = [8, 16, 32, 64, 128, 256].into_iter().flat_map(|b| {
            Precision::ALL
                .into_iter()
                .filter(move |p| p.default_block().is_some_and(|d| d != b))
                .map(move |p| PrecisionSpec {
                    precision: p,
                    block: Some(b),
                })
        });
        candidates
            .chain(blocks)
            .find(|s| Self::from_spec(*s) == *self)
    }

    pub fn elem_bits(&self) -> u32 {
        self.scalar.element_bits()
    }

    pub fn is_float(&self) -> bool {
        self.scalar.is_float()
    }

    /// Structural check (E-WL-DT-002): scalar and scale types must be plain registry entries.
    pub fn check(&self) -> Result<(), Diagnostic> {
        let plain = |p: Precision| {
            !matches!(
                p.kind(),
                PrecisionKind::Mx | PrecisionKind::StorageVariant | PrecisionKind::BlockFloat
            )
        };
        let bad = |what: &str, p: Precision| {
            Diagnostic::error("E-WL-DT-002", format!("{what} {p} is not a plain element type"))
                .hint("use an element precision (e.g. fp4_e2m1) with a Scaling, or a shorthand string such as \"mxfp4\"")
        };
        if !plain(self.scalar) || self.scalar == Precision::E8m0 {
            return Err(bad("scalar", self.scalar));
        }
        let scales: Vec<Precision> = match self.scaling {
            Scaling::None => vec![],
            Scaling::PerTensor { scale } | Scaling::PerAxis { scale, .. } => vec![scale],
            Scaling::Block {
                block,
                scale,
                zero_point,
                tensor_scale,
                ..
            } => {
                if block == 0 {
                    return Err(Diagnostic::error(
                        "E-WL-DT-002",
                        "block size must be positive",
                    ));
                }
                [Some(scale), zero_point, tensor_scale]
                    .into_iter()
                    .flatten()
                    .collect()
            }
        };
        match scales.into_iter().find(|p| !plain(*p)) {
            Some(p) => Err(bad("scale", p)),
            None => Ok(()),
        }
    }

    /// Exact storage bytes for `elems` elements of a tensor with `numel` elements and the given shape
    /// (02 §3.1/§3.2). Partial reads touch `ceil(elems / covered)` scales, where `covered` is the number of
    /// elements one scale serves; per-tensor scales are counted once when any element is touched.
    pub fn bytes(&self, elems: u128, shape: &[u64]) -> u128 {
        if elems == 0 {
            return 0;
        }
        let numel: u128 = shape
            .iter()
            .map(|&d| u128::from(d))
            .product::<u128>()
            .max(1);
        let data = (elems * u128::from(self.elem_bits())).div_ceil(8);
        let axis_len = |axis: i32| {
            let r = shape.len() as i32;
            let a = if axis < 0 { axis + r } else { axis };
            usize::try_from(a)
                .ok()
                .and_then(|a| shape.get(a))
                .map_or(1, |&d| u128::from(d.max(1)))
        };
        let scales = match self.scaling {
            Scaling::None => 0,
            Scaling::PerTensor { scale } => u128::from(scale.element_bits()).div_ceil(8),
            Scaling::PerAxis { axis, scale } => {
                let covered = (numel / axis_len(axis)).max(1);
                (elems.div_ceil(covered) * u128::from(scale.element_bits())).div_ceil(8)
            }
            Scaling::Block {
                block,
                scale,
                zero_point,
                tensor_scale,
                ..
            } => {
                let per =
                    u128::from(scale.element_bits() + zero_point.map_or(0, |z| z.element_bits()));
                (elems.div_ceil(u128::from(block)) * per).div_ceil(8)
                    + tensor_scale.map_or(0, |t| u128::from(t.element_bits()).div_ceil(8))
            }
        };
        data + scales
    }
}

impl From<Precision> for ElemType {
    fn from(p: Precision) -> Self {
        Self::from_spec(PrecisionSpec::new(p))
    }
}

impl<'de> Deserialize<'de> for ElemType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Full {
            scalar: Precision,
            #[serde(default = "none")]
            scaling: Scaling,
        }
        fn none() -> Scaling {
            Scaling::None
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Src {
            Short(PrecisionSpec),
            Full(Full),
        }
        Ok(match Src::deserialize(d)? {
            Src::Short(s) => Self::from_spec(s),
            Src::Full(f) => Self {
                scalar: f.scalar,
                scaling: f.scaling,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shorthand_expansion_matches_registry_storage_bits() {
        for p in Precision::ALL {
            let et = ElemType::from(p);
            if p == Precision::E8m0 {
                continue;
            }
            et.check().unwrap_or_else(|e| panic!("{p}: {e}"));
            let n: u128 = 1 << 20;
            let bits = et.bytes(n, &[1024, 1024]) as f64 * 8.0 / n as f64;
            let expect = p.storage_bits();
            let per_tensor_overhead = 8.0 * 8.0 / n as f64;
            assert!(
                (bits - expect).abs() <= per_tensor_overhead + 1024.0 * 32.0 / n as f64,
                "{p}: {bits} vs {expect}"
            );
            assert_eq!(et.shorthand().map(|s| s.precision), Some(p), "{p}");
        }
    }

    #[test]
    fn serde_accepts_shorthand_and_emits_struct() {
        let e: ElemType = serde_json::from_str("\"int4_g128\"").unwrap();
        assert_eq!(e.scalar, Precision::Int4);
        let s = serde_json::to_value(e).unwrap();
        assert_eq!(s["scaling"]["block"], 128);
        let back: ElemType = serde_json::from_value(s).unwrap();
        assert_eq!(back, e);
        let mx16: ElemType = serde_json::from_str("\"mxfp4/16\"").unwrap();
        assert_eq!(mx16.shorthand().unwrap().to_string(), "mxfp4/16");
    }

    #[test]
    fn exact_bytes() {
        assert_eq!(ElemType::BF16.bytes(10, &[10]), 20);
        assert_eq!(ElemType::from(Precision::Mxfp4).bytes(64, &[64]), 32 + 2);
        assert_eq!(
            ElemType::from(Precision::Nvfp4).bytes(32, &[32]),
            16 + 2 + 4
        );
        assert_eq!(ElemType::from(Precision::Int8Pc).bytes(8, &[4, 8]), 8 + 4);
        assert!(ElemType::plain(Precision::Mxfp4).check().is_err());
    }
}
