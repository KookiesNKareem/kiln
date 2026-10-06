//! The canonical bench descriptor of 06 §4.2, shared by bench export (`kiln-wl`) and measurement import
//! (`kiln-trace`) so measurements and predictions join on identical keys.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::common::content_hash;
use crate::precision::PrecisionSpec;
use crate::wl::{CacheState, ElemType, EvalMode};

pub const KEY_PREFIX: &str = "bop1-";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchKind {
    /// `out[m,n] = a[m,k] @ b[k,n]`.
    Gemm,
    /// `nn.Linear`: weight stored `[n,k]`.
    Linear,
    /// Batched contraction, no shared operand.
    Bmm,
    Sdpa,
    Elementwise,
    Softmax,
    Rmsnorm,
    Collective,
    Copy,
    ReadReduce,
    Scale,
    Other,
}

impl BenchKind {
    pub const fn is_contraction(self) -> bool {
        matches!(self, Self::Gemm | Self::Linear | Self::Bmm)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BenchOperand {
    /// Registry name (`bf16`, `fp32`, `uint8`, `mxfp4`), else the canonical JSON of the element type.
    pub dtype: String,
    /// Index-letter layout (`mk`, `nk`, `bmn`) for contractions, `row_major` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    /// Concrete shape when `dims` does not determine it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<u64>,
}

impl BenchOperand {
    pub fn new(dtype: impl Into<String>, layout: Option<&str>) -> Self {
        Self {
            dtype: dtype.into(),
            layout: layout.map(Into::into),
            shape: vec![],
        }
    }
}

pub fn dtype_name(t: &ElemType) -> String {
    t.shorthand().map_or_else(
        || crate::common::canonical_json(&serde_json::to_value(t).expect("ElemType serializes")),
        |s| s.to_string(),
    )
}

pub fn elem_type(dtype: &str) -> Option<ElemType> {
    dtype
        .parse::<PrecisionSpec>()
        .map(ElemType::from_spec)
        .ok()
        .or_else(|| serde_json::from_str(dtype).ok())
}

/// Canonical descriptor of one measured or predicted op. Residency and occurrence data are not part of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchOp {
    pub kind: BenchKind,
    /// IR op name for kinds that do not determine the computation (`elementwise`: `act`, `rope`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    /// Contractions: `m`, `n`, `k` (+ `batch` for `bmm`); streaming ops: `bytes`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dims: BTreeMap<String, u64>,
    /// Contractions: `a`, `b`, `out`; other ops: `in0..`, `out0..` by position.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub operands: BTreeMap<String, BenchOperand>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    /// Fused epilogue ops; order-insensitive (sorted and deduplicated in the key).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub epilogue: Vec<String>,
}

impl BenchOp {
    pub fn new(kind: BenchKind) -> Self {
        Self {
            kind,
            op: None,
            dims: BTreeMap::new(),
            operands: BTreeMap::new(),
            mask: None,
            epilogue: vec![],
        }
    }

    /// `out = a @ b` with operand dtypes `[a, b, out]`. `bmm` when `batch` is `Some`, else `gemm` (`b` in
    /// `[k,n]`) or `linear` (`b` in `[n,k]`, `transposed_weight`).
    pub fn contraction(
        batch: Option<u64>,
        m: u64,
        n: u64,
        k: u64,
        transposed_weight: bool,
        dtypes: [&str; 3],
    ) -> Self {
        let (kind, layouts) = match (batch, transposed_weight) {
            (Some(_), _) => (BenchKind::Bmm, ["bmk", "bkn", "bmn"]),
            (None, true) => (BenchKind::Linear, ["mk", "nk", "mn"]),
            (None, false) => (BenchKind::Gemm, ["mk", "kn", "mn"]),
        };
        let mut op = Self::new(kind);
        op.dims = [("m", m), ("n", n), ("k", k)]
            .into_iter()
            .chain(batch.map(|b| ("batch", b)))
            .map(|(d, v)| (d.into(), v))
            .collect();
        op.operands = ["a", "b", "out"]
            .into_iter()
            .zip(layouts.into_iter().zip(dtypes))
            .map(|(name, (l, t))| (name.into(), BenchOperand::new(t, Some(l))))
            .collect();
        op
    }

    /// bf16 `gemm`/`linear`, the harness's weight GEMM.
    pub fn gemm(m: u64, n: u64, k: u64, transposed_weight: bool) -> Self {
        Self::contraction(None, m, n, k, transposed_weight, ["bf16"; 3])
    }

    /// bf16 `bmm`.
    pub fn bmm(batch: u64, m: u64, n: u64, k: u64) -> Self {
        Self::contraction(Some(batch), m, n, k, false, ["bf16"; 3])
    }

    /// Streaming memory op over `bytes` of input; operands carry dtypes only.
    pub fn stream(kind: BenchKind, bytes: u64, dtypes: &[(&str, &str)]) -> Self {
        let mut op = Self::new(kind);
        op.dims.insert("bytes".into(), bytes);
        op.operands = dtypes
            .iter()
            .map(|&(k, t)| (k.into(), BenchOperand::new(t, None)))
            .collect();
        op
    }

    /// The same contraction with the weight operand in the other layout (`gemm` <-> `linear`).
    pub fn transposed(&self) -> Option<Self> {
        let (kind, layout) = match self.kind {
            BenchKind::Gemm => (BenchKind::Linear, "nk"),
            BenchKind::Linear => (BenchKind::Gemm, "kn"),
            _ => return None,
        };
        let mut op = self.clone();
        op.kind = kind;
        op.operands.get_mut("b")?.layout = Some(layout.into());
        Some(op)
    }

    pub fn dim(&self, d: &str) -> Option<u64> {
        self.dims.get(d).copied()
    }

    pub fn to_canonical_value(&self) -> Value {
        let mut op = self.clone();
        op.epilogue.sort();
        op.epilogue.dedup();
        serde_json::to_value(op).expect("BenchOp serializes")
    }

    /// `bop1-` + sha256 over the canonical descriptor (06 §4.2), so joins never rely on names.
    pub fn key(&self) -> String {
        content_hash(KEY_PREFIX, &self.to_canonical_value())
    }

    /// Harness op-list key (02 §13.2 `key_for`): `gemm_M_N_K` for either weight layout, `bmm_B_M_N_K`.
    pub fn legacy_key(&self) -> Option<String> {
        let d = |k| self.dim(k);
        match self.kind {
            BenchKind::Gemm | BenchKind::Linear => {
                Some(format!("gemm_{}_{}_{}", d("m")?, d("n")?, d("k")?))
            }
            BenchKind::Bmm => Some(format!(
                "bmm_{}_{}_{}_{}",
                d("batch")?,
                d("m")?,
                d("n")?,
                d("k")?
            )),
            _ => None,
        }
    }

    /// Name of the harness record that measures exactly this descriptor: the key, `_linear` for `[n,k]` weights.
    pub fn legacy_name(&self) -> Option<String> {
        let key = self.legacy_key()?;
        Some(match self.kind {
            BenchKind::Linear => format!("{key}_linear"),
            _ => key,
        })
    }

    /// An operand's explicit shape, or a contraction operand's shape from its index-letter layout.
    pub fn operand_shape(&self, name: &str) -> Option<Vec<u64>> {
        let o = self.operands.get(name)?;
        if !o.shape.is_empty() {
            return Some(o.shape.clone());
        }
        o.layout
            .as_deref()?
            .chars()
            .map(|c| match c {
                'b' => self.dim("batch"),
                'm' | 'n' | 'k' => self.dim(&c.to_string()),
                _ => None,
            })
            .collect()
    }

    /// Compulsory bytes: every operand moved once, at its element type's storage size.
    pub fn min_bytes(&self) -> Option<u128> {
        self.operands
            .iter()
            .map(|(name, o)| {
                let shape = self.operand_shape(name)?;
                let numel = shape.iter().map(|&d| u128::from(d)).product();
                Some(elem_type(&o.dtype)?.bytes(numel, &shape))
            })
            .sum()
    }

    /// Dense contraction FLOPs.
    pub fn flops(&self) -> Option<u128> {
        if !self.kind.is_contraction() {
            return None;
        }
        let d = |k| self.dim(k).map(u128::from);
        Some(2 * d("batch").unwrap_or(1) * d("m")? * d("n")? * d("k")?)
    }
}

/// Operand residency at op start (06 §2.5, 02 §12.6 eval modes). A property of a timing mode, never of the
/// descriptor: one measurement record holds several modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    ColdDram,
    L2Warm,
    Resident,
}

impl Residency {
    pub fn from_eval(e: EvalMode) -> Self {
        match e {
            EvalMode::Isolated {
                cache: CacheState::Cold,
                ..
            } => Self::ColdDram,
            EvalMode::Isolated {
                cache: CacheState::Warm,
                ..
            } => Self::L2Warm,
            EvalMode::WholeGraph => Self::Resident,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_distinguish_layout_and_kind() {
        let g = BenchOp::gemm(1, 4096, 4096, false);
        let l = BenchOp::gemm(1, 4096, 4096, true);
        assert_ne!(g.key(), l.key());
        assert_eq!(g.transposed().unwrap(), l);
        assert_ne!(g.key(), BenchOp::bmm(1, 1, 4096, 4096).key());
        assert!(g.key().starts_with(KEY_PREFIX) && g.key().len() == 37);
        assert_eq!(g.legacy_name().unwrap(), "gemm_1_4096_4096");
        assert_eq!(l.legacy_name().unwrap(), "gemm_1_4096_4096_linear");
        assert_eq!(l.legacy_key().unwrap(), "gemm_1_4096_4096");
        assert_eq!(
            BenchOp::bmm(256, 4, 2048, 128).legacy_name().unwrap(),
            "bmm_256_4_2048_128"
        );
        assert_eq!(BenchOp::bmm(2, 3, 4, 5).flops(), Some(240));
        assert_eq!(
            BenchOp::bmm(2, 3, 4, 5).min_bytes(),
            Some(2 * 2 * (15 + 20 + 12))
        );
        assert_eq!(
            BenchOp::gemm(3, 4, 5, true).operand_shape("b"),
            Some(vec![4, 5])
        );
    }

    #[test]
    fn epilogue_order_is_not_in_the_key() {
        let mut a = BenchOp::gemm(8, 8, 8, true);
        let mut b = a.clone();
        a.epilogue = vec!["bias".into(), "silu".into()];
        b.epilogue = vec!["silu".into(), "bias".into(), "bias".into()];
        assert_eq!(a.key(), b.key());
    }

    #[test]
    fn key_survives_round_trip() {
        for op in [
            BenchOp::bmm(3, 5, 7, 11),
            BenchOp::gemm(1, 2, 3, true),
            BenchOp::stream(BenchKind::Copy, 1 << 30, &[("x", "uint8")]),
        ] {
            let back: BenchOp = serde_json::from_str(&serde_json::to_string(&op).unwrap()).unwrap();
            assert_eq!(back.key(), op.key());
        }
    }

    #[test]
    fn dtype_names_use_the_registry() {
        assert_eq!(dtype_name(&ElemType::BF16), "bf16");
        assert_eq!(dtype_name(&ElemType::FP32), "fp32");
    }
}
