//! Tensor declarations (02 §3).

use serde::{Deserialize, Serialize};

use super::Meta;
use super::dim::DimExpr;
use super::dtype::ElemType;
use crate::common::Id;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorClass {
    Weight,
    KvCache,
    Activation,
    Input,
    Output,
    Constant,
}

impl TensorClass {
    pub const fn is_model_level(self) -> bool {
        matches!(self, Self::Weight | Self::KvCache | Self::Constant)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Layout {
    RowMajor,
    Permuted {
        order: Vec<u8>,
    },
    Tiled {
        order: Vec<u8>,
        tiles: Vec<(u8, DimExpr)>,
    },
    Paged {
        token_axis: u8,
        page_tokens: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutSpec {
    pub layout: Layout,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Sparsity {
    Structured { n: u32, m: u32, axis: i32 },
    Block { block: Vec<u32>, density: f64 },
    Unstructured { density: f64 },
    Dynamic { density: DimExpr, source: Id },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum InitHint {
    Zeros,
    Random,
    File(String),
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDecl {
    pub shape: Vec<DimExpr>,
    pub dtype: ElemType,
    pub class: TensorClass,
    #[serde(default)]
    pub stack: Option<DimExpr>,
    #[serde(default)]
    pub layout: Option<LayoutSpec>,
    #[serde(default)]
    pub sparsity: Option<Sparsity>,
    #[serde(default)]
    pub alias_of: Option<Id>,
    #[serde(default)]
    pub init: Option<InitHint>,
    #[serde(default = "yes")]
    pub upcast_ok: bool,
    #[serde(default, skip_serializing_if = "Meta::is_empty")]
    pub meta: Meta,
}

impl TensorDecl {
    pub fn new(
        shape: impl IntoIterator<Item = DimExpr>,
        dtype: ElemType,
        class: TensorClass,
    ) -> Self {
        Self {
            shape: shape.into_iter().collect(),
            dtype,
            class,
            stack: None,
            layout: None,
            sparsity: None,
            alias_of: None,
            init: None,
            upcast_ok: true,
            meta: Meta::default(),
        }
    }

    pub fn stacked(mut self, count: DimExpr) -> Self {
        self.stack = Some(count);
        self
    }

    pub fn alias(mut self, of: Id) -> Self {
        self.alias_of = Some(of);
        self
    }
}

/// Bound (all-integer) tensor type: what op inference and lowering see.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeInfo {
    pub shape: Vec<u64>,
    pub dtype: ElemType,
    pub class: TensorClass,
}

impl TypeInfo {
    pub fn new(shape: Vec<u64>, dtype: ElemType, class: TensorClass) -> Self {
        Self {
            shape,
            dtype,
            class,
        }
    }

    pub fn checked_numel(&self) -> Option<u128> {
        self.shape.iter().try_fold(1u128, |n, &d| n.checked_mul(u128::from(d)))
    }

    /// Saturates at `u128::MAX`; bound types are rejected before lowering when `checked_footprint` overflows.
    pub fn numel(&self) -> u128 {
        self.checked_numel().unwrap_or(u128::MAX)
    }

    /// Storage bytes of `elems` elements of this tensor.
    pub fn bytes(&self, elems: u128) -> u128 {
        self.dtype.bytes(elems, &self.shape)
    }

    pub fn checked_footprint(&self) -> Option<u128> {
        self.dtype.checked_bytes(self.checked_numel()?, &self.shape)
    }

    pub fn footprint(&self) -> u128 {
        self.checked_footprint().unwrap_or(u128::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_count_overflow_is_detected_not_wrapped() {
        let t = TypeInfo::new(vec![1 << 43, 1 << 43, 1 << 42], ElemType::BF16, TensorClass::Weight);
        assert_eq!(t.checked_numel(), None);
        assert_eq!(t.checked_footprint(), None);
        assert_eq!(t.numel(), u128::MAX);
        assert_eq!(t.footprint(), u128::MAX);
        let big = TypeInfo::new(vec![1 << 63, 1 << 63], ElemType::BF16, TensorClass::Weight);
        assert_eq!(big.checked_numel(), Some(1 << 126));
        assert_eq!(big.checked_footprint(), None);
        let ok = TypeInfo::new(vec![4, 8], ElemType::BF16, TensorClass::Weight);
        assert_eq!((ok.checked_numel(), ok.checked_footprint()), (Some(32), Some(64)));
    }
}
