//! Op classes and special functions (01 §6.3, §5.4), shared with the workload IR.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpClass {
    Matmul,
    Conv,
    Elementwise,
    Transcendental,
    Reduction,
    Convert,
    Permute,
    GatherScatter,
    Scan,
    SortTopk,
    Control,
    CollectiveReduce,
}

impl OpClass {
    pub const ALL: [OpClass; 12] = [
        Self::Matmul,
        Self::Conv,
        Self::Elementwise,
        Self::Transcendental,
        Self::Reduction,
        Self::Convert,
        Self::Permute,
        Self::GatherScatter,
        Self::Scan,
        Self::SortTopk,
        Self::Control,
        Self::CollectiveReduce,
    ];

    /// Default `class_rate` on a vector unit (table 6.3).
    pub const fn default_vector_rate(self) -> f64 {
        match self {
            Self::Transcendental | Self::SortTopk => 0.125,
            Self::Permute | Self::Scan => 0.5,
            Self::GatherScatter => 0.25,
            _ => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialFn {
    Exp,
    Exp2,
    Log,
    Log2,
    Rsqrt,
    Sqrt,
    Recip,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Gelu,
    Silu,
    Erf,
}
