//! Compute units, precision modes, on-chip and off-chip memory, near-memory compute (01 §5-§9).

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::phys::{Footprint, Placement, PowerOverride, TechRef};
use super::quantity::{BitsPerSec, Bytes, BytesPerSec, Cycles, JoulesPerByte, Mm2, Seconds, Watts};
use super::types::{ClockRef, Contents, LayerRef, Ref, Replication, Selector, one, one_f};
use crate::common::Id;
use crate::op_class::{OpClass, SpecialFn};
use crate::precision::{Precision, PrecisionSpec};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputeUnit {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    #[serde(flatten)]
    pub kind: ComputeKind,
    pub precisions: Vec<PrecisionMode>,
    #[serde(default)]
    pub ops: Option<Vec<OpClass>>,
    #[serde(default)]
    pub feeds: BTreeMap<OperandRole, Feed>,
    #[serde(default)]
    pub local: Vec<LocalBuffer>,
    #[serde(default)]
    pub pipeline: Pipeline,
    #[serde(default)]
    pub near: Option<NearBinding>,
    #[serde(default)]
    pub clock: Option<ClockRef>,
    #[serde(default)]
    pub footprint: Option<Footprint>,
    #[serde(default)]
    pub placement: Placement,
    #[serde(default)]
    pub power: PowerOverride,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComputeKind {
    Matrix(MatrixSpec),
    Vector(VectorSpec),
    Scalar(ScalarSpec),
    Special(SpecialSpec),
    Cim(CimSpec),
}

/// Union of every variant's field names: a flattened struct only claims the keys it names, so listing them
/// all lets the outer struct keep `deny_unknown_fields` while the variant struct rejects foreign keys.
const KIND_FIELDS: &[&str] = &[
    "kind",
    "geometry",
    "dataflow",
    "sparsity",
    "accumulate_in",
    "operand_run",
    "lanes",
    "sublanes",
    "class_rates",
    "reduce_tree",
    "issue_width",
    "controls",
    "functions",
    "fn_rates",
    "rows",
    "cols",
    "cell_bits",
    "input_bits_per_cycle",
    "parallel_rows",
    "weight_sets",
    "weight_capacity",
    "style",
    "adc_bits",
    "weight_write",
];

impl ComputeKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Matrix(_) => "matrix",
            Self::Vector(_) => "vector",
            Self::Scalar(_) => "scalar",
            Self::Special(_) => "special",
            Self::Cim(_) => "cim",
        }
    }

    /// MACs/cycle (matrix), lane-ops/cycle (vector, scalar, special), or 1b x 1b products/cycle (cim), before
    /// precision rate. Use [`Self::ops_per_cycle`] for a mode's throughput.
    pub fn base_ops_per_cycle(&self) -> u64 {
        self.checked_base_ops_per_cycle().unwrap_or(u64::MAX)
    }

    /// [`Self::base_ops_per_cycle`], `None` when the geometry product overflows `u64`.
    pub fn checked_base_ops_per_cycle(&self) -> Option<u64> {
        match self {
            Self::Matrix(m) => m.geometry.checked_macs_per_cycle(),
            Self::Cim(c) => product([c.active_rows(), c.cols, c.cell_bits, c.input_bits_per_cycle]),
            Self::Vector(v) => product([v.lanes, v.sublanes]),
            Self::Scalar(s) => Some(u64::from(s.issue_width)),
            Self::Special(s) => Some(u64::from(s.lanes)),
        }
    }

    /// MACs/cycle (MAC kinds) or lane-ops/cycle in mode `m`, including its rate (01 §5.1). CIM throughput is
    /// derived per mode from the array (§9.3); its `@rate` only de-rates.
    pub fn ops_per_cycle(&self, m: &PrecisionMode) -> f64 {
        match (self, m) {
            (Self::Cim(c), PrecisionMode::Mac { a, b, rate, .. }) => {
                c.macs_per_cycle(a.precision.element_bits(), b.precision.element_bits()) * rate
            }
            _ => self.base_ops_per_cycle() as f64 * m.rate(),
        }
    }

    pub fn is_mac(&self) -> bool {
        matches!(self, Self::Matrix(_) | Self::Cim(_))
    }

    pub fn default_ops(&self) -> Vec<OpClass> {
        use OpClass::*;
        match self {
            Self::Matrix(_) => vec![Matmul, Conv],
            Self::Cim(_) => vec![Matmul],
            Self::Vector(_) => {
                vec![Elementwise, Transcendental, Reduction, Convert, Permute, GatherScatter, Scan, SortTopk]
            }
            Self::Scalar(_) => vec![Elementwise, GatherScatter, Control],
            Self::Special(_) => vec![Transcendental, CollectiveReduce],
        }
    }

    pub fn legal_ops(&self) -> &'static [OpClass] {
        use OpClass::*;
        match self {
            Self::Matrix(_) | Self::Cim(_) => &[Matmul, Conv, CollectiveReduce],
            Self::Vector(_) => &[
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
            ],
            Self::Scalar(_) => &[Elementwise, GatherScatter, Control, Reduction, Convert],
            Self::Special(_) => &[Transcendental, CollectiveReduce],
        }
    }

    /// Per-class rate multiplier (table 6.3, overridable per vector unit).
    pub fn class_rate(&self, class: OpClass) -> f64 {
        match self {
            Self::Vector(v) => v.class_rates.get(&class).copied().unwrap_or(class.default_vector_rate()),
            _ => 1.0,
        }
    }
}

fn product(dims: impl IntoIterator<Item = u32>) -> Option<u64> {
    dims.into_iter().try_fold(1u64, |p, d| p.checked_mul(u64::from(d)))
}

impl<'de> Deserialize<'de> for ComputeKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ComputeKind;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a compute unit with a `kind`")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ComputeKind, A::Error> {
                let mut fields = Map::new();
                while let Some(k) = map.next_key::<String>()? {
                    fields.insert(k, map.next_value::<Value>()?);
                }
                let kind = match fields.remove("kind") {
                    Some(Value::String(s)) => s,
                    Some(other) => return Err(de::Error::custom(format!("kind must be a string, got {other}"))),
                    None => return Err(de::Error::missing_field("kind")),
                };
                let v = Value::Object(fields);
                let r = match kind.as_str() {
                    "matrix" => serde_json::from_value(v).map(ComputeKind::Matrix),
                    "vector" => serde_json::from_value(v).map(ComputeKind::Vector),
                    "scalar" => serde_json::from_value(v).map(ComputeKind::Scalar),
                    "special" => serde_json::from_value(v).map(ComputeKind::Special),
                    "cim" => serde_json::from_value(v).map(ComputeKind::Cim),
                    _ => {
                        return Err(de::Error::unknown_variant(&kind, &["matrix", "vector", "scalar", "special", "cim"]));
                    }
                };
                r.map_err(|e| de::Error::custom(format!("{e} (in {kind} unit fields)")))
            }
        }
        d.deserialize_struct("ComputeKind", KIND_FIELDS, V)
    }
}

impl Serialize for ComputeKind {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let spec = match self {
            Self::Matrix(x) => serde_json::to_value(x),
            Self::Vector(x) => serde_json::to_value(x),
            Self::Scalar(x) => serde_json::to_value(x),
            Self::Special(x) => serde_json::to_value(x),
            Self::Cim(x) => serde_json::to_value(x),
        }
        .map_err(serde::ser::Error::custom)?;
        let mut map = match spec {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        map.insert("kind".into(), Value::from(self.name()));
        map.serialize(s)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixSpec {
    pub geometry: Geometry,
    /// None => geometry default (materialized during normalization).
    #[serde(default)]
    pub dataflow: Option<DataflowChoice>,
    #[serde(default)]
    pub sparsity: Vec<SparsitySupport>,
    #[serde(default)]
    pub accumulate_in: AccumulateIn,
    /// Per input role, the most consecutive temporal steps an operand stays in the unit between reads from its
    /// feed (an instruction's span: Hopper wgmma keeps `a` over N <= 256, i.e. 32 n8 steps; `mma.sync` re-reads
    /// every operand, 1). Roles not listed stay as long as the loop order allows (systolic stationarity).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub operand_run: BTreeMap<OperandRole, u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Geometry {
    Systolic { rows: u32, cols: u32 },
    Mma { m: u32, n: u32, k: u32 },
    OuterProduct { rows: u32, cols: u32 },
    Spatial { dims: BTreeMap<String, u32> },
}

impl Geometry {
    pub fn macs_per_cycle(&self) -> u64 {
        self.checked_macs_per_cycle().unwrap_or(u64::MAX)
    }

    pub fn checked_macs_per_cycle(&self) -> Option<u64> {
        match self {
            Self::Systolic { rows, cols } | Self::OuterProduct { rows, cols } => product([*rows, *cols]),
            Self::Mma { m, n, k } => product([*m, *n, *k]),
            Self::Spatial { dims } => product(dims.values().copied()),
        }
    }

    pub fn dims(&self) -> Vec<u32> {
        match self {
            Self::Systolic { rows, cols } | Self::OuterProduct { rows, cols } => vec![*rows, *cols],
            Self::Mma { m, n, k } => vec![*m, *n, *k],
            Self::Spatial { dims } => dims.values().copied().collect(),
        }
    }

    pub fn default_dataflow(&self) -> Dataflow {
        match self {
            Self::Systolic { .. } => Dataflow::WeightStationary,
            Self::Mma { .. } | Self::OuterProduct { .. } => Dataflow::OutputStationary,
            Self::Spatial { .. } => Dataflow::Any,
        }
    }

    /// Spatial unrolling of the reduction dim `k` (table 5.2), used for MX block divisibility.
    pub fn reduction_extent(&self) -> Option<u32> {
        match self {
            Self::Systolic { rows, .. } => Some(*rows),
            Self::Mma { k, .. } => Some(*k),
            Self::OuterProduct { .. } => None,
            Self::Spatial { dims } => dims.get("k").copied(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DataflowChoice {
    One(Dataflow),
    Set(Vec<Dataflow>),
}

impl DataflowChoice {
    pub fn all(&self) -> Vec<Dataflow> {
        match self {
            Self::One(d) => vec![*d],
            Self::Set(v) => v.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dataflow {
    WeightStationary,
    OutputStationary,
    InputStationary,
    RowStationary,
    Any,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsitySupport {
    /// `"2:4"`, `"1:4"`, `"block:<n>"`, `"unstructured"`.
    pub pattern: String,
    pub operand: OperandRole,
    pub speedup: f64,
    #[serde(default)]
    pub metadata_bits_per_nz: u32,
}

impl SparsitySupport {
    /// `(n, m)` of an `n:m` pattern.
    pub fn n_of_m(&self) -> Option<(u32, u32)> {
        let (n, m) = self.pattern.split_once(':')?;
        Some((n.parse().ok()?, m.parse().ok()?))
    }

    /// Largest speedup the pattern's structure supports: `m/n` for `n:m` (only the nonzeros are computed);
    /// `None` when the structure bounds nothing (`block:<n>`, `unstructured`).
    pub fn max_speedup(&self) -> Option<f64> {
        self.n_of_m().map(|(n, m)| f64::from(m) / f64::from(n))
    }

    /// Position bits each nonzero needs within its group of `m`.
    pub fn min_metadata_bits(&self) -> Option<u32> {
        self.n_of_m().map(|(_, m)| m.next_power_of_two().trailing_zeros())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccumulateIn {
    #[default]
    Local,
    Feed,
    Any,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSpec {
    pub lanes: u32,
    #[serde(default = "one")]
    pub sublanes: u32,
    #[serde(default)]
    pub class_rates: BTreeMap<OpClass, f64>,
    #[serde(default)]
    pub reduce_tree: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarSpec {
    #[serde(default = "one")]
    pub issue_width: u32,
    #[serde(default)]
    pub controls: Vec<Ref>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialSpec {
    pub lanes: u32,
    pub functions: Vec<SpecialFn>,
    #[serde(default)]
    pub fn_rates: BTreeMap<SpecialFn, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CimSpec {
    pub rows: u32,
    pub cols: u32,
    #[serde(default = "one")]
    pub cell_bits: u32,
    #[serde(default = "one")]
    pub input_bits_per_cycle: u32,
    #[serde(default)]
    pub parallel_rows: Option<u32>,
    #[serde(default = "one")]
    pub weight_sets: u32,
    #[serde(default)]
    pub weight_capacity: Option<Bytes>,
    #[serde(default)]
    pub style: CimStyle,
    #[serde(default)]
    pub adc_bits: Option<u32>,
    #[serde(default)]
    pub weight_write: Option<BytesPerSec>,
}

impl CimSpec {
    pub fn active_rows(&self) -> u32 {
        self.parallel_rows.unwrap_or(self.rows)
    }

    /// 01 §9.3: `parallel_rows * floor(cols / ceil(w_bits / cell_bits)) / ceil(in_bits / input_bits_per_cycle)`.
    pub fn macs_per_cycle(&self, in_bits: u32, w_bits: u32) -> f64 {
        let weights_per_row = self.cols / w_bits.div_ceil(self.cell_bits.max(1));
        let cycles = in_bits.div_ceil(self.input_bits_per_cycle.max(1));
        f64::from(self.active_rows()) * f64::from(weights_per_row) / f64::from(cycles)
    }

    /// Weight storage of the array: `rows * cols * cell_bits * weight_sets` bits; `None` beyond `u64` bytes.
    pub fn derived_capacity(&self) -> Option<Bytes> {
        let bits = [self.cols, self.cell_bits, self.weight_sets].into_iter().fold(u128::from(self.rows), |a, x| a * u128::from(x));
        u64::try_from(bits / 8).ok().map(Bytes)
    }

    /// Lossless ADC resolution for one analog conversion: `ceil(log2(parallel_rows * (2^cell_bits - 1) *
    /// (2^input_bits_per_cycle - 1) + 1))`.
    pub fn boundary_adc_bits(&self) -> u32 {
        let levels = |b: u32| (1u64 << b.min(32)) - 1;
        let max = u64::from(self.active_rows()) * levels(self.cell_bits) * levels(self.input_bits_per_cycle);
        (max + 1).next_power_of_two().trailing_zeros()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CimStyle {
    #[default]
    Digital,
    Analog,
}

/// A datapath mode. Authoring shorthands: `"bf16*bf16+fp32"`, `"int8*int8+int32@2"`, `"fp32@1"`, `"bf16@4"`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrecisionMode {
    Mac { a: PrecisionSpec, b: PrecisionSpec, acc: PrecisionSpec, out: Option<PrecisionSpec>, rate: f64 },
    Elem { dtype: PrecisionSpec, rate: f64 },
}

impl PrecisionMode {
    pub fn rate(&self) -> f64 {
        match self {
            Self::Mac { rate, .. } | Self::Elem { rate, .. } => *rate,
        }
    }

    pub fn operands(&self) -> Vec<Precision> {
        match self {
            Self::Mac { a, b, acc, out, .. } => {
                let mut v = vec![a.precision, b.precision, acc.precision];
                v.extend(out.map(|o| o.precision));
                v
            }
            Self::Elem { dtype, .. } => vec![dtype.precision],
        }
    }

    /// `(a, b, acc, out)` identity used for duplicate detection (W-IR-0313).
    pub fn key(&self) -> String {
        match self {
            Self::Mac { a, b, acc, out, .. } => format!("{a}*{b}+{acc}->{}", out.unwrap_or(*acc)),
            Self::Elem { dtype, .. } => dtype.to_string(),
        }
    }

    pub fn parse(s: &str) -> Result<Self, crate::common::Diagnostic> {
        let (body, rate) = match s.split_once('@') {
            Some((b, r)) => (
                b,
                r.trim().parse::<f64>().map_err(|_| {
                    crate::common::Diagnostic::error("E-IR-0103", format!("precision mode {s:?}: bad rate {r:?}"))
                        .hint("write the rate as a number, e.g. \"int8*int8+int32@2\"")
                })?,
            ),
            None => (s, 1.0),
        };
        let p = |x: &str| x.trim().parse::<PrecisionSpec>();
        match body.split_once('*') {
            Some((a, rest)) => {
                let (b, acc) = rest.split_once('+').ok_or_else(|| {
                    crate::common::Diagnostic::error("E-IR-0103", format!("precision mode {s:?} lacks '+<acc>'"))
                        .hint("MAC modes are written \"<a>*<b>+<acc>[@rate]\", e.g. \"bf16*bf16+fp32\"")
                })?;
                Ok(Self::Mac { a: p(a)?, b: p(b)?, acc: p(acc)?, out: None, rate })
            }
            None => Ok(Self::Elem { dtype: p(body)?, rate }),
        }
    }
}

impl fmt::Display for PrecisionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mac { a, b, acc, rate, .. } => write!(f, "{a}*{b}+{acc}@{rate}"),
            Self::Elem { dtype, rate } => write!(f, "{dtype}@{rate}"),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum ModeRepr {
    Short(String),
    Mac {
        a: PrecisionSpec,
        b: PrecisionSpec,
        acc: PrecisionSpec,
        #[serde(default)]
        out: Option<PrecisionSpec>,
        #[serde(default = "one_f")]
        rate: f64,
    },
    Elem {
        dtype: PrecisionSpec,
        #[serde(default = "one_f")]
        rate: f64,
    },
}

impl Serialize for PrecisionMode {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match *self {
            Self::Mac { a, b, acc, out, rate } => ModeRepr::Mac { a, b, acc, out, rate },
            Self::Elem { dtype, rate } => ModeRepr::Elem { dtype, rate },
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for PrecisionMode {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        if let Value::String(s) = &v {
            return Self::parse(s).map_err(super::diag::de_error);
        }
        match serde_json::from_value::<ModeRepr>(v).map_err(de::Error::custom)? {
            ModeRepr::Short(s) => Self::parse(&s).map_err(super::diag::de_error),
            ModeRepr::Mac { a, b, acc, out, rate } => Ok(Self::Mac { a, b, acc, out, rate }),
            ModeRepr::Elem { dtype, rate } => Ok(Self::Elem { dtype, rate }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperandRole {
    A,
    B,
    C,
    O,
    In,
    Out,
    Any,
}

impl OperandRole {
    pub fn is_input(self) -> bool {
        matches!(self, Self::A | Self::B | Self::C | Self::In | Self::Any)
    }
    pub fn is_output(self) -> bool {
        matches!(self, Self::O | Self::Out | Self::C | Self::Any)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feed {
    pub from: Ref,
    #[serde(default)]
    pub width_bits: Option<u32>,
    #[serde(default)]
    pub latency: Option<Cycles>,
    #[serde(default)]
    pub via: Option<Ref>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalBuffer {
    pub id: Id,
    pub holds: OperandRole,
    pub capacity: Bytes,
    #[serde(default)]
    pub double_buffered: bool,
    #[serde(default)]
    pub refill_from: Option<Ref>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pipeline {
    #[serde(default)]
    pub fill: Option<Cycles>,
    #[serde(default)]
    pub drain: Option<Cycles>,
    #[serde(default)]
    pub issue_overhead: Cycles,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NearBinding {
    pub memory: Ref,
    pub granularity: NearGranularity,
    #[serde(default)]
    pub internal_bandwidth: Option<BytesPerSec>,
    #[serde(default)]
    pub access_mode: NearAccessMode,
    #[serde(default)]
    pub residency: NearResidency,
    #[serde(default)]
    pub command_latency: Option<Seconds>,
    #[serde(default)]
    pub result_path: ResultPath,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NearGranularity {
    PerInstance,
    PerBank,
    PerBankGroup,
    PerPseudoChannel,
    PerChannel,
    PerStack,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NearAccessMode {
    #[default]
    Concurrent,
    ExclusiveAllBank,
    ExclusivePerBank,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NearResidency {
    AllOperands,
    #[default]
    Weights,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultPath {
    #[default]
    WriteBack,
    Channel,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicDie {
    #[serde(default)]
    pub tech: Option<TechRef>,
    #[serde(default)]
    pub area_budget: Option<Mm2>,
    #[serde(default)]
    pub power_budget: Option<Watts>,
    #[serde(flatten)]
    pub contents: Contents,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub kind: MemKind,
    pub capacity: Bytes,
    #[serde(default = "word32")]
    pub word_bits: u32,
    #[serde(default = "one")]
    pub banks: u32,
    #[serde(default)]
    pub bank_interleave: Option<Bytes>,
    pub ports: Vec<PortSpec>,
    #[serde(default)]
    pub operands: OperandPolicy,
    #[serde(default)]
    pub cache: Option<CacheSpec>,
    #[serde(default)]
    pub backing: Option<Selector>,
    #[serde(default)]
    pub implementation: MemImpl,
    #[serde(default)]
    pub bitcell: Option<Bitcell>,
    #[serde(default)]
    pub power_gated: bool,
    /// Register file intentionally fed by units of several cluster instances (W-IR-0410).
    #[serde(default)]
    pub shared_rf: bool,
    /// Visualizer grouping only (01 §7).
    #[serde(default)]
    pub level_hint: Option<u8>,
    #[serde(default)]
    pub clock: Option<ClockRef>,
    #[serde(default)]
    pub footprint: Option<Footprint>,
    #[serde(default)]
    pub placement: Placement,
    #[serde(default)]
    pub overrides: MemOverrides,
}

fn word32() -> u32 {
    32
}

impl Memory {
    /// Bits per cycle over all ports.
    pub fn port_bits_per_cycle(&self) -> u128 {
        self.ports
            .iter()
            .map(|p| u128::from(p.count) * if p.per_bank { u128::from(self.banks) } else { 1 } * u128::from(p.width_bits))
            .sum()
    }

    pub fn widest_port_bits(&self) -> u32 {
        self.ports.iter().map(|p| p.width_bits).max().unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemKind {
    RegisterFile,
    Scratchpad,
    Cache,
    Fifo,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortSpec {
    pub dir: PortDir,
    #[serde(default = "one")]
    pub count: u32,
    pub width_bits: u32,
    #[serde(default)]
    pub per_bank: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDir {
    Read,
    Write,
    Rw,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperandPolicy {
    #[default]
    Unified,
    Partitioned {
        parts: BTreeMap<OperandRole, Bytes>,
    },
    Carveout {
        options: Vec<CarveOption>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarveOption {
    pub scratch: Bytes,
    pub cache: Bytes,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheSpec {
    pub line: Bytes,
    #[serde(default = "one")]
    pub sectors: u32,
    pub ways: u32,
    #[serde(default)]
    pub write: WritePolicy,
    #[serde(default)]
    pub allocate: AllocPolicy,
    #[serde(default)]
    pub replacement: Replacement,
    #[serde(default)]
    pub coherent_with: Vec<Ref>,
    #[serde(default)]
    pub pinnable: Option<Bytes>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WritePolicy {
    #[default]
    WriteBack,
    WriteThrough,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocPolicy {
    #[default]
    WriteAllocate,
    NoWriteAllocate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Replacement {
    #[default]
    Lru,
    Plru,
    Random,
    Fifo,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemImpl {
    #[default]
    Auto,
    Flop,
    LatchArray,
    SramBanked,
    SramMacro,
    Edram,
    Mram,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bitcell {
    Hd,
    Hc,
    TwoPort,
}

/// Replace 04's derived values. `source` (a citation id) is an addition to 01 §7 so the `reference` profile's
/// E-IR-1103 rule can be satisfied for memory overrides.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemOverrides {
    #[serde(default)]
    pub read_energy: Option<JoulesPerByte>,
    #[serde(default)]
    pub write_energy: Option<JoulesPerByte>,
    #[serde(default)]
    pub latency: Option<Cycles>,
    #[serde(default)]
    pub leakage: Option<Watts>,
    #[serde(default)]
    pub area: Option<Mm2>,
    #[serde(default)]
    pub bandwidth: Option<BytesPerSec>,
    #[serde(default)]
    pub source: Option<String>,
}

impl MemOverrides {
    pub fn performance_fields(&self) -> Vec<&'static str> {
        let mut v = vec![];
        let fields = [
            ("read_energy", self.read_energy.is_some()),
            ("write_energy", self.write_energy.is_some()),
            ("latency", self.latency.is_some()),
            ("leakage", self.leakage.is_some()),
            ("area", self.area.is_some()),
            ("bandwidth", self.bandwidth.is_some()),
        ];
        v.extend(fields.iter().filter(|(_, set)| *set).map(|(n, _)| *n));
        v
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemStack {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub kind: DramKind,
    pub capacity: Bytes,
    pub io_width_bits: u32,
    pub pin_rate_bits_per_s: BitsPerSec,
    #[serde(default)]
    pub channels: Option<u32>,
    #[serde(default)]
    pub pseudo_channels_per_channel: Option<u32>,
    #[serde(default)]
    pub banks_per_pc: Option<u32>,
    #[serde(default)]
    pub dies_high: Option<u32>,
    #[serde(default)]
    pub row_bytes: Option<Bytes>,
    #[serde(default)]
    pub timing: Option<DramTiming>,
    /// None is E-IR-0501.
    #[serde(default)]
    pub attach: Option<StackAttach>,
    #[serde(default)]
    pub logic_die: Option<LogicDie>,
    #[serde(default)]
    pub site: Placement,
    #[serde(default)]
    pub layer: Option<LayerRef>,
    #[serde(default)]
    pub over: Option<Ref>,
    #[serde(default)]
    pub clock: Option<ClockRef>,
    #[serde(default)]
    pub footprint: Option<Footprint>,
    #[serde(default)]
    pub overrides: StackOverrides,
}

impl MemStack {
    /// Structural peak bandwidth per stack: `io_width_bits * pin_rate / 8`.
    pub fn derived_bandwidth(&self) -> BytesPerSec {
        BytesPerSec(f64::from(self.io_width_bits) * self.pin_rate_bits_per_s.0 / 8.0)
    }

    pub fn bandwidth(&self) -> BytesPerSec {
        self.overrides.bandwidth.unwrap_or_else(|| self.derived_bandwidth())
    }

    pub fn channel_count(&self) -> u32 {
        self.channels.unwrap_or_else(|| self.kind.default_channels())
    }

    pub fn pseudo_channels(&self) -> u32 {
        self.channel_count() * self.pseudo_channels_per_channel.unwrap_or_else(|| self.kind.default_pcs_per_channel())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DramKind {
    Hbm2,
    Hbm2e,
    Hbm3,
    Hbm3e,
    Hbm4,
    Hbm4e,
    Lpddr4x,
    Lpddr5,
    Lpddr5x,
    Lpddr6,
    Gddr6,
    Gddr6x,
    Gddr7,
    Ddr5,
    StackedDram,
    Custom,
}

impl DramKind {
    pub fn default_channels(self) -> u32 {
        match self {
            Self::Hbm2 | Self::Hbm2e => 8,
            Self::Hbm3 | Self::Hbm3e => 16,
            Self::Hbm4 | Self::Hbm4e => 32,
            Self::Gddr6 | Self::Gddr6x | Self::Gddr7 | Self::Ddr5 => 2,
            _ => 1,
        }
    }

    pub fn default_pcs_per_channel(self) -> u32 {
        match self {
            Self::Hbm2 | Self::Hbm2e | Self::Hbm3 | Self::Hbm3e | Self::Hbm4 | Self::Hbm4e => 2,
            _ => 1,
        }
    }

    /// Known per-stack/device capacity range in bytes and max pin rate (bit/s), for W-IR-0502/0503.
    /// Assumed sanity bounds from JEDEC generations; 04's table supersedes them once it lands.
    pub fn sanity(self) -> Option<(u64, u64, f64)> {
        const G: u64 = 1 << 30;
        Some(match self {
            Self::Hbm2 => (G, 8 * G, 2.4e9 * 1.05),
            Self::Hbm2e => (4 * G, 24 * G, 3.6e9),
            Self::Hbm3 => (8 * G, 32 * G, 6.4e9),
            Self::Hbm3e => (16 * G, 48 * G, 9.8e9),
            Self::Hbm4 | Self::Hbm4e => (16 * G, 64 * G, 12e9),
            Self::Lpddr5 | Self::Lpddr5x => (G / 2, 64 * G, 10.7e9),
            Self::Gddr6 => (G / 2, 4 * G, 20e9),
            Self::Gddr6x => (G, 4 * G, 24e9),
            Self::Gddr7 => (G, 8 * G, 40e9),
            _ => return None,
        })
    }

    pub fn is_hbm(self) -> bool {
        matches!(self, Self::Hbm2 | Self::Hbm2e | Self::Hbm3 | Self::Hbm3e | Self::Hbm4 | Self::Hbm4e)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DramTiming {
    #[serde(default)]
    pub t_rcd: Option<Seconds>,
    #[serde(default)]
    pub t_rp: Option<Seconds>,
    #[serde(default)]
    pub t_cl: Option<Seconds>,
    #[serde(default)]
    pub t_ras: Option<Seconds>,
    #[serde(default)]
    pub t_rrd: Option<Seconds>,
    #[serde(default)]
    pub t_faw: Option<Seconds>,
    #[serde(default)]
    pub t_rfc: Option<Seconds>,
    #[serde(default)]
    pub t_refi: Option<Seconds>,
    #[serde(default)]
    pub t_ccd_l: Option<Seconds>,
    #[serde(default)]
    pub t_ccd_s: Option<Seconds>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum StackAttach {
    Phys { phys: Vec<Ref> },
    Controllers { controllers: Vec<Ref> },
    Network { network: Ref },
    Vertical { die: Ref },
}

/// `source` is an addition (see [`MemOverrides`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackOverrides {
    #[serde(default)]
    pub bandwidth: Option<BytesPerSec>,
    #[serde(default)]
    pub energy_per_byte: Option<JoulesPerByte>,
    #[serde(default)]
    pub latency: Option<Seconds>,
    #[serde(default)]
    pub power: Option<Watts>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemControllerSpec {
    pub serves: Ref,
    #[serde(default)]
    pub channels: Option<u32>,
    #[serde(default)]
    pub width_bits: Option<u32>,
    #[serde(default)]
    pub queue_depth: Option<u32>,
    #[serde(default)]
    pub scheduler: DramScheduler,
    pub endpoint_of: Vec<Ref>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DramScheduler {
    #[default]
    FrFcfs,
    Fcfs,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DmaSpec {
    pub endpoint_of: Vec<Ref>,
    #[serde(default)]
    pub bandwidth: Option<BytesPerSec>,
    #[serde(default)]
    pub outstanding: Option<u32>,
    #[serde(default)]
    pub scope: Option<Selector>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cim(v: Value) -> CimSpec {
        serde_json::from_value(v).unwrap()
    }

    fn mode(s: &str) -> PrecisionMode {
        PrecisionMode::parse(s).unwrap()
    }

    #[test]
    fn cim_throughput_is_derived_per_precision() {
        let digital = ComputeKind::Cim(cim(serde_json::json!({ "rows": 256, "cols": 256 })));
        for (m, macs) in [
            ("int8*int4+int32", 256.0 * 64.0 / 8.0),
            ("int8*int8+int32", 256.0 * 32.0 / 8.0),
            ("int4*int4+int32", 256.0 * 64.0 / 4.0),
            ("int8*int8+int32@0.5", 256.0 * 32.0 / 16.0),
            ("bf16*bf16+fp32", 256.0 * 16.0 / 16.0),
        ] {
            assert_eq!(digital.ops_per_cycle(&mode(m)), macs, "{m}");
        }
        let wide = ComputeKind::Cim(cim(serde_json::json!({ "rows": 256, "cols": 256, "cell_bits": 2, "input_bits_per_cycle": 4, "parallel_rows": 64 })));
        assert_eq!(wide.ops_per_cycle(&mode("int8*int4+int32")), 64.0 * 128.0 / 2.0);
        let multi = ComputeKind::Cim(cim(serde_json::json!({ "rows": 64, "cols": 64, "cell_bits": 8 })));
        assert_eq!(multi.ops_per_cycle(&mode("int4*int4+int32")), 64.0 * 64.0 / 4.0, "a narrow weight still occupies a whole cell");
    }

    #[test]
    fn cim_capacity_and_adc_boundary() {
        let c = cim(serde_json::json!({ "rows": 256, "cols": 256, "weight_sets": 4 }));
        assert_eq!(c.derived_capacity(), Some(Bytes(32 << 10)));
        assert_eq!(c.boundary_adc_bits(), 9);
        let a = cim(serde_json::json!({ "rows": 256, "cols": 64, "parallel_rows": 256, "input_bits_per_cycle": 4, "style": "analog", "adc_bits": 6 }));
        assert_eq!(a.boundary_adc_bits(), 12, "ASiM 2411.11022: 256 rows x 4b inputs span 0..3840");
    }
}
