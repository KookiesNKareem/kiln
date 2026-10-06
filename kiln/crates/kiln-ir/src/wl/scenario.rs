//! Sequence batches, phase instances and scenarios (02 §8).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::dim::DimExpr;
use super::dtype::ElemType;
use crate::common::{Diagnostic, Id};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub count: u64,
    pub q_len: u64,
    pub kv_len: u64,
    #[serde(default)]
    pub tag: Option<String>,
}

impl Segment {
    pub fn new(count: u64, q_len: u64, kv_len: u64) -> Self {
        Self {
            count,
            q_len,
            kv_len,
            tag: None,
        }
    }

    pub fn past(&self) -> u64 {
        self.kv_len - self.q_len
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeqBatch {
    pub segments: Vec<Segment>,
}

impl SeqBatch {
    pub fn uniform(count: u64, q_len: u64, kv_len: u64) -> Self {
        Self {
            segments: vec![Segment::new(count, q_len, kv_len)],
        }
    }

    /// Tokens this step: Σ count·q_len.
    pub fn tokens(&self) -> u64 {
        self.segments.iter().fold(0u64, |t, s| t.saturating_add(s.count.saturating_mul(s.q_len)))
    }

    /// Sequences this step: Σ count.
    pub fn seqs(&self) -> u64 {
        self.segments.iter().fold(0u64, |t, s| t.saturating_add(s.count))
    }

    pub fn max_kv(&self) -> u64 {
        self.segments.iter().map(|s| s.kv_len).max().unwrap_or(0)
    }

    pub fn max_q(&self) -> u64 {
        self.segments.iter().map(|s| s.q_len).max().unwrap_or(0)
    }

    /// Segment invariants (E-WL-SCN-002): count ≥ 1, q_len ≥ 1, kv_len ≥ q_len, at least one segment, and
    /// Σ count·kv_len (which bounds tokens, sequences and merged counts) within `i64`.
    pub fn check(&self) -> Result<(), Diagnostic> {
        if self.segments.is_empty() {
            return Err(Diagnostic::error(
                "E-WL-SCN-002",
                "sequence batch has no segments",
            ));
        }
        for (i, s) in self.segments.iter().enumerate() {
            if s.count == 0 || s.q_len == 0 || s.kv_len < s.q_len {
                return Err(Diagnostic::error(
                    "E-WL-SCN-002",
                    format!("segment {i} {{count {}, q_len {}, kv_len {}}} violates count ≥ 1, q_len ≥ 1, kv_len ≥ q_len", s.count, s.q_len, s.kv_len),
                )
                .at(format!("segments.{i}"))
                .hint("kv_len includes the tokens processed this step"));
            }
        }
        let total = self.segments.iter().try_fold(0u64, |t, s| s.count.checked_mul(s.kv_len).and_then(|n| t.checked_add(n)));
        if total.is_none_or(|t| t > i64::MAX as u64) {
            return Err(Diagnostic::error("E-WL-SCN-002", format!("sequence batch Σ count·kv_len exceeds {}", i64::MAX))
                .hint("token, sequence and KV-slot counts must fit in i64"));
        }
        Ok(())
    }

    /// Canonical order: sorted by (q_len, kv_len, tag), equal entries merged.
    pub fn canonical(&self) -> Self {
        let mut segs = self.segments.clone();
        segs.sort_by(|a, b| (a.q_len, a.kv_len, &a.tag).cmp(&(b.q_len, b.kv_len, &b.tag)));
        let mut out: Vec<Segment> = Vec::with_capacity(segs.len());
        for s in segs {
            match out.last_mut() {
                Some(l) if (l.q_len, l.kv_len, &l.tag) == (s.q_len, s.kv_len, &s.tag) => {
                    l.count = l.count.saturating_add(s.count)
                }
                _ => out.push(s),
            }
        }
        Self { segments: out }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    Prefill,
    Decode,
    Mixed,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseInstance {
    pub kind: PhaseKind,
    pub entry: Id,
    pub seqs: SeqBatch,
    pub bindings: IndexMap<String, u64>,
    /// Exact rational occurrence count.
    pub multiplicity: DimExpr,
    pub index: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheState {
    Cold,
    Warm,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalMode {
    #[default]
    WholeGraph,
    Isolated {
        cache: CacheState,
        launch: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringScope {
    #[default]
    Step,
    Layer,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HostModel {
    pub per_step_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingModel {
    Uniform,
    Zipf {
        s: f64,
        seed: u64,
    },
    Histogram {
        per_layer: bool,
        loads: Vec<Vec<f64>>,
    },
    WorstCase,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeSampling {
    Exact,
    Points { n: u32, max_rel_err: f64 },
}

impl Default for DecodeSampling {
    fn default() -> Self {
        Self::Points {
            n: 3,
            max_rel_err: 0.005,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LenDist {
    Fixed(u64),
    Uniform {
        lo: u64,
        hi: u64,
    },
    Normal {
        mean: f64,
        std: f64,
        lo: u64,
        hi: u64,
    },
    Empirical {
        name: String,
        hash: String,
        values: Vec<u64>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestClass {
    pub weight: f64,
    pub prompt: LenDist,
    pub output: LenDist,
    #[serde(default)]
    pub prefix_cached: Option<LenDist>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arrival {
    ClosedLoop { concurrency: u64 },
    Poisson { rate_per_s: f64 },
    AllAtZero,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedPolicy {
    #[default]
    VllmV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousSpec {
    pub classes: Vec<RequestClass>,
    pub arrival: Arrival,
    pub n_requests: u64,
    pub max_num_seqs: u64,
    pub max_batched_tokens: u64,
    pub chunked_prefill: bool,
    #[serde(default)]
    pub policy: SchedPolicy,
    pub kv_bucket: u64,
    pub warmup_requests: u64,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioMode {
    Snapshot {
        kind: PhaseKind,
        seqs: SeqBatch,
    },
    Static {
        batch: u64,
        prompt_len: u64,
        gen_len: u64,
        #[serde(default)]
        prefix_cached: u64,
        #[serde(default)]
        decode_sampling: DecodeSampling,
    },
    Continuous(ContinuousSpec),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub mode: ScenarioMode,
    #[serde(default)]
    pub bindings: IndexMap<String, u64>,
    #[serde(default)]
    pub routing: Option<RoutingModel>,
    #[serde(default)]
    pub eval_mode: EvalMode,
    #[serde(default)]
    pub mfu_ref_dtype: Option<ElemType>,
    #[serde(default)]
    pub host: HostModel,
    #[serde(default)]
    pub scope: ScoringScope,
}

impl Scenario {
    pub fn snapshot(kind: PhaseKind, seqs: SeqBatch) -> Self {
        Self {
            mode: ScenarioMode::Snapshot { kind, seqs },
            bindings: IndexMap::new(),
            routing: None,
            eval_mode: EvalMode::WholeGraph,
            mfu_ref_dtype: None,
            host: HostModel::default(),
            scope: ScoringScope::Step,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_counts_must_fit() {
        let big = SeqBatch::uniform(1 << 63, 2, 2);
        assert_eq!(big.check().unwrap_err().code, "E-WL-SCN-002");
        assert_eq!(big.tokens(), u64::MAX, "saturates rather than wrapping to 0");
        let split = SeqBatch { segments: vec![Segment::new(1 << 62, 1, 1), Segment::new(1 << 62, 1, 1), Segment::new(1 << 62, 1, 2)] };
        assert!(split.check().is_err(), "sum of counts overflows i64");
        assert_eq!(split.canonical().seqs(), 3 << 62);
        assert!(SeqBatch::uniform(1 << 20, 4096, 8192).check().is_ok());
    }
}
