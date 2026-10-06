//! Scenario expansion to phase instances (02 §8.4).

use kiln_ir::common::Diagnostic;
use kiln_ir::wl::*;

/// Decode step indices evaluated under `Points{n}`: n evenly spread over `1..=steps`, both ends included. The
/// residual-driven refinement of 02 §8.4 needs step times and belongs to the engine.
pub fn decode_points(steps: u64, n: u32) -> Vec<u64> {
    let n = u64::from(n.max(2));
    if steps <= n {
        return (1..=steps).collect();
    }
    let mut v: Vec<u64> = (0..n)
        .map(|j| 1 + (j * (steps - 1) + (n - 1) / 2) / (n - 1))
        .collect();
    v.dedup();
    v
}

pub fn expand(s: &Scenario, model: &Model) -> Result<Vec<PhaseInstance>, Diagnostic> {
    let entry = model.entry.forward.clone();
    let one = DimExpr::int(1);
    match &s.mode {
        ScenarioMode::Snapshot { kind, seqs } => {
            seqs.check()?;
            Ok(vec![PhaseInstance {
                kind: *kind,
                entry,
                seqs: seqs.canonical(),
                bindings: s.bindings.clone(),
                multiplicity: one,
                index: None,
            }])
        }
        ScenarioMode::Static {
            batch,
            prompt_len,
            gen_len,
            prefix_cached,
            decode_sampling,
        } => {
            let (b, p, g, c) = (*batch, *prompt_len, *gen_len, *prefix_cached);
            if b == 0 || p == 0 || g == 0 || c >= p {
                return Err(Diagnostic::error(
                    "E-WL-SCN-001",
                    "static scenario needs batch, prompt_len, gen_len ≥ 1 and prefix_cached < prompt_len",
                ));
            }
            let mut bindings = s.bindings.clone();
            bindings.entry("kv_cap".into()).or_insert(p + g - 1);
            let mut out = vec![PhaseInstance {
                kind: PhaseKind::Prefill,
                entry: entry.clone(),
                seqs: SeqBatch::uniform(b, p - c, p),
                bindings: bindings.clone(),
                multiplicity: one.clone(),
                index: None,
            }];
            let steps = g - 1;
            let idx: Vec<u64> = match decode_sampling {
                DecodeSampling::Exact => (1..=steps).collect(),
                DecodeSampling::Points { n, .. } => decode_points(steps, *n),
            };
            out.extend(idx.into_iter().map(|i| PhaseInstance {
                kind: PhaseKind::Decode,
                entry: entry.clone(),
                seqs: SeqBatch::uniform(b, 1, p + i),
                bindings: bindings.clone(),
                multiplicity: one.clone(),
                index: Some(i),
            }));
            Ok(out)
        }
        ScenarioMode::Continuous(_) => Err(Diagnostic::error(
            "E-WL-SCN-001",
            "continuous scenarios are expanded online by kiln-wl::serve (deferred past M0)",
        )),
    }
}
