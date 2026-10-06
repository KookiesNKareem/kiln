//! Evolution archive (05 §3.9, consumed by 06 §6.9): `archive.json`, `designs.arrow`, `generations.arrow`,
//! `runs/<design_id>.kiln`.
//!
//! Writers may either append (each append is one complete Arrow IPC *stream*: schema, batches, end-of-stream
//! marker, written to the end of the file) or rewrite the file atomically (temp file + rename) in either IPC
//! stream or IPC file format. Readers accept all three and tolerate a truncated last stream. Columns are
//! matched by name: extra columns are ignored, missing nullable columns read as null, and any integer width,
//! utf8/large_utf8/utf8_view and list/large_list are accepted, so pyarrow and polars writers need no casts.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;
use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::arrowx::{self, Table};

pub const ARCHIVE_SCHEMA: &str = "kiln.archive/1";
/// Spelling of the first draft of this schema, still accepted on read.
const ARCHIVE_FORMAT_V0: &str = "kiln-archive";
const TABLE_VERSION: &str = "1.0";
pub const ARCHIVE_JSON: &str = "archive.json";
pub const DESIGNS: &str = "designs.arrow";
pub const GENERATIONS: &str = "generations.arrow";
pub const RUNS_DIR: &str = "runs";
const CODE: &str = "E-ARCHIVE";

fn err(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(CODE, msg)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    #[default]
    Maximize,
    Minimize,
}

/// One MAP-Elites descriptor axis. Binning is the evolution loop's choice (06 §6.3): either `edges`
/// (explicit, `bins + 1` values) or `range` + `bins` (uniform, in log space when `log`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DescriptorAxis {
    pub name: String,
    #[serde(default)]
    pub unit: String,
    #[serde(default)]
    pub log: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bins: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<Vec<f64>>,
}

impl DescriptorAxis {
    pub fn n_bins(&self) -> u32 {
        match (&self.edges, self.bins) {
            (Some(e), _) if e.len() >= 2 => e.len() as u32 - 1,
            (_, Some(b)) => b,
            _ => 1,
        }
    }

    /// Lower edge of every bin plus the top edge.
    pub fn bin_edges(&self) -> Vec<f64> {
        if let Some(e) = &self.edges {
            return e.clone();
        }
        let n = self.n_bins();
        let [lo, hi] = self.range.unwrap_or([0.0, 1.0]);
        (0..=n)
            .map(|i| {
                let f = f64::from(i) / f64::from(n);
                if self.log && lo > 0.0 && hi > 0.0 {
                    (lo.ln() + f * (hi.ln() - lo.ln())).exp()
                } else {
                    lo + f * (hi - lo)
                }
            })
            .collect()
    }

    /// Bin of a value, clamped to the axis.
    pub fn bin_of(&self, x: f64) -> u32 {
        let e = self.bin_edges();
        let n = self.n_bins();
        (1..e.len())
            .find(|&i| x < e[i])
            .map_or(n.saturating_sub(1), |i| (i - 1) as u32)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FitnessSpec {
    pub name: String,
    #[serde(default)]
    pub direction: Direction,
}

/// `archive.json`. Unknown keys (e.g. a driver's `kiln` provenance block) are ignored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArchiveMeta {
    #[serde(alias = "format")]
    pub schema: String,
    #[serde(alias = "descriptors")]
    pub axes: Vec<DescriptorAxis>,
    pub fitness: FitnessSpec,
    #[serde(default)]
    pub run_config: serde_json::Value,
}

impl ArchiveMeta {
    pub fn new(axes: Vec<DescriptorAxis>, fitness: FitnessSpec) -> Self {
        Self {
            schema: ARCHIVE_SCHEMA.into(),
            axes,
            fitness,
            run_config: serde_json::Value::Null,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesignStatus {
    Elite,
    Displaced,
    Invalid,
    Failed,
}

impl DesignStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DesignStatus::Elite => "elite",
            DesignStatus::Displaced => "displaced",
            DesignStatus::Invalid => "invalid",
            DesignStatus::Failed => "failed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "elite" => DesignStatus::Elite,
            "displaced" => DesignStatus::Displaced,
            "invalid" => DesignStatus::Invalid,
            "failed" => DesignStatus::Failed,
            _ => return None,
        })
    }
}

/// One row of `designs.arrow`: every evaluated design, not only elites. A later row with the same
/// `design_id` supersedes earlier ones (status changes such as elite -> displaced are appended).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DesignRecord {
    pub design_id: String,
    pub generation: u32,
    pub parent_ids: Vec<String>,
    pub mutation_summary: String,
    pub operator: String,
    /// Descriptor values in `archive.json` axis order.
    pub descriptor_values: Vec<f64>,
    /// Bin index per axis.
    pub cell: Vec<u32>,
    pub fitness: f64,
    pub fitness_components: BTreeMap<String, f64>,
    pub status: DesignStatus,
    /// Relative to the archive directory, e.g. `runs/d17.kiln`.
    pub run_path: Option<String>,
    pub thumbnail_path: Option<String>,
    pub design_hash: String,
    pub wall_time: f64,
    // 06 §6.9 (nullable).
    pub trust_level: Option<String>,
    pub audit_status: Option<String>,
    pub heldout_score: Option<f64>,
    pub calibration_set_hash: Option<String>,
    pub kiln_git_hash: Option<String>,
    pub stage_reached: Option<String>,
    pub extrapolated: Option<Vec<String>>,
    pub fitness_low: Option<f64>,
    pub fitness_high: Option<f64>,
    pub interval_method: Option<String>,
}

/// One row of `generations.arrow`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation: u32,
    pub best: f64,
    pub median: f64,
    pub qd_score: f64,
    pub coverage: f64,
    pub evaluations: u64,
    pub invalid_count: u64,
    pub best_low: Option<f64>,
    pub best_high: Option<f64>,
}

pub fn designs_batch(rows: &[DesignRecord]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push(
        "design_id",
        false,
        utf8s(rows.iter().map(|r| r.design_id.as_str())),
    )
    .push(
        "generation",
        false,
        u32s(rows.iter().map(|r| Some(r.generation))),
    )
    .push(
        "parent_ids",
        false,
        list_utf8(rows.iter().map(|r| Some(&r.parent_ids[..]))),
    )
    .push(
        "mutation_summary",
        false,
        utf8s(rows.iter().map(|r| r.mutation_summary.as_str())),
    )
    .push(
        "operator",
        false,
        utf8s(rows.iter().map(|r| r.operator.as_str())),
    )
    .push(
        "descriptor_values",
        false,
        list_f64(rows.iter().map(|r| Some(&r.descriptor_values[..]))),
    )
    .push(
        "cell",
        false,
        list_u32(rows.iter().map(|r| Some(&r.cell[..]))),
    )
    .push("fitness", false, f64v(rows.iter().map(|r| r.fitness)))
    .push(
        "fitness_components",
        false,
        map_utf8_f64(
            rows.iter()
                .map(|r| r.fitness_components.iter().map(|(k, v)| (k.as_str(), *v))),
        ),
    )
    .push(
        "status",
        false,
        utf8s(rows.iter().map(|r| r.status.as_str())),
    )
    .push(
        "run_path",
        true,
        utf8(rows.iter().map(|r| r.run_path.as_deref())),
    )
    .push(
        "thumbnail_path",
        true,
        utf8(rows.iter().map(|r| r.thumbnail_path.as_deref())),
    )
    .push(
        "design_hash",
        false,
        utf8s(rows.iter().map(|r| r.design_hash.as_str())),
    )
    .push("wall_time", false, f64v(rows.iter().map(|r| r.wall_time)))
    .push(
        "trust_level",
        true,
        utf8(rows.iter().map(|r| r.trust_level.as_deref())),
    )
    .push(
        "audit_status",
        true,
        utf8(rows.iter().map(|r| r.audit_status.as_deref())),
    )
    .push(
        "heldout_score",
        true,
        f64s(rows.iter().map(|r| r.heldout_score)),
    )
    .push(
        "calibration_set_hash",
        true,
        utf8(rows.iter().map(|r| r.calibration_set_hash.as_deref())),
    )
    .push(
        "kiln_git_hash",
        true,
        utf8(rows.iter().map(|r| r.kiln_git_hash.as_deref())),
    )
    .push(
        "stage_reached",
        true,
        utf8(rows.iter().map(|r| r.stage_reached.as_deref())),
    )
    .push(
        "extrapolated",
        true,
        list_utf8(rows.iter().map(|r| r.extrapolated.as_deref())),
    )
    .push(
        "fitness_low",
        true,
        f64s(rows.iter().map(|r| r.fitness_low)),
    )
    .push(
        "fitness_high",
        true,
        f64s(rows.iter().map(|r| r.fitness_high)),
    )
    .push(
        "interval_method",
        true,
        utf8(rows.iter().map(|r| r.interval_method.as_deref())),
    );
    c.batch("designs", TABLE_VERSION)
}

pub fn generations_batch(rows: &[GenerationRecord]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push(
        "generation",
        false,
        u32s(rows.iter().map(|r| Some(r.generation))),
    )
    .push("best", false, f64v(rows.iter().map(|r| r.best)))
    .push("median", false, f64v(rows.iter().map(|r| r.median)))
    .push("qd_score", false, f64v(rows.iter().map(|r| r.qd_score)))
    .push("coverage", false, f64v(rows.iter().map(|r| r.coverage)))
    .push(
        "evaluations",
        false,
        u64s(rows.iter().map(|r| Some(r.evaluations))),
    )
    .push(
        "invalid_count",
        false,
        u64s(rows.iter().map(|r| Some(r.invalid_count))),
    )
    .push("best_low", true, f64s(rows.iter().map(|r| r.best_low)))
    .push("best_high", true, f64s(rows.iter().map(|r| r.best_high)));
    c.batch("generations", TABLE_VERSION)
}

pub fn read_designs(batches: &[RecordBatch]) -> Result<Vec<DesignRecord>, Diagnostic> {
    let t = Table::new("designs", batches);
    let n = t.rows();
    let ids = t.str("design_id")?;
    let generation = t.u32("generation")?;
    let parents = t.list_str("parent_ids")?;
    let summary = t.opt_str("mutation_summary")?;
    let operator = t.opt_str("operator")?;
    let desc = match t.has("descriptor_values") {
        true => t.list_f64("descriptor_values")?,
        false => t.list_f64("descriptors")?,
    };
    let cell = t.list_u32("cell")?;
    let fitness = t.f64("fitness")?;
    // A map<utf8, f64>, or a JSON object per row as utf8.
    let comps: Vec<Vec<(String, f64)>> = match t.map_str_f64("fitness_components") {
        Ok(m) => m,
        Err(_) => t
            .opt_str("fitness_components")?
            .into_iter()
            .map(|j| {
                j.and_then(|j| serde_json::from_str::<BTreeMap<String, f64>>(&j).ok())
                    .map(|m| m.into_iter().collect())
                    .unwrap_or_default()
            })
            .collect(),
    };
    let status = t.str("status")?;
    let run_path = t.opt_str("run_path")?;
    let thumb = t.opt_str("thumbnail_path")?;
    let hash = t.opt_str("design_hash")?;
    let wall = match t.has("wall_time") {
        true => t.opt_f64("wall_time")?,
        false => t.opt_f64("wall_time_s")?,
    };
    let trust = t.opt_str("trust_level")?;
    let audit = t.opt_str("audit_status")?;
    let heldout = t.opt_f64("heldout_score")?;
    let calib = t.opt_str("calibration_set_hash")?;
    let git = t.opt_str("kiln_git_hash")?;
    let stage = t.opt_str("stage_reached")?;
    let extrap = t.list_str("extrapolated")?;
    let lo = t.opt_f64("fitness_low")?;
    let hi = t.opt_f64("fitness_high")?;
    let method = t.opt_str("interval_method")?;
    (0..n)
        .map(|i| {
            Ok(DesignRecord {
                design_id: ids[i].clone(),
                generation: generation[i],
                parent_ids: parents[i].clone().unwrap_or_default(),
                mutation_summary: summary[i].clone().unwrap_or_default(),
                operator: operator[i].clone().unwrap_or_default(),
                descriptor_values: desc[i].clone().unwrap_or_default(),
                cell: cell[i].clone().unwrap_or_default(),
                fitness: fitness[i],
                fitness_components: comps[i].iter().cloned().collect(),
                status: DesignStatus::parse(&status[i]).ok_or_else(|| {
                    err(format!("design {}: unknown status {:?}", ids[i], status[i]))
                        .hint("status is one of elite, displaced, invalid, failed")
                })?,
                run_path: run_path[i].clone(),
                thumbnail_path: thumb[i].clone(),
                design_hash: hash[i].clone().unwrap_or_default(),
                wall_time: wall[i].unwrap_or(0.0),
                trust_level: trust[i].clone(),
                audit_status: audit[i].clone(),
                heldout_score: heldout[i],
                calibration_set_hash: calib[i].clone(),
                kiln_git_hash: git[i].clone(),
                stage_reached: stage[i].clone(),
                extrapolated: extrap[i].clone(),
                fitness_low: lo[i],
                fitness_high: hi[i],
                interval_method: method[i].clone(),
            })
        })
        .collect()
}

pub fn read_generations(batches: &[RecordBatch]) -> Result<Vec<GenerationRecord>, Diagnostic> {
    let t = Table::new("generations", batches);
    let g = t.u32("generation")?;
    let best = t.f64("best")?;
    let median = t.opt_f64("median")?;
    let qd = t.opt_f64("qd_score")?;
    let cov = t.opt_f64("coverage")?;
    let ev = t.opt_u64("evaluations")?;
    let inv = t.opt_u64("invalid_count")?;
    let bl = t.opt_f64("best_low")?;
    let bh = t.opt_f64("best_high")?;
    Ok((0..t.rows())
        .map(|i| GenerationRecord {
            generation: g[i],
            best: best[i],
            median: median[i].unwrap_or(f64::NAN),
            qd_score: qd[i].unwrap_or(f64::NAN),
            coverage: cov[i].unwrap_or(f64::NAN),
            evaluations: ev[i].unwrap_or(0),
            invalid_count: inv[i].unwrap_or(0),
            best_low: bl[i],
            best_high: bh[i],
        })
        .collect())
}

/// An archive directory read into memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Archive {
    pub dir: PathBuf,
    pub meta: ArchiveMeta,
    /// All rows in file order (history); see [`Archive::latest`].
    pub designs: Vec<DesignRecord>,
    pub generations: Vec<GenerationRecord>,
    /// A file ended in a truncated stream (a writer is mid-append or died).
    pub partial: bool,
}

impl Archive {
    pub fn read_dir(dir: &Path) -> Result<Archive, Diagnostic> {
        let at = |p: &Path| p.display().to_string();
        let meta_path = dir.join(ARCHIVE_JSON);
        let text = std::fs::read(&meta_path).map_err(|e| {
            err(format!("cannot read {}: {e}", at(&meta_path)))
                .hint("an archive directory holds archive.json, designs.arrow and generations.arrow (05 §3.9)")
        })?;
        let meta: ArchiveMeta = serde_json::from_slice(&text).map_err(|e| {
            err(format!(
                "{} is not a kiln archive manifest: {e}",
                at(&meta_path)
            ))
        })?;
        if meta.schema != ARCHIVE_SCHEMA && meta.schema != ARCHIVE_FORMAT_V0 {
            return Err(err(format!(
                "{}: schema {:?}, expected {ARCHIVE_SCHEMA:?}",
                at(&meta_path),
                meta.schema
            )));
        }
        let mut partial = false;
        let mut load = |name: &str| -> Result<Vec<RecordBatch>, Diagnostic> {
            let p = dir.join(name);
            match std::fs::read(&p) {
                Ok(b) if b.is_empty() => Ok(vec![]),
                Ok(b) => {
                    let (batches, part) = arrowx::read_ipc(&b).map_err(|d| d.at(at(&p)))?;
                    partial |= part;
                    Ok(batches)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
                Err(e) => Err(err(format!("cannot read {}: {e}", at(&p)))),
            }
        };
        let designs = read_designs(&load(DESIGNS)?)?;
        let generations = read_generations(&load(GENERATIONS)?)?;
        Ok(Archive {
            dir: dir.to_path_buf(),
            meta,
            designs,
            generations,
            partial,
        })
    }

    /// The last row per `design_id`, in first-seen order.
    pub fn latest(&self) -> Vec<&DesignRecord> {
        let mut pos: BTreeMap<&str, usize> = BTreeMap::new();
        let mut out: Vec<&DesignRecord> = Vec::new();
        for d in &self.designs {
            match pos.get(d.design_id.as_str()) {
                Some(&i) => out[i] = d,
                None => {
                    pos.insert(&d.design_id, out.len());
                    out.push(d);
                }
            }
        }
        out
    }

    pub fn elites(&self) -> Vec<&DesignRecord> {
        self.latest()
            .into_iter()
            .filter(|d| d.status == DesignStatus::Elite)
            .collect()
    }

    pub fn design(&self, id: &str) -> Option<&DesignRecord> {
        self.designs.iter().rev().find(|d| d.design_id == id)
    }

    pub fn run_path(&self, d: &DesignRecord) -> Option<PathBuf> {
        d.run_path.as_ref().map(|p| self.dir.join(p))
    }

    /// Writes `archive.json` and both tables from scratch (atomic per file).
    pub fn write_dir(
        dir: &Path,
        meta: &ArchiveMeta,
        designs: &[DesignRecord],
        generations: &[GenerationRecord],
    ) -> Result<(), Diagnostic> {
        std::fs::create_dir_all(dir.join(RUNS_DIR))
            .map_err(|e| err(format!("cannot create {}: {e}", dir.display())))?;
        let json = serde_json::to_vec_pretty(meta).expect("archive meta serializes");
        atomic_write(&dir.join(ARCHIVE_JSON), &json)?;
        atomic_write(
            &dir.join(DESIGNS),
            &arrowx::to_ipc_stream(&designs_batch(designs)),
        )?;
        atomic_write(
            &dir.join(GENERATIONS),
            &arrowx::to_ipc_stream(&generations_batch(generations)),
        )
    }

    /// Appends rows as one IPC stream at the end of `designs.arrow`.
    pub fn append_designs(dir: &Path, rows: &[DesignRecord]) -> Result<(), Diagnostic> {
        append(
            &dir.join(DESIGNS),
            &arrowx::to_ipc_stream(&designs_batch(rows)),
        )
    }

    pub fn append_generations(dir: &Path, rows: &[GenerationRecord]) -> Result<(), Diagnostic> {
        append(
            &dir.join(GENERATIONS),
            &arrowx::to_ipc_stream(&generations_batch(rows)),
        )
    }
}

fn atomic_write(p: &Path, bytes: &[u8]) -> Result<(), Diagnostic> {
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, p))
        .map_err(|e| err(format!("cannot write {}: {e}", p.display())))
}

fn append(p: &Path, bytes: &[u8]) -> Result<(), Diagnostic> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| err(format!("cannot append to {}: {e}", p.display())))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn sample_meta() -> ArchiveMeta {
        ArchiveMeta::new(
            vec![
                DescriptorAxis {
                    name: "die_mm2_total".into(),
                    unit: "mm^2".into(),
                    log: false,
                    range: Some([10.0, 1000.0]),
                    bins: Some(8),
                    edges: None,
                },
                DescriptorAxis {
                    name: "power_w".into(),
                    unit: "W".into(),
                    log: true,
                    range: Some([5.0, 1000.0]),
                    bins: Some(6),
                    edges: None,
                },
            ],
            FitnessSpec {
                name: "matched_envelope".into(),
                direction: Direction::Maximize,
            },
        )
    }

    pub fn design(
        id: &str,
        g: u32,
        parents: &[&str],
        fitness: f64,
        status: DesignStatus,
    ) -> DesignRecord {
        let meta = sample_meta();
        let d = [100.0 + 90.0 * f64::from(g), 50.0 * (1.0 + fitness)];
        DesignRecord {
            design_id: id.into(),
            generation: g,
            parent_ids: parents.iter().map(|s| s.to_string()).collect(),
            mutation_summary: format!("mutation of {}", parents.first().unwrap_or(&"seed")),
            operator: if parents.len() > 1 {
                "crossover".into()
            } else {
                "mutate".into()
            },
            descriptor_values: d.to_vec(),
            cell: meta.axes.iter().zip(d).map(|(a, x)| a.bin_of(x)).collect(),
            fitness,
            fitness_components: [
                ("decode".to_string(), fitness * 0.9),
                ("prefill".to_string(), fitness * 1.1),
            ]
            .into(),
            status,
            run_path: Some(format!("runs/{id}.kiln")),
            thumbnail_path: None,
            design_hash: format!("hw1-{id}"),
            wall_time: 0.5,
            trust_level: Some("uncalibrated".into()),
            audit_status: None,
            heldout_score: None,
            calibration_set_hash: None,
            kiln_git_hash: Some("abc".into()),
            stage_reached: Some("S2".into()),
            extrapolated: Some(vec![]),
            fitness_low: Some(fitness * 0.9),
            fitness_high: Some(fitness * 1.05),
            interval_method: Some("corners".into()),
        }
    }

    #[test]
    fn write_append_read() {
        let dir = tempfile::tempdir().unwrap();
        let meta = sample_meta();
        let a = vec![
            design("ref", 0, &[], 1.0, DesignStatus::Elite),
            design("d1", 1, &["ref"], 1.2, DesignStatus::Elite),
        ];
        let g = vec![GenerationRecord {
            generation: 0,
            best: 1.0,
            median: 1.0,
            qd_score: 1.0,
            coverage: 0.02,
            evaluations: 1,
            invalid_count: 0,
            best_low: None,
            best_high: None,
        }];
        Archive::write_dir(dir.path(), &meta, &a, &g).unwrap();
        Archive::append_designs(
            dir.path(),
            &[
                design("d2", 2, &["d1", "ref"], 0.7, DesignStatus::Invalid),
                design("ref", 2, &[], 1.0, DesignStatus::Displaced),
            ],
        )
        .unwrap();
        let r = Archive::read_dir(dir.path()).unwrap();
        assert!(!r.partial);
        assert_eq!(r.meta, meta);
        assert_eq!(r.designs.len(), 4);
        assert_eq!(r.designs[1], a[1]);
        assert_eq!(r.latest().len(), 3);
        assert_eq!(
            r.elites()
                .iter()
                .map(|d| d.design_id.as_str())
                .collect::<Vec<_>>(),
            vec!["d1"]
        );
        assert_eq!(r.generations, g);
        let mut bytes = std::fs::read(dir.path().join(DESIGNS)).unwrap();
        bytes.truncate(bytes.len() - 30);
        std::fs::write(dir.path().join(DESIGNS), bytes).unwrap();
        let r = Archive::read_dir(dir.path()).unwrap();
        assert!(r.partial);
        assert_eq!(r.designs.len(), 2);
    }

    #[test]
    fn bins() {
        let a = &sample_meta().axes[1];
        assert_eq!(a.bin_edges().len(), 7);
        assert_eq!(a.bin_of(4.0), 0);
        assert_eq!(a.bin_of(5000.0), 5);
        assert!((a.bin_edges()[6] - 1000.0).abs() < 1e-9);
    }
}
