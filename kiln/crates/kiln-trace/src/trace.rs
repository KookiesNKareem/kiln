//! In-memory `.kiln` trace (05 §3.4): typed rows per table, Arrow encode/decode, and the enum string tables
//! that live in the manifest. Row index = the table's dense `idx`.

use std::collections::BTreeMap;

use arrow_array::RecordBatch;
use kiln_ir::common::Diagnostic;

use crate::arrowx::{self, Cols, Table};
use crate::container::{Manifest, TRACE_SCHEMA_VERSION};

pub const NONE_U32: u32 = u32::MAX;

/// Default enum string tables (05 §3.4); codes index these lists. Dynamic enums (`ops.kind`, `ops.phase`)
/// are filled per trace.
pub fn default_enums() -> BTreeMap<String, Vec<String>> {
    let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    [
        (
            "resources.kind",
            v(&[
                "system",
                "host",
                "board",
                "package",
                "die",
                "cluster",
                "unit_matrix",
                "unit_vector",
                "unit_scalar",
                "unit_special",
                "unit_cim",
                "nmp_unit",
                "memory",
                "mem_stack",
                "network",
                "router",
                "channel",
                "switch",
                "port",
                "block",
                "sequencer",
                "dma",
                "bank",
            ]),
        ),
        (
            "resources.class",
            v(&["compute", "memory", "interconnect", "io", "container"]),
        ),
        (
            "spans.kind",
            v(&[
                "compute",
                "load",
                "store",
                "transfer_send",
                "transfer_recv",
                "collective_step",
                "nmp_compute",
                "nmp_command",
                "mode_switch",
                "stall_contention",
                "stall_dependency",
                "launch_overhead",
                "idle_powergated",
                "estimated",
            ]),
        ),
        (
            "limiters.binding",
            v(&[
                "compute",
                "dram",
                "link",
                "port",
                "dependency",
                "overhead",
                "nmp",
                "contention",
                "pipeline_bubble",
            ]),
        ),
        (
            "groups.kind",
            v(&["single", "fused", "pipelined", "layer_by_layer"]),
        ),
        ("ops.target", v(&["host", "nmp"])),
        (
            "aggregates_op_resource.energy_component",
            v(&[
                "compute",
                "sram_read",
                "sram_write",
                "dram",
                "noc",
                "link",
                "leakage",
                "other",
            ]),
        ),
        (
            "bottleneck.section",
            v(&["time_by_binding", "top_resource", "slack"]),
        ),
        ("ceilings.kind", v(&["compute", "memory", "link"])),
        ("diagnostics.severity", v(&["error", "warning", "info"])),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// Code of `name` in enum `table` of `enums`.
pub fn code(enums: &BTreeMap<String, Vec<String>>, table: &str, name: &str) -> Option<u32> {
    enums
        .get(table)?
        .iter()
        .position(|s| s == name)
        .map(|i| i as u32)
}

pub mod span_flags {
    pub const ON_CRITICAL_PATH: u8 = 1;
    pub const ESTIMATED: u8 = 2;
    pub const CLIPPED: u8 = 4;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceRow {
    pub path: String,
    pub kind: u16,
    pub class: u8,
    pub parent: Option<u32>,
    pub chip: u32,
    pub array_id: Option<u32>,
    pub array_pos: Option<u32>,
    pub mem_level: Option<u8>,
    pub capacity_b: Option<f64>,
    pub peak_bw_bps: Option<f64>,
    pub peak_flops: Vec<(u8, f64)>,
    pub lanes: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpRow {
    pub path: String,
    /// Path with iteration and kernel suffixes stripped (`layers.gate_up.i1.k0` -> `layers.gate_up`); the
    /// "aggregate across layers" key of the roofline and bottleneck views.
    pub family: String,
    pub kind: u16,
    pub phase: u8,
    pub layer: Option<i32>,
    pub flops: f64,
    pub precision: u8,
    pub bytes_by_level: Vec<f64>,
    pub link_bytes: f64,
    pub t_start: i64,
    pub t_end: i64,
    pub chips: Vec<u32>,
    pub mapping: u32,
    pub macs_useful: u64,
    pub macs_issued: u64,
    pub target: u8,
    pub host_vs_nmp_s: Option<[f64; 2]>,
    pub group: u32,
    pub energy_j: f64,
    /// Op time at the slow and fast parameter corners (03 §9.1) when the run evaluated them.
    pub time_low_s: Option<f64>,
    pub time_high_s: Option<f64>,
}

impl OpRow {
    pub fn time_s(&self, tick_s: f64) -> f64 {
        (self.t_end - self.t_start) as f64 * tick_s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpanRow {
    pub resource: u32,
    pub lane: u16,
    pub kind: u8,
    pub flags: u8,
    pub op: u32,
    pub task: u32,
    pub slice: u32,
    pub t_start: i64,
    pub dur: i64,
    pub bytes: f64,
    pub energy_j: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggResourceRow {
    pub resource: u32,
    pub phase: u8,
    pub busy_s: f64,
    pub stall_s: f64,
    pub bytes: f64,
    pub flops: f64,
    pub energy_dyn_j: f64,
    pub energy_leak_j: f64,
    pub avg_power_w: f64,
    pub peak_power_w: f64,
    pub power_density_w_mm2: Option<f64>,
    pub peak_temp_k: Option<f64>,
    pub utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggOpResourceRow {
    pub op: u32,
    pub resource: u32,
    pub time_s: f64,
    pub energy_component: u8,
    pub energy_j: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LimiterRow {
    pub op: u32,
    pub group: u32,
    pub binding: u8,
    pub resource: Option<u32>,
    pub time_s: f64,
    pub attained_frac: f64,
    pub rank: u8,
    pub share: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BottleneckRow {
    pub phase: u8,
    pub section: u8,
    pub binding: Option<u8>,
    pub resource: Option<u32>,
    pub time_s: Option<f64>,
    pub utilization: Option<f64>,
    pub shadow_price: Option<f64>,
    pub slack: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupRow {
    pub group: u32,
    pub phase: u8,
    pub ops: Vec<u32>,
    pub kind: u8,
    pub binding: u8,
    pub t_start: i64,
    pub t_end: i64,
    pub bubble_s: f64,
    pub exposed_overhead_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectiveRow {
    pub collective: u32,
    pub name: String,
    pub phase: u8,
    pub op: u32,
    pub algorithm: String,
    pub group_chips: Vec<u32>,
    pub steps: u16,
    pub bytes: f64,
    pub t_start: i64,
    pub t_end: i64,
    pub link_bytes_by_tier: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloorplanRow {
    pub resource: u32,
    pub die: u32,
    pub layer: u8,
    pub x_um: f64,
    pub y_um: f64,
    pub w_um: f64,
    pub h_um: f64,
    /// Rectilinear outline as flattened `x0, y0, x1, y1, ...`; `None` = the rect.
    pub poly: Option<Vec<f64>>,
    pub rotation: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticRow {
    pub code: String,
    pub severity: u8,
    pub message: String,
    pub path: Option<String>,
    pub hint: Option<String>,
}

/// Roofline ceilings per chip (05 §6.4): peak FLOP/s per precision mode, bandwidth per memory level and
/// inter-chip link bandwidth. `low`/`high` are set when the ceiling depends on ranged parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct CeilingRow {
    pub chip: u32,
    pub kind: u8,
    pub name: String,
    pub level: Option<u8>,
    pub value: f64,
    pub low: Option<f64>,
    pub high: Option<f64>,
}

/// One simulated phase of the run. Phases are laid out back to back on the trace time axis from `t_offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseRow {
    pub phase: u8,
    pub id: String,
    pub scope: String,
    pub t_offset: i64,
    /// End of the scheduled ops (Tier A: the repeat window, not the extrapolated step).
    pub t_end: i64,
    pub makespan_s: f64,
    pub makespan_low_s: Option<f64>,
    pub makespan_high_s: Option<f64>,
    pub t_a0_s: f64,
    pub t_a2_s: f64,
    pub energy_j: f64,
    pub avg_power_w: f64,
    pub tokens_per_s: Option<f64>,
    pub summary: String,
}

/// A whole `.kiln` trace in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub manifest: Manifest,
    pub resources: Vec<ResourceRow>,
    pub ops: Vec<OpRow>,
    pub spans: Vec<SpanRow>,
    pub aggregates_resource: Vec<AggResourceRow>,
    pub aggregates_op_resource: Vec<AggOpResourceRow>,
    pub limiters: Vec<LimiterRow>,
    pub bottleneck: Vec<BottleneckRow>,
    pub groups: Vec<GroupRow>,
    pub collectives: Vec<CollectiveRow>,
    pub floorplan: Vec<FloorplanRow>,
    pub diagnostics: Vec<DiagnosticRow>,
    pub ceilings: Vec<CeilingRow>,
    pub phases: Vec<PhaseRow>,
    /// `(name, canonical JSON)`: power, energy, invariants, calibration per phase, and the result summary.
    pub run_scalars: Vec<(String, String)>,
}

impl Trace {
    pub fn empty(manifest: Manifest) -> Self {
        Trace {
            manifest,
            resources: vec![],
            ops: vec![],
            spans: vec![],
            aggregates_resource: vec![],
            aggregates_op_resource: vec![],
            limiters: vec![],
            bottleneck: vec![],
            groups: vec![],
            collectives: vec![],
            floorplan: vec![],
            diagnostics: vec![],
            ceilings: vec![],
            phases: vec![],
            run_scalars: vec![],
        }
    }

    pub fn tick_s(&self) -> f64 {
        self.manifest.tick_s
    }

    pub fn enum_name(&self, table: &str, code: u32) -> &str {
        self.manifest
            .enums
            .get(table)
            .and_then(|v| v.get(code as usize))
            .map_or("?", String::as_str)
    }

    pub fn resource_kind(&self, r: &ResourceRow) -> &str {
        self.enum_name("resources.kind", u32::from(r.kind))
    }

    pub fn op_kind(&self, o: &OpRow) -> &str {
        self.enum_name("ops.kind", u32::from(o.kind))
    }

    pub fn phase_name(&self, p: u8) -> &str {
        self.enum_name("ops.phase", u32::from(p))
    }

    pub fn binding_name(&self, b: u8) -> &str {
        self.enum_name("limiters.binding", u32::from(b))
    }

    pub fn resource_by_path(&self, path: &str) -> Option<u32> {
        self.resources
            .binary_search_by(|r| r.path.as_str().cmp(path))
            .ok()
            .map(|i| i as u32)
    }

    pub fn scalar(&self, name: &str) -> Option<serde_json::Value> {
        self.run_scalars
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, j)| serde_json::from_str(j).ok())
    }

    /// Children of every resource (CSR-free: a vector per row).
    pub fn children(&self) -> Vec<Vec<u32>> {
        let mut c = vec![Vec::new(); self.resources.len()];
        for (i, r) in self.resources.iter().enumerate() {
            if let Some(p) = r.parent {
                c[p as usize].push(i as u32);
            }
        }
        c
    }

    /// Rank-0 limiter of every op.
    pub fn binding_of_ops(&self) -> Vec<Option<&LimiterRow>> {
        let mut out = vec![None; self.ops.len()];
        for l in &self.limiters {
            if l.rank == 0 && (l.op as usize) < out.len() {
                out[l.op as usize] = Some(l);
            }
        }
        out
    }

    /// Table name -> encoded batch, in container member order. Empty tables are included.
    pub fn batches(&self) -> Vec<(&'static str, RecordBatch)> {
        vec![
            ("resources", resources_batch(&self.resources)),
            ("ops", ops_batch(&self.ops)),
            ("phases", phases_batch(&self.phases)),
            (
                "aggregates_resource",
                agg_resource_batch(&self.aggregates_resource),
            ),
            (
                "aggregates_op_resource",
                agg_op_resource_batch(&self.aggregates_op_resource),
            ),
            ("limiters", limiters_batch(&self.limiters)),
            ("bottleneck", bottleneck_batch(&self.bottleneck)),
            ("groups", groups_batch(&self.groups)),
            ("collectives", collectives_batch(&self.collectives)),
            ("floorplan", floorplan_batch(&self.floorplan)),
            ("ceilings", ceilings_batch(&self.ceilings)),
            ("diagnostics", diagnostics_batch(&self.diagnostics)),
            ("run_scalars", run_scalars_batch(&self.run_scalars)),
            ("spans", spans_batch(&self.spans)),
        ]
    }

    /// Fills the table named `name` from its batches; unknown tables are ignored (05 §3.10).
    pub fn set_table(&mut self, name: &str, b: &[RecordBatch]) -> Result<bool, Diagnostic> {
        match name {
            "resources" => self.resources = read_resources(b)?,
            "ops" => self.ops = read_ops(b)?,
            "phases" => self.phases = read_phases(b)?,
            "aggregates_resource" => self.aggregates_resource = read_agg_resource(b)?,
            "aggregates_op_resource" => self.aggregates_op_resource = read_agg_op_resource(b)?,
            "limiters" => self.limiters = read_limiters(b)?,
            "bottleneck" => self.bottleneck = read_bottleneck(b)?,
            "groups" => self.groups = read_groups(b)?,
            "collectives" => self.collectives = read_collectives(b)?,
            "floorplan" => self.floorplan = read_floorplan(b)?,
            "ceilings" => self.ceilings = read_ceilings(b)?,
            "diagnostics" => self.diagnostics = read_diagnostics(b)?,
            "run_scalars" => self.run_scalars = read_run_scalars(b)?,
            "spans" => self.spans = read_spans(b)?,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Schema of an implemented table (derived from its encoder, so it always matches what is written).
pub fn implemented_schema(name: &str) -> Option<arrow_schema::Schema> {
    let t = Trace::empty(Manifest::new(
        crate::TraceLevel::Summary,
        crate::Provenance::unknown(crate::Tier::A),
    ));
    t.batches()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, b)| b.schema().as_ref().clone())
}

fn batch(name: &str, c: Cols) -> RecordBatch {
    c.batch(name, TRACE_SCHEMA_VERSION)
}

macro_rules! col {
    ($rows:expr, |$r:ident| $e:expr) => {
        $rows.iter().map(|$r| $e)
    };
}

fn resources_batch(rows: &[ResourceRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("idx", false, u32s((0..rows.len() as u32).map(Some)))
        .push("path", false, dict_utf8(col!(rows, |r| r.path.as_str())))
        .push("kind", false, u16s(col!(rows, |r| Some(r.kind))))
        .push("class", false, u8s(col!(rows, |r| Some(r.class))))
        .push("parent", true, u32s(col!(rows, |r| r.parent)))
        .push("chip", false, u32s(col!(rows, |r| Some(r.chip))))
        .push("array_id", true, u32s(col!(rows, |r| r.array_id)))
        .push("array_pos", true, u32s(col!(rows, |r| r.array_pos)))
        .push("mem_level", true, u8s(col!(rows, |r| r.mem_level)))
        .push("capacity_b", true, f64s(col!(rows, |r| r.capacity_b)))
        .push("peak_bw_bps", true, f64s(col!(rows, |r| r.peak_bw_bps)))
        .push(
            "peak_flops",
            false,
            map_u8_f64(col!(rows, |r| r.peak_flops.iter())),
        )
        .push("lanes", false, u16s(col!(rows, |r| Some(r.lanes))));
    batch("resources", c)
}

fn read_resources(b: &[RecordBatch]) -> Result<Vec<ResourceRow>, Diagnostic> {
    let t = Table::new("resources", b);
    let path = t.str("path")?;
    let kind = t.int::<u16>("kind")?;
    let class = t.int::<u8>("class")?;
    let parent = t.opt_u32("parent")?;
    let chip = t.u32("chip")?;
    let aid = t.opt_u32("array_id")?;
    let apos = t.opt_u32("array_pos")?;
    let lvl = t.opt_int::<u8>("mem_level")?;
    let cap = t.opt_f64("capacity_b")?;
    let bw = t.opt_f64("peak_bw_bps")?;
    let pf = t.map_u8_f64("peak_flops")?;
    let lanes = t.opt_int::<u16>("lanes")?;
    Ok((0..t.rows())
        .map(|i| ResourceRow {
            path: path[i].clone(),
            kind: kind[i],
            class: class[i],
            parent: parent[i],
            chip: chip[i],
            array_id: aid[i],
            array_pos: apos[i],
            mem_level: lvl[i],
            capacity_b: cap[i],
            peak_bw_bps: bw[i],
            peak_flops: pf[i].clone(),
            lanes: lanes[i].unwrap_or(1),
        })
        .collect())
}

fn ops_batch(rows: &[OpRow]) -> RecordBatch {
    use arrowx::*;
    let hvn: Vec<Option<Vec<f64>>> = rows
        .iter()
        .map(|r| r.host_vs_nmp_s.map(|x| x.to_vec()))
        .collect();
    let mut c = Cols::new();
    c.push("idx", false, u32s((0..rows.len() as u32).map(Some)))
        .push("path", false, utf8s(col!(rows, |r| r.path.as_str())))
        .push(
            "family",
            false,
            dict_utf8(col!(rows, |r| r.family.as_str())),
        )
        .push("kind", false, u16s(col!(rows, |r| Some(r.kind))))
        .push("phase", false, u8s(col!(rows, |r| Some(r.phase))))
        .push("layer", true, i32s(col!(rows, |r| r.layer)))
        .push("flops", false, f64v(col!(rows, |r| r.flops)))
        .push("precision", false, u8s(col!(rows, |r| Some(r.precision))))
        .push(
            "bytes_by_level",
            false,
            list_f64(col!(rows, |r| Some(&r.bytes_by_level[..]))),
        )
        .push("link_bytes", false, f64v(col!(rows, |r| r.link_bytes)))
        .push("t_start", false, i64s(col!(rows, |r| Some(r.t_start))))
        .push("t_end", false, i64s(col!(rows, |r| Some(r.t_end))))
        .push("chips", false, list_u32(col!(rows, |r| Some(&r.chips[..]))))
        .push("mapping", false, u32s(col!(rows, |r| Some(r.mapping))))
        .push(
            "macs_useful",
            false,
            u64s(col!(rows, |r| Some(r.macs_useful))),
        )
        .push(
            "macs_issued",
            false,
            u64s(col!(rows, |r| Some(r.macs_issued))),
        )
        .push("target", false, u8s(col!(rows, |r| Some(r.target))))
        .push(
            "host_vs_nmp_s",
            true,
            list_f64(hvn.iter().map(|x| x.as_deref())),
        )
        .push("group", false, u32s(col!(rows, |r| Some(r.group))))
        .push("energy_j", false, f64v(col!(rows, |r| r.energy_j)))
        .push("time_low_s", true, f64s(col!(rows, |r| r.time_low_s)))
        .push("time_high_s", true, f64s(col!(rows, |r| r.time_high_s)));
    batch("ops", c)
}

fn read_ops(b: &[RecordBatch]) -> Result<Vec<OpRow>, Diagnostic> {
    let t = Table::new("ops", b);
    let path = t.str("path")?;
    let family = t.opt_str("family")?;
    let kind = t.int::<u16>("kind")?;
    let phase = t.int::<u8>("phase")?;
    let layer = t.opt_int::<i32>("layer")?;
    let flops = t.f64("flops")?;
    let prec = t.opt_int::<u8>("precision")?;
    let bbl = t.list_f64("bytes_by_level")?;
    let lb = t.opt_f64("link_bytes")?;
    let ts = t.i64("t_start")?;
    let te = t.i64("t_end")?;
    let chips = t.list_u32("chips")?;
    let mapping = t.opt_u32("mapping")?;
    let mu = t.opt_u64("macs_useful")?;
    let mi = t.opt_u64("macs_issued")?;
    let target = t.opt_int::<u8>("target")?;
    let hvn = t.list_f64("host_vs_nmp_s")?;
    let group = t.opt_u32("group")?;
    let e = t.opt_f64("energy_j")?;
    let lo = t.opt_f64("time_low_s")?;
    let hi = t.opt_f64("time_high_s")?;
    Ok((0..t.rows())
        .map(|i| OpRow {
            family: family[i].clone().unwrap_or_else(|| op_family(&path[i])),
            path: path[i].clone(),
            kind: kind[i],
            phase: phase[i],
            layer: layer[i],
            flops: flops[i],
            precision: prec[i].unwrap_or(0),
            bytes_by_level: bbl[i].clone().unwrap_or_default(),
            link_bytes: lb[i].unwrap_or(0.0),
            t_start: ts[i],
            t_end: te[i],
            chips: chips[i].clone().unwrap_or_default(),
            mapping: mapping[i].unwrap_or(NONE_U32),
            macs_useful: mu[i].unwrap_or(0),
            macs_issued: mi[i].unwrap_or(0),
            target: target[i].unwrap_or(0),
            host_vs_nmp_s: hvn[i]
                .as_ref()
                .and_then(|v| (v.len() == 2).then(|| [v[0], v[1]])),
            group: group[i].unwrap_or(NONE_U32),
            energy_j: e[i].unwrap_or(0.0),
            time_low_s: lo[i],
            time_high_s: hi[i],
        })
        .collect())
}

/// `layers.gate_up.i1.k0` -> `layers.gate_up`: strips trailing `i<N>` (iteration) and `k<N>` (kernel)
/// segments.
pub fn op_family(path: &str) -> String {
    let mut parts: Vec<&str> = path.split('.').collect();
    while parts.len() > 1 {
        let last = parts[parts.len() - 1];
        let suffix = last.len() > 1
            && (last.starts_with('i') || last.starts_with('k'))
            && last[1..].bytes().all(|b| b.is_ascii_digit());
        if !suffix {
            break;
        }
        parts.pop();
    }
    parts.join(".")
}

fn phases_batch(rows: &[PhaseRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("phase", false, u8s(col!(rows, |r| Some(r.phase))))
        .push("id", false, utf8s(col!(rows, |r| r.id.as_str())))
        .push("scope", false, utf8s(col!(rows, |r| r.scope.as_str())))
        .push("t_offset", false, i64s(col!(rows, |r| Some(r.t_offset))))
        .push("t_end", false, i64s(col!(rows, |r| Some(r.t_end))))
        .push("makespan_s", false, f64v(col!(rows, |r| r.makespan_s)))
        .push(
            "makespan_low_s",
            true,
            f64s(col!(rows, |r| r.makespan_low_s)),
        )
        .push(
            "makespan_high_s",
            true,
            f64s(col!(rows, |r| r.makespan_high_s)),
        )
        .push("t_a0_s", false, f64v(col!(rows, |r| r.t_a0_s)))
        .push("t_a2_s", false, f64v(col!(rows, |r| r.t_a2_s)))
        .push("energy_j", false, f64v(col!(rows, |r| r.energy_j)))
        .push("avg_power_w", false, f64v(col!(rows, |r| r.avg_power_w)))
        .push("tokens_per_s", true, f64s(col!(rows, |r| r.tokens_per_s)))
        .push("summary", false, utf8s(col!(rows, |r| r.summary.as_str())));
    batch("phases", c)
}

fn read_phases(b: &[RecordBatch]) -> Result<Vec<PhaseRow>, Diagnostic> {
    let t = Table::new("phases", b);
    let phase = t.int::<u8>("phase")?;
    let id = t.str("id")?;
    let scope = t.opt_str("scope")?;
    let off = t.i64("t_offset")?;
    let end = t.i64("t_end")?;
    let ms = t.f64("makespan_s")?;
    let lo = t.opt_f64("makespan_low_s")?;
    let hi = t.opt_f64("makespan_high_s")?;
    let a0 = t.opt_f64("t_a0_s")?;
    let a2 = t.opt_f64("t_a2_s")?;
    let e = t.opt_f64("energy_j")?;
    let p = t.opt_f64("avg_power_w")?;
    let tok = t.opt_f64("tokens_per_s")?;
    let sum = t.opt_str("summary")?;
    Ok((0..t.rows())
        .map(|i| PhaseRow {
            phase: phase[i],
            id: id[i].clone(),
            scope: scope[i].clone().unwrap_or_default(),
            t_offset: off[i],
            t_end: end[i],
            makespan_s: ms[i],
            makespan_low_s: lo[i],
            makespan_high_s: hi[i],
            t_a0_s: a0[i].unwrap_or(0.0),
            t_a2_s: a2[i].unwrap_or(0.0),
            energy_j: e[i].unwrap_or(0.0),
            avg_power_w: p[i].unwrap_or(0.0),
            tokens_per_s: tok[i],
            summary: sum[i].clone().unwrap_or_default(),
        })
        .collect())
}

fn agg_resource_batch(rows: &[AggResourceRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("resource", false, u32s(col!(rows, |r| Some(r.resource))))
        .push("phase", false, u8s(col!(rows, |r| Some(r.phase))))
        .push("busy_s", false, f64v(col!(rows, |r| r.busy_s)))
        .push("stall_s", false, f64v(col!(rows, |r| r.stall_s)))
        .push("bytes", false, f64v(col!(rows, |r| r.bytes)))
        .push("flops", false, f64v(col!(rows, |r| r.flops)))
        .push("energy_dyn_j", false, f64v(col!(rows, |r| r.energy_dyn_j)))
        .push(
            "energy_leak_j",
            false,
            f64v(col!(rows, |r| r.energy_leak_j)),
        )
        .push("avg_power_w", false, f64v(col!(rows, |r| r.avg_power_w)))
        .push("peak_power_w", false, f64v(col!(rows, |r| r.peak_power_w)))
        .push(
            "power_density_w_mm2",
            true,
            f64s(col!(rows, |r| r.power_density_w_mm2)),
        )
        .push("peak_temp_k", true, f64s(col!(rows, |r| r.peak_temp_k)))
        .push("utilization", false, f64v(col!(rows, |r| r.utilization)));
    batch("aggregates_resource", c)
}

fn read_agg_resource(b: &[RecordBatch]) -> Result<Vec<AggResourceRow>, Diagnostic> {
    let t = Table::new("aggregates_resource", b);
    let res = t.u32("resource")?;
    let phase = t.opt_int::<u8>("phase")?;
    let busy = t.f64("busy_s")?;
    let stall = t.opt_f64("stall_s")?;
    let bytes = t.opt_f64("bytes")?;
    let flops = t.opt_f64("flops")?;
    let ed = t.opt_f64("energy_dyn_j")?;
    let el = t.opt_f64("energy_leak_j")?;
    let ap = t.opt_f64("avg_power_w")?;
    let pp = t.opt_f64("peak_power_w")?;
    let pd = t.opt_f64("power_density_w_mm2")?;
    let pt = t.opt_f64("peak_temp_k")?;
    let u = t.opt_f64("utilization")?;
    Ok((0..t.rows())
        .map(|i| AggResourceRow {
            resource: res[i],
            phase: phase[i].unwrap_or(0),
            busy_s: busy[i],
            stall_s: stall[i].unwrap_or(0.0),
            bytes: bytes[i].unwrap_or(0.0),
            flops: flops[i].unwrap_or(0.0),
            energy_dyn_j: ed[i].unwrap_or(0.0),
            energy_leak_j: el[i].unwrap_or(0.0),
            avg_power_w: ap[i].unwrap_or(0.0),
            peak_power_w: pp[i].unwrap_or(0.0),
            power_density_w_mm2: pd[i],
            peak_temp_k: pt[i],
            utilization: u[i].unwrap_or(0.0),
        })
        .collect())
}

fn agg_op_resource_batch(rows: &[AggOpResourceRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("op", false, u32s(col!(rows, |r| Some(r.op))))
        .push("resource", false, u32s(col!(rows, |r| Some(r.resource))))
        .push("time_s", false, f64v(col!(rows, |r| r.time_s)))
        .push(
            "energy_component",
            false,
            u8s(col!(rows, |r| Some(r.energy_component))),
        )
        .push("energy_j", false, f64v(col!(rows, |r| r.energy_j)));
    batch("aggregates_op_resource", c)
}

fn read_agg_op_resource(b: &[RecordBatch]) -> Result<Vec<AggOpResourceRow>, Diagnostic> {
    let t = Table::new("aggregates_op_resource", b);
    let op = t.u32("op")?;
    let res = t.u32("resource")?;
    let time = t.f64("time_s")?;
    let comp = t.int::<u8>("energy_component")?;
    let e = t.f64("energy_j")?;
    Ok((0..t.rows())
        .map(|i| AggOpResourceRow {
            op: op[i],
            resource: res[i],
            time_s: time[i],
            energy_component: comp[i],
            energy_j: e[i],
        })
        .collect())
}

fn limiters_batch(rows: &[LimiterRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("op", false, u32s(col!(rows, |r| Some(r.op))))
        .push("group", false, u32s(col!(rows, |r| Some(r.group))))
        .push("binding", false, u8s(col!(rows, |r| Some(r.binding))))
        .push("resource", true, u32s(col!(rows, |r| r.resource)))
        .push("time_s", false, f64v(col!(rows, |r| r.time_s)))
        .push(
            "attained_frac",
            false,
            f64v(col!(rows, |r| r.attained_frac)),
        )
        .push("rank", false, u8s(col!(rows, |r| Some(r.rank))))
        .push("share", false, f64v(col!(rows, |r| r.share)));
    batch("limiters", c)
}

fn read_limiters(b: &[RecordBatch]) -> Result<Vec<LimiterRow>, Diagnostic> {
    let t = Table::new("limiters", b);
    let op = t.u32("op")?;
    let group = t.opt_u32("group")?;
    let binding = t.int::<u8>("binding")?;
    let res = t.opt_u32("resource")?;
    let time = t.f64("time_s")?;
    let att = t.opt_f64("attained_frac")?;
    let rank = t.int::<u8>("rank")?;
    let share = t.opt_f64("share")?;
    Ok((0..t.rows())
        .map(|i| LimiterRow {
            op: op[i],
            group: group[i].unwrap_or(NONE_U32),
            binding: binding[i],
            resource: res[i],
            time_s: time[i],
            attained_frac: att[i].unwrap_or(f64::NAN),
            rank: rank[i],
            share: share[i].unwrap_or(0.0),
        })
        .collect())
}

fn bottleneck_batch(rows: &[BottleneckRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("phase", false, u8s(col!(rows, |r| Some(r.phase))))
        .push("section", false, u8s(col!(rows, |r| Some(r.section))))
        .push("binding", true, u8s(col!(rows, |r| r.binding)))
        .push("resource", true, u32s(col!(rows, |r| r.resource)))
        .push("time_s", true, f64s(col!(rows, |r| r.time_s)))
        .push("utilization", true, f64s(col!(rows, |r| r.utilization)))
        .push("shadow_price", true, f64s(col!(rows, |r| r.shadow_price)))
        .push("slack", true, f64s(col!(rows, |r| r.slack)));
    batch("bottleneck", c)
}

fn read_bottleneck(b: &[RecordBatch]) -> Result<Vec<BottleneckRow>, Diagnostic> {
    let t = Table::new("bottleneck", b);
    let phase = t.opt_int::<u8>("phase")?;
    let section = t.int::<u8>("section")?;
    let binding = t.opt_int::<u8>("binding")?;
    let res = t.opt_u32("resource")?;
    let time = t.opt_f64("time_s")?;
    let u = t.opt_f64("utilization")?;
    let sp = t.opt_f64("shadow_price")?;
    let sl = t.opt_f64("slack")?;
    Ok((0..t.rows())
        .map(|i| BottleneckRow {
            phase: phase[i].unwrap_or(0),
            section: section[i],
            binding: binding[i],
            resource: res[i],
            time_s: time[i],
            utilization: u[i],
            shadow_price: sp[i],
            slack: sl[i],
        })
        .collect())
}

fn groups_batch(rows: &[GroupRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("group", false, u32s(col!(rows, |r| Some(r.group))))
        .push("phase", false, u8s(col!(rows, |r| Some(r.phase))))
        .push("ops", false, list_u32(col!(rows, |r| Some(&r.ops[..]))))
        .push("kind", false, u8s(col!(rows, |r| Some(r.kind))))
        .push("binding", false, u8s(col!(rows, |r| Some(r.binding))))
        .push("t_start", false, i64s(col!(rows, |r| Some(r.t_start))))
        .push("t_end", false, i64s(col!(rows, |r| Some(r.t_end))))
        .push("bubble_s", false, f64v(col!(rows, |r| r.bubble_s)))
        .push(
            "exposed_overhead_s",
            false,
            f64v(col!(rows, |r| r.exposed_overhead_s)),
        );
    batch("groups", c)
}

fn read_groups(b: &[RecordBatch]) -> Result<Vec<GroupRow>, Diagnostic> {
    let t = Table::new("groups", b);
    let g = t.u32("group")?;
    let phase = t.opt_int::<u8>("phase")?;
    let ops = t.list_u32("ops")?;
    let kind = t.int::<u8>("kind")?;
    let binding = t.opt_int::<u8>("binding")?;
    let ts = t.i64("t_start")?;
    let te = t.i64("t_end")?;
    let bub = t.opt_f64("bubble_s")?;
    let eo = t.opt_f64("exposed_overhead_s")?;
    Ok((0..t.rows())
        .map(|i| GroupRow {
            group: g[i],
            phase: phase[i].unwrap_or(0),
            ops: ops[i].clone().unwrap_or_default(),
            kind: kind[i],
            binding: binding[i].unwrap_or(0),
            t_start: ts[i],
            t_end: te[i],
            bubble_s: bub[i].unwrap_or(0.0),
            exposed_overhead_s: eo[i].unwrap_or(0.0),
        })
        .collect())
}

fn collectives_batch(rows: &[CollectiveRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push(
        "collective",
        false,
        u32s(col!(rows, |r| Some(r.collective))),
    )
    .push("name", false, utf8s(col!(rows, |r| r.name.as_str())))
    .push("phase", false, u8s(col!(rows, |r| Some(r.phase))))
    .push("op", false, u32s(col!(rows, |r| Some(r.op))))
    .push(
        "algorithm",
        false,
        utf8s(col!(rows, |r| r.algorithm.as_str())),
    )
    .push(
        "group_chips",
        false,
        list_u32(col!(rows, |r| Some(&r.group_chips[..]))),
    )
    .push("steps", false, u16s(col!(rows, |r| Some(r.steps))))
    .push("bytes", false, f64v(col!(rows, |r| r.bytes)))
    .push("t_start", false, i64s(col!(rows, |r| Some(r.t_start))))
    .push("t_end", false, i64s(col!(rows, |r| Some(r.t_end))))
    .push(
        "link_bytes_by_tier",
        false,
        list_f64(col!(rows, |r| Some(&r.link_bytes_by_tier[..]))),
    );
    batch("collectives", c)
}

fn read_collectives(b: &[RecordBatch]) -> Result<Vec<CollectiveRow>, Diagnostic> {
    let t = Table::new("collectives", b);
    let id = t.u32("collective")?;
    let name = t.opt_str("name")?;
    let phase = t.opt_int::<u8>("phase")?;
    let op = t.u32("op")?;
    let alg = t.opt_str("algorithm")?;
    let chips = t.list_u32("group_chips")?;
    let steps = t.opt_int::<u16>("steps")?;
    let bytes = t.opt_f64("bytes")?;
    let ts = t.i64("t_start")?;
    let te = t.i64("t_end")?;
    let lb = t.list_f64("link_bytes_by_tier")?;
    Ok((0..t.rows())
        .map(|i| CollectiveRow {
            collective: id[i],
            name: name[i].clone().unwrap_or_default(),
            phase: phase[i].unwrap_or(0),
            op: op[i],
            algorithm: alg[i].clone().unwrap_or_default(),
            group_chips: chips[i].clone().unwrap_or_default(),
            steps: steps[i].unwrap_or(0),
            bytes: bytes[i].unwrap_or(0.0),
            t_start: ts[i],
            t_end: te[i],
            link_bytes_by_tier: lb[i].clone().unwrap_or_default(),
        })
        .collect())
}

fn floorplan_batch(rows: &[FloorplanRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("resource", false, u32s(col!(rows, |r| Some(r.resource))))
        .push("die", false, u32s(col!(rows, |r| Some(r.die))))
        .push("layer", false, u8s(col!(rows, |r| Some(r.layer))))
        .push("x_um", false, f64v(col!(rows, |r| r.x_um)))
        .push("y_um", false, f64v(col!(rows, |r| r.y_um)))
        .push("w_um", false, f64v(col!(rows, |r| r.w_um)))
        .push("h_um", false, f64v(col!(rows, |r| r.h_um)))
        .push("poly", true, list_f64(col!(rows, |r| r.poly.as_deref())))
        .push("rotation", false, u8s(col!(rows, |r| Some(r.rotation))));
    batch("floorplan", c)
}

fn read_floorplan(b: &[RecordBatch]) -> Result<Vec<FloorplanRow>, Diagnostic> {
    let t = Table::new("floorplan", b);
    let res = t.u32("resource")?;
    let die = t.opt_u32("die")?;
    let layer = t.opt_int::<u8>("layer")?;
    let x = t.f64("x_um")?;
    let y = t.f64("y_um")?;
    let w = t.f64("w_um")?;
    let h = t.f64("h_um")?;
    let poly = t.list_f64("poly")?;
    let rot = t.opt_int::<u8>("rotation")?;
    Ok((0..t.rows())
        .map(|i| FloorplanRow {
            resource: res[i],
            die: die[i].unwrap_or(NONE_U32),
            layer: layer[i].unwrap_or(0),
            x_um: x[i],
            y_um: y[i],
            w_um: w[i],
            h_um: h[i],
            poly: poly[i].clone(),
            rotation: rot[i].unwrap_or(0),
        })
        .collect())
}

fn ceilings_batch(rows: &[CeilingRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("chip", false, u32s(col!(rows, |r| Some(r.chip))))
        .push("kind", false, u8s(col!(rows, |r| Some(r.kind))))
        .push("name", false, utf8s(col!(rows, |r| r.name.as_str())))
        .push("level", true, u8s(col!(rows, |r| r.level)))
        .push("value", false, f64v(col!(rows, |r| r.value)))
        .push("low", true, f64s(col!(rows, |r| r.low)))
        .push("high", true, f64s(col!(rows, |r| r.high)));
    batch("ceilings", c)
}

fn read_ceilings(b: &[RecordBatch]) -> Result<Vec<CeilingRow>, Diagnostic> {
    let t = Table::new("ceilings", b);
    let chip = t.u32("chip")?;
    let kind = t.int::<u8>("kind")?;
    let name = t.str("name")?;
    let level = t.opt_int::<u8>("level")?;
    let v = t.f64("value")?;
    let lo = t.opt_f64("low")?;
    let hi = t.opt_f64("high")?;
    Ok((0..t.rows())
        .map(|i| CeilingRow {
            chip: chip[i],
            kind: kind[i],
            name: name[i].clone(),
            level: level[i],
            value: v[i],
            low: lo[i],
            high: hi[i],
        })
        .collect())
}

fn diagnostics_batch(rows: &[DiagnosticRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("code", false, utf8s(col!(rows, |r| r.code.as_str())))
        .push("severity", false, u8s(col!(rows, |r| Some(r.severity))))
        .push("message", false, utf8s(col!(rows, |r| r.message.as_str())))
        .push("path", true, utf8(col!(rows, |r| r.path.as_deref())))
        .push("hint", true, utf8(col!(rows, |r| r.hint.as_deref())));
    batch("diagnostics", c)
}

fn read_diagnostics(b: &[RecordBatch]) -> Result<Vec<DiagnosticRow>, Diagnostic> {
    let t = Table::new("diagnostics", b);
    let code = t.str("code")?;
    let sev = t.int::<u8>("severity")?;
    let msg = t.str("message")?;
    let path = t.opt_str("path")?;
    let hint = t.opt_str("hint")?;
    Ok((0..t.rows())
        .map(|i| DiagnosticRow {
            code: code[i].clone(),
            severity: sev[i],
            message: msg[i].clone(),
            path: path[i].clone(),
            hint: hint[i].clone(),
        })
        .collect())
}

fn run_scalars_batch(rows: &[(String, String)]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("name", false, utf8s(rows.iter().map(|r| r.0.as_str())))
        .push("json", false, utf8s(rows.iter().map(|r| r.1.as_str())));
    batch("run_scalars", c)
}

fn read_run_scalars(b: &[RecordBatch]) -> Result<Vec<(String, String)>, Diagnostic> {
    let t = Table::new("run_scalars", b);
    let n = t.str("name")?;
    let j = t.str("json")?;
    Ok(n.into_iter().zip(j).collect())
}

fn spans_batch(rows: &[SpanRow]) -> RecordBatch {
    use arrowx::*;
    let mut c = Cols::new();
    c.push("resource", false, u32s(col!(rows, |r| Some(r.resource))))
        .push("lane", false, u16s(col!(rows, |r| Some(r.lane))))
        .push("kind", false, u8s(col!(rows, |r| Some(r.kind))))
        .push("flags", false, u8s(col!(rows, |r| Some(r.flags))))
        .push("op", false, u32s(col!(rows, |r| Some(r.op))))
        .push("task", false, u32s(col!(rows, |r| Some(r.task))))
        .push("slice", false, u32s(col!(rows, |r| Some(r.slice))))
        .push("t_start", false, i64s(col!(rows, |r| Some(r.t_start))))
        .push("dur", false, i64s(col!(rows, |r| Some(r.dur))))
        .push("bytes", false, f64v(col!(rows, |r| r.bytes)))
        .push(
            "energy_j",
            false,
            std::sync::Arc::new(arrow_array::Float32Array::from_iter_values(col!(
                rows,
                |r| r.energy_j
            ))),
        );
    batch("spans", c)
}

fn read_spans(b: &[RecordBatch]) -> Result<Vec<SpanRow>, Diagnostic> {
    let t = Table::new("spans", b);
    let res = t.u32("resource")?;
    let lane = t.opt_int::<u16>("lane")?;
    let kind = t.int::<u8>("kind")?;
    let flags = t.opt_int::<u8>("flags")?;
    let op = t.opt_u32("op")?;
    let task = t.opt_u32("task")?;
    let slice = t.opt_u32("slice")?;
    let ts = t.i64("t_start")?;
    let dur = t.i64("dur")?;
    let bytes = t.opt_f64("bytes")?;
    let e = t.opt_f64("energy_j")?;
    Ok((0..t.rows())
        .map(|i| SpanRow {
            resource: res[i],
            lane: lane[i].unwrap_or(0),
            kind: kind[i],
            flags: flags[i].unwrap_or(0),
            op: op[i].unwrap_or(NONE_U32),
            task: task[i].unwrap_or(0),
            slice: slice[i].unwrap_or(0),
            t_start: ts[i],
            dur: dur[i],
            bytes: bytes[i].unwrap_or(0.0),
            energy_j: e[i].unwrap_or(0.0) as f32,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families() {
        assert_eq!(op_family("layers.gate_up.i1.k0"), "layers.gate_up");
        assert_eq!(op_family("lm_head.k0"), "lm_head");
        assert_eq!(op_family("embed"), "embed");
        assert_eq!(op_family("k0"), "k0");
        assert_eq!(op_family("layers.ik.i2"), "layers.ik");
    }
}
