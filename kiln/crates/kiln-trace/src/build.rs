//! Builds a [`Trace`] from an evaluation (06 `kiln.result/1` with its 03 `SimResult`s) and, when given, the
//! expanded hardware model: the resource hierarchy, memory levels, structural peaks (roofline ceilings) and
//! link bandwidths come from the model; without it the hierarchy is recovered from dotted resource paths.
//!
//! Tier A writes `summary` (tables without spans) or `ops` (plus one estimated span per op on its binding
//! resource, 05 §3.3). Phases are laid out back to back on one time axis (`phases.t_offset`).

use std::collections::{BTreeMap, BTreeSet};

use kiln_ir::common::Severity;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::compute::ComputeKind;
use kiln_ir::hw::model::{ContainerKind, MemSpec, NodeIx};

use crate::container::{Headline, Manifest};
use crate::interval::{Corner, Interval};
use crate::layout;
use crate::phys::{self, Placed, ProfileCheck};
use crate::provenance::TraceLevel;
use crate::result::EvalResult;
use crate::sim::{BindingClass, ResourceKind, SimResult, Target};
use crate::trace::*;

pub struct BuildInput<'a> {
    pub result: &'a EvalResult,
    pub hw: Option<&'a HwModel>,
    /// Canonical design JSON embedded in the manifest so the trace is self-contained (05 §3.5).
    pub design_json: Option<serde_json::Value>,
    pub design_name: Option<String>,
    pub workload_name: Option<String>,
    pub level: TraceLevel,
    /// A placed floorplan and its placer id; `None` = kiln-phys's placement of `hw` (the unplaced hierarchy
    /// layout without a model, or when kiln-phys fails).
    pub floorplan: Option<(Vec<FloorplanRow>, String)>,
    /// Validation of the design per profile, for the design sheet.
    pub checks: &'a [ProfileCheck],
}

/// 03 binding class -> `limiters.binding` code (default enum order).
pub fn binding_code(c: BindingClass) -> u8 {
    match c {
        BindingClass::Compute => 0,
        BindingClass::Dram => 1,
        BindingClass::Link => 2,
        BindingClass::Port => 3,
        BindingClass::Dependency => 4,
        BindingClass::Overhead => 5,
        BindingClass::Nmp => 6,
        BindingClass::Contention => 7,
        BindingClass::PipelineBubble => 8,
    }
}

fn span_kind_for(c: BindingClass) -> &'static str {
    match c {
        BindingClass::Compute => "compute",
        BindingClass::Dram | BindingClass::Port => "load",
        BindingClass::Link => "transfer_send",
        BindingClass::Overhead => "launch_overhead",
        BindingClass::Dependency | BindingClass::PipelineBubble => "stall_dependency",
        BindingClass::Contention => "stall_contention",
        BindingClass::Nmp => "nmp_compute",
    }
}

/// Coarse op kind from the op path for coloring (the workload IR's op tag is not carried by `SimResult`).
pub fn op_kind_of(family: &str, macs: u64) -> &'static str {
    let f = family
        .rsplit('.')
        .next()
        .unwrap_or(family)
        .to_ascii_lowercase();
    let has = |xs: &[&str]| xs.iter().any(|x| f.contains(x));
    if has(&[
        "all_reduce",
        "allreduce",
        "all_gather",
        "allgather",
        "reduce_scatter",
        "all_to_all",
        "collective",
        "send",
        "recv",
    ]) {
        "collective"
    } else if has(&["attn", "attention", "sdpa", "softmax"]) && !has(&["norm"]) {
        "attention"
    } else if has(&["norm"]) {
        "norm"
    } else if has(&["embed"]) {
        "embedding"
    } else if macs > 0 {
        "matmul"
    } else if has(&[
        "kv", "cache", "append", "copy", "select", "gather", "sample",
    ]) {
        "memory"
    } else if has(&["rope", "rotary"]) {
        "rope"
    } else {
        "elementwise"
    }
}

struct Res {
    path: String,
    /// Template entity path (01 instance suffixes stripped); 03 names multi-instance memory groups by it.
    entity: Option<String>,
    kind: &'static str,
    class: &'static str,
    mem_level: Option<u8>,
    capacity_b: Option<f64>,
    peak_bw_bps: Option<f64>,
    array: Option<(String, u32)>,
}

fn class_of(kind: &str) -> &'static str {
    match kind {
        k if k.starts_with("unit_") || k == "nmp_unit" => "compute",
        "memory" | "mem_stack" | "bank" => "memory",
        "network" | "router" | "channel" | "switch" | "dma" => "interconnect",
        "port" => "io",
        "sequencer" => "compute",
        _ => "container",
    }
}

fn hw_resources(hw: &HwModel) -> (Vec<Res>, BTreeMap<String, f64>) {
    let mut out = Vec::with_capacity(hw.nodes.len());
    for n in &hw.nodes {
        if !n.enabled {
            continue;
        }
        let (kind, mem_level, capacity_b, peak_bw_bps) = match n.ix {
            NodeIx::Container(c) => (
                match hw.tree[c].kind {
                    ContainerKind::System => "system",
                    ContainerKind::Host => "host",
                    ContainerKind::Board => "board",
                    ContainerKind::Package => "package",
                    ContainerKind::Die => "die",
                    ContainerKind::Cluster => "cluster",
                    ContainerKind::Switch => "switch",
                },
                None,
                None,
                None,
            ),
            NodeIx::Unit(u) => {
                let u = &hw.units[u];
                let k = if u.near.is_some() {
                    "nmp_unit"
                } else {
                    match u.spec.kind {
                        ComputeKind::Matrix(_) => "unit_matrix",
                        ComputeKind::Vector(_) => "unit_vector",
                        ComputeKind::Scalar(_) => "unit_scalar",
                        ComputeKind::Special(_) => "unit_special",
                        ComputeKind::Cim(_) => "unit_cim",
                    }
                };
                (k, None, None, None)
            }
            NodeIx::Mem(m) => {
                let mi = &hw.memories[m];
                let lvl = hw.levels.get(m).copied().filter(|&l| l != u8::MAX);
                let k = if matches!(mi.spec, MemSpec::Stack(_)) {
                    "mem_stack"
                } else {
                    "memory"
                };
                (
                    k,
                    lvl,
                    Some(mi.capacity.0 as f64),
                    mi.bandwidth.map(|b| b.0),
                )
            }
            NodeIx::Block(_) => ("block", None, None, None),
            NodeIx::Router(_) => ("router", None, None, None),
            NodeIx::Port(_) => ("port", None, None, None),
            NodeIx::Net(_) => ("network", None, None, None),
        };
        out.push(Res {
            path: n.path.clone(),
            entity: Some(n.entity.clone()),
            kind,
            class: class_of(kind),
            mem_level,
            capacity_b,
            peak_bw_bps,
            array: (n.count > 1).then(|| (n.entity.clone(), n.index)),
        });
    }
    // Every channel is a resource (05 §3.4 `channel`: expanded link/port pair), named as 03 names links.
    let mut chan_bw = BTreeMap::new();
    for c in &hw.channels {
        if !hw.enabled(c.src) || !hw.enabled(c.dst) {
            continue;
        }
        let key = format!("{}-{}", hw.path(c.src), hw.path(c.dst));
        if chan_bw.contains_key(&key) {
            continue;
        }
        chan_bw.insert(key.clone(), c.bandwidth.map_or(0.0, |b| b.0));
        out.push(Res {
            path: key,
            entity: None,
            kind: "channel",
            class: "interconnect",
            mem_level: None,
            capacity_b: None,
            peak_bw_bps: c.bandwidth.map(|b| b.0),
            array: None,
        });
    }
    (out, chan_bw)
}

fn res_kind(k: ResourceKind) -> &'static str {
    match k {
        ResourceKind::ComputeUnit => "unit_vector",
        ResourceKind::Link | ResourceKind::MemPort => "channel",
        ResourceKind::Bank => "bank",
        ResourceKind::DramChannel => "mem_stack",
        ResourceKind::NmpSite => "nmp_unit",
        ResourceKind::Sequencer => "sequencer",
        ResourceKind::DmaEngine => "dma",
    }
}

/// Longest proper dotted prefix of `path` that is in `known`, also trying each side of an `a-b` link name.
fn parent_path(path: &str, known: &BTreeSet<String>) -> Option<String> {
    let mut best: Option<String> = None;
    let mut consider = |p: &str| {
        let mut cur = p;
        while let Some(i) = cur.rfind('.') {
            cur = &cur[..i];
            if known.contains(cur) && cur != path {
                if best.as_ref().is_none_or(|b| cur.len() > b.len()) {
                    best = Some(cur.to_string());
                }
                return;
            }
        }
    };
    consider(path);
    if best.is_none() {
        for (i, _) in path.match_indices('-') {
            let (a, b) = (&path[..i], &path[i + 1..]);
            if known.contains(a) && known.contains(b) {
                return lca(a, b, known);
            }
        }
    }
    best
}

fn lca(a: &str, b: &str, known: &BTreeSet<String>) -> Option<String> {
    let pa: Vec<&str> = a.split('.').collect();
    let pb: Vec<&str> = b.split('.').collect();
    let n = pa.iter().zip(&pb).take_while(|(x, y)| x == y).count();
    (1..=n)
        .rev()
        .map(|k| pa[..k].join("."))
        .find(|p| known.contains(p))
}

pub fn build(input: &BuildInput) -> Trace {
    let r = input.result;
    let mut m = Manifest::new(input.level.max(TraceLevel::Summary), r.provenance.clone());
    m.design = input.design_json.clone();
    m.design_name = input.design_name.clone();
    m.workload_name = input.workload_name.clone();
    m.enums = default_enums();
    let mut t = Trace::empty(m);

    // Resources: model nodes, plus engine resources (links, ports, sequencer) and memory paths not in it.
    let (mut res, chan_bw) = match input.hw {
        Some(hw) => hw_resources(hw),
        None => (vec![], BTreeMap::new()),
    };
    let centrals: Vec<&SimResult> = r
        .sim
        .iter()
        .filter(|s| s.corner == Corner::Central)
        .collect();
    let mut known: BTreeSet<String> = res.iter().map(|x| x.path.clone()).collect();
    let entity_level: BTreeMap<String, u8> = res
        .iter()
        .filter_map(|x| Some((x.entity.clone()?, x.mem_level?)))
        .collect();
    let entities: BTreeSet<&str> = entity_level.keys().map(String::as_str).collect();
    let mut extra: BTreeMap<String, &'static str> = BTreeMap::new();
    for s in &centrals {
        for x in &s.resources {
            if !known.contains(x.resource.as_str()) {
                extra
                    .entry(x.resource.to_string())
                    .or_insert(res_kind(x.kind));
            }
        }
        for o in &s.ops {
            for b in &o.bytes_by_level {
                if !known.contains(&b.level) && !entities.contains(b.level.as_str()) {
                    extra.entry(b.level.clone()).or_insert("memory");
                }
            }
        }
    }
    if input.hw.is_none() {
        // Recover containers from dotted prefixes (not link names).
        let mut prefixes = BTreeSet::new();
        for p in extra.keys() {
            let head = p.split('-').next().unwrap_or(p);
            let mut cur = head;
            while let Some(i) = cur.rfind('.') {
                cur = &cur[..i];
                prefixes.insert(cur.to_string());
            }
        }
        for p in prefixes {
            extra.entry(p).or_insert("cluster");
        }
    }
    known.extend(extra.keys().cloned());
    for (p, k) in &extra {
        res.push(Res {
            path: p.clone(),
            entity: None,
            kind: k,
            class: class_of(k),
            mem_level: None,
            capacity_b: None,
            peak_bw_bps: chan_bw.get(p).copied(),
            array: None,
        });
    }
    res.sort_by(|a, b| a.path.cmp(&b.path));
    let index: BTreeMap<&str, u32> = res
        .iter()
        .enumerate()
        .map(|(i, x)| (x.path.as_str(), i as u32))
        .collect();
    let mut arrays: BTreeMap<String, u32> = BTreeMap::new();
    for x in &res {
        if let Some((e, _)) = &x.array {
            let n = arrays.len() as u32;
            arrays.entry(e.clone()).or_insert(n);
        }
    }
    let code_of = |table: &str, name: &str| code(&t.manifest.enums, table, name).unwrap_or(0);
    let mut rows: Vec<ResourceRow> = res
        .iter()
        .map(|x| ResourceRow {
            path: x.path.clone(),
            kind: code_of("resources.kind", x.kind) as u16,
            class: code_of("resources.class", x.class) as u8,
            parent: parent_path(&x.path, &known).and_then(|p| index.get(p.as_str()).copied()),
            chip: 0,
            array_id: x.array.as_ref().map(|(e, _)| arrays[e]),
            array_pos: x.array.as_ref().map(|(_, i)| *i),
            mem_level: x.mem_level,
            capacity_b: x.capacity_b,
            peak_bw_bps: x.peak_bw_bps,
            peak_flops: vec![],
            lanes: 1,
        })
        .collect();
    let package = code_of("resources.kind", "package") as u16;
    for i in 0..rows.len() {
        let mut cur = Some(i as u32);
        let mut root = i as u32;
        let mut chip = None;
        while let Some(c) = cur {
            if rows[c as usize].kind == package {
                chip = Some(c);
                break;
            }
            root = c;
            cur = rows[c as usize].parent;
        }
        rows[i].chip = chip.unwrap_or(root);
    }
    t.resources = rows;
    let resolve = |p: &str| index.get(p).copied();
    let mut level_of: BTreeMap<&str, u8> = res
        .iter()
        .filter_map(|x| Some((x.path.as_str(), x.mem_level?)))
        .collect();
    for (e, l) in &entity_level {
        level_of.entry(e.as_str()).or_insert(*l);
    }
    let offchip_level: Option<u8> = res
        .iter()
        .filter(|x| x.kind == "mem_stack")
        .filter_map(|x| x.mem_level)
        .min();

    // Ceilings from the structural summary of every chip.
    if let Some(hw) = input.hw {
        let s = hw.summary();
        for c in &s.chips {
            let Some(chip) = resolve(&c.path) else {
                continue;
            };
            let mut best: BTreeMap<String, f64> = BTreeMap::new();
            for (k, v) in &c.peak_ops {
                if k.ends_with(":sparse") {
                    continue;
                }
                let e = best.entry(k.clone()).or_insert(0.0);
                *e = e.max(*v);
            }
            for (k, v) in best {
                t.ceilings.push(CeilingRow {
                    chip,
                    kind: 0,
                    name: k,
                    level: None,
                    value: v,
                    low: None,
                    high: None,
                });
            }
            for l in &c.levels {
                if l.level == u8::MAX || l.bandwidth.0 <= 0.0 {
                    continue;
                }
                let name = level_name(&res, l.level, offchip_level);
                t.ceilings.push(CeilingRow {
                    chip,
                    kind: 1,
                    name,
                    level: Some(l.level),
                    value: l.bandwidth.0,
                    low: None,
                    high: None,
                });
            }
        }
        for n in &s.inter_chip {
            if let (Some(bw), Some(&chip)) = (
                n.link_bandwidth,
                t.resources
                    .iter()
                    .position(|r| r.kind == package)
                    .map(|x| x as u32)
                    .as_ref(),
            ) {
                t.ceilings.push(CeilingRow {
                    chip,
                    kind: 2,
                    name: n.path.clone(),
                    level: None,
                    value: bw.0,
                    low: None,
                    high: None,
                });
            }
        }
    }

    // Phases, ops, limiters, groups, aggregates.
    let tick = t.manifest.tick_s;
    let ticks = |s: f64| (s / tick).round() as i64;
    let mut kinds: Vec<String> = vec![];
    let mut phase_names: Vec<String> = vec![];
    let mut t_offset = 0i64;
    let root_res = t
        .resources
        .iter()
        .position(|r| r.kind == package)
        .or_else(|| {
            t.resources
                .iter()
                .position(|r| r.parent.is_none() && !r.path.is_empty())
        })
        .map(|i| i as u32);
    for s in &centrals {
        let pcode = phase_names.len() as u8;
        phase_names.push(s.phase.to_string());
        let pr = r.phase(s.phase.as_str());
        let corners: Vec<&SimResult> = r
            .sim
            .iter()
            .filter(|x| x.phase == s.phase && x.corner != Corner::Central)
            .collect();
        let op_base = t.ops.len() as u32;
        let mut op_index: BTreeMap<&str, u32> = BTreeMap::new();
        let mut t_end = 0i64;
        for o in &s.ops {
            let family = op_family(o.op.as_str());
            let kname = op_kind_of(&family, o.macs_useful);
            let kcode = match kinds.iter().position(|k| k == kname) {
                Some(i) => i,
                None => {
                    kinds.push(kname.into());
                    kinds.len() - 1
                }
            } as u16;
            let mut bbl: Vec<f64> = vec![];
            let mut link_bytes = 0.0;
            for b in &o.bytes_by_level {
                if let Some(&l) = level_of.get(b.level.as_str()) {
                    let l = usize::from(l);
                    if bbl.len() <= l {
                        bbl.resize(l + 1, 0.0);
                    }
                    bbl[l] += b.bytes as f64;
                } else if b.level.contains('-') {
                    link_bytes += b.bytes as f64;
                }
            }
            let times: Vec<f64> = corners
                .iter()
                .filter_map(|c| c.op(o.op.as_str()).map(|x| x.time_s()))
                .collect();
            let (lo, hi) = if times.is_empty() {
                (None, None)
            } else {
                let i = Interval::from_corners(
                    o.time_s(),
                    times[0],
                    *times.get(1).unwrap_or(&times[0]),
                );
                (Some(i.low), Some(i.high))
            };
            let idx = t.ops.len() as u32;
            op_index.insert(o.op.as_str(), idx);
            let (ts, te) = (t_offset + ticks(o.start_s), t_offset + ticks(o.end_s));
            t_end = t_end.max(te);
            let chip = o
                .binding
                .resource()
                .and_then(|p| resolve(p.as_str()))
                .map(|i| t.resources[i as usize].chip);
            t.ops.push(OpRow {
                path: o.op.to_string(),
                family,
                kind: kcode,
                phase: pcode,
                layer: o.layer.map(|l| l as i32),
                flops: 2.0 * o.macs_useful as f64,
                precision: 0,
                bytes_by_level: bbl,
                link_bytes,
                t_start: ts,
                t_end: te,
                chips: chip.into_iter().collect(),
                mapping: NONE_U32,
                macs_useful: o.macs_useful,
                macs_issued: o.macs_issued,
                target: u8::from(o.target == Target::Nmp),
                host_vs_nmp_s: o.host_vs_nmp_s,
                group: o.group.unwrap_or(NONE_U32),
                energy_j: o.energy.total_j,
                time_low_s: lo,
                time_high_s: hi,
            });
            let res_of = o.binding.resource().and_then(|p| resolve(p.as_str()));
            t.limiters.push(LimiterRow {
                op: idx,
                group: o.group.unwrap_or(NONE_U32),
                binding: binding_code(o.binding.class()),
                resource: res_of,
                time_s: o.time_s(),
                attained_frac: 1.0,
                rank: 0,
                share: 1.0,
            });
            if let Some(ru) = &o.runner_up {
                t.limiters.push(LimiterRow {
                    op: idx,
                    group: o.group.unwrap_or(NONE_U32),
                    binding: binding_code(ru.binding.class()),
                    resource: ru.binding.resource().and_then(|p| resolve(p.as_str())),
                    time_s: ru.ratio * o.time_s(),
                    attained_frac: ru.ratio,
                    rank: 1,
                    share: 0.0,
                });
            }
            let on = res_of.or(root_res).unwrap_or(0);
            let e = &o.energy;
            let mut comp = |name: &str, j: f64| {
                if j != 0.0 {
                    t.aggregates_op_resource.push(AggOpResourceRow {
                        op: idx,
                        resource: on,
                        time_s: o.time_s(),
                        energy_component: code(
                            &t.manifest.enums,
                            "aggregates_op_resource.energy_component",
                            name,
                        )
                        .unwrap_or(7) as u8,
                        energy_j: j,
                    });
                }
            };
            comp("compute", e.compute_j);
            let dram: f64 = e
                .memory_j
                .iter()
                .filter(|(k, _)| Some(k.as_str()) == offchip_name(&res, offchip_level).as_deref())
                .map(|x| *x.1)
                .sum();
            let mem: f64 = e.memory_j.values().sum::<f64>() - dram;
            comp("sram_read", mem);
            comp("dram", dram);
            let noc: f64 = e
                .link_j
                .iter()
                .filter(|(k, _)| k.contains("noc"))
                .map(|x| *x.1)
                .sum();
            comp("noc", noc);
            comp("link", e.link_j.values().sum::<f64>() - noc);
            comp("leakage", e.static_j);
            comp("other", e.nmp_j + e.conversion_j + e.padding_j);
            if input.level >= TraceLevel::Ops && te > ts {
                let bc = o.binding.class();
                t.spans.push(SpanRow {
                    resource: on,
                    lane: 0,
                    kind: code(&t.manifest.enums, "spans.kind", span_kind_for(bc)).unwrap_or(13)
                        as u8,
                    flags: span_flags::ESTIMATED,
                    op: idx,
                    task: idx - op_base,
                    slice: 0,
                    t_start: ts,
                    dur: te - ts,
                    bytes: o.bytes_by_level.iter().map(|b| b.bytes as f64).sum(),
                    energy_j: o.energy.total_j as f32,
                });
            }
        }
        for g in &s.groups {
            t_end = t_end.max(t_offset + ticks(g.end_s));
            t.groups.push(GroupRow {
                group: g.group,
                phase: pcode,
                ops: g
                    .ops
                    .iter()
                    .filter_map(|o| op_index.get(o.as_str()).copied())
                    .collect(),
                kind: match g.kind {
                    crate::sim::GroupKind::Single => 0,
                    crate::sim::GroupKind::Fused => 1,
                    crate::sim::GroupKind::Pipelined => 2,
                    crate::sim::GroupKind::LayerByLayer => 3,
                },
                binding: binding_code(g.binding.class()),
                t_start: t_offset + ticks(g.start_s),
                t_end: t_offset + ticks(g.end_s),
                bubble_s: g.bubble_s,
                exposed_overhead_s: g.exposed_overhead_s,
            });
        }
        for (k, c) in s.collectives.iter().enumerate() {
            t.collectives.push(CollectiveRow {
                collective: k as u32,
                name: c.collective.to_string(),
                phase: pcode,
                op: op_index.get(c.op.as_str()).copied().unwrap_or(NONE_U32),
                algorithm: c.algorithm.clone(),
                group_chips: c.chips.iter().filter_map(|p| resolve(p.as_str())).collect(),
                steps: c.steps.min(u32::from(u16::MAX)) as u16,
                bytes: c.bytes,
                t_start: t_offset + ticks(c.start_s),
                t_end: t_offset + ticks(c.end_s),
                link_bytes_by_tier: c.link_bytes_by_tier.clone(),
            });
        }
        for x in &s.resources {
            let Some(ri) = resolve(x.resource.as_str()) else {
                continue;
            };
            let p = if s.makespan_s > 0.0 {
                x.energy_j / s.makespan_s
            } else {
                0.0
            };
            t.aggregates_resource.push(AggResourceRow {
                resource: ri,
                phase: pcode,
                busy_s: x.busy_s,
                stall_s: x.stall_s,
                bytes: x.bytes,
                flops: 2.0 * x.macs as f64,
                energy_dyn_j: x.energy_j,
                energy_leak_j: 0.0,
                avg_power_w: p,
                peak_power_w: p,
                power_density_w_mm2: None,
                peak_temp_k: None,
                utilization: x.utilization,
            });
        }
        let b = &s.bottleneck;
        for (c, x) in &b.time_by_binding {
            t.bottleneck.push(BottleneckRow {
                phase: pcode,
                section: 0,
                binding: Some(binding_code(*c)),
                resource: None,
                time_s: Some(*x),
                utilization: None,
                shadow_price: None,
                slack: None,
            });
        }
        for x in &b.top_resources {
            t.bottleneck.push(BottleneckRow {
                phase: pcode,
                section: 1,
                binding: None,
                resource: resolve(x.resource.as_str()),
                time_s: None,
                utilization: Some(x.utilization),
                shadow_price: Some(x.shadow_price),
                slack: None,
            });
        }
        for x in &b.slack {
            t.bottleneck.push(BottleneckRow {
                phase: pcode,
                section: 2,
                binding: None,
                resource: resolve(x.resource.as_str()),
                time_s: None,
                utilization: None,
                shadow_price: None,
                slack: Some(x.slack),
            });
        }
        let (lo, hi) = pr.map_or((None, None), |p| (Some(p.time_s.low), Some(p.time_s.high)));
        t.phases.push(PhaseRow {
            phase: pcode,
            id: s.phase.to_string(),
            scope: format!("{:?}", s.scope).to_lowercase(),
            t_offset,
            t_end,
            makespan_s: s.makespan_s,
            makespan_low_s: lo,
            makespan_high_s: hi,
            t_a0_s: s.t_a0_s,
            t_a2_s: s.t_a2_s,
            energy_j: s.energy.total_j,
            avg_power_w: s.power.avg_w,
            tokens_per_s: pr.map(|p| p.tokens_per_s.central),
            summary: b.summary.clone(),
        });
        let j = |v: serde_json::Value| kiln_ir::common::canonical_json(&v);
        t.run_scalars.push((
            format!("energy.{}", s.phase),
            j(serde_json::to_value(&s.energy).expect("energy")),
        ));
        t.run_scalars.push((
            format!("power.{}", s.phase),
            j(serde_json::to_value(&s.power).expect("power")),
        ));
        t.run_scalars.push((
            format!("invariants.{}", s.phase),
            j(serde_json::to_value(&s.invariants).expect("invariants")),
        ));
        t.run_scalars.push((
            format!("calibration.{}", s.phase),
            j(serde_json::to_value(&s.calibration).expect("calibration")),
        ));
        t_offset = t_end.max(t_offset + 1);
    }
    t.manifest.enums.insert("ops.kind".into(), kinds);
    t.manifest.enums.insert("ops.phase".into(), phase_names);
    t.manifest
        .enums
        .insert("ops.precision".into(), vec!["unknown".into()]);
    let mut summary = r.to_value();
    if let Some(o) = summary.as_object_mut() {
        o.remove("sim");
        o.remove("timing");
    }
    t.run_scalars
        .push(("result".into(), kiln_ir::common::canonical_json(&summary)));
    assign_lanes(&mut t);

    // Diagnostics: result errors, warnings, violations, failed invariants.
    for (sev, list) in [(0u8, &r.violations), (0, &r.errors), (1, &r.warnings)] {
        for e in list {
            t.diagnostics.push(DiagnosticRow {
                code: e.diag.code.clone(),
                severity: if e.diag.severity == Severity::Error {
                    sev
                } else {
                    1
                },
                message: e.diag.message.clone(),
                path: e.diag.path.clone(),
                hint: e.diag.hint.clone(),
            });
        }
    }
    for s in &centrals {
        for c in s.invariants.failures() {
            t.diagnostics.push(DiagnosticRow {
                code: c.id.code(),
                severity: 0,
                message: format!("{}: {}", s.phase, c.message),
                path: c.path.clone(),
                hint: None,
            });
        }
    }

    let placed = match (&input.floorplan, input.hw) {
        (None, Some(hw)) => match place_phys(hw, &index, input.checks) {
            Ok(p) => Some(p),
            Err(why) => {
                t.manifest.notes.push(format!(
                    "kiln-phys placement unavailable ({why}): the floorplan falls back to the unplaced hierarchy layout"
                ));
                None
            }
        },
        _ => None,
    };
    if let Some((rows, source)) = &input.floorplan {
        t.floorplan = rows.clone();
        t.manifest.floorplan_source = Some(source.clone());
    } else if let Some(p) = placed {
        t.floorplan = p.floorplan;
        t.wires = p.wires;
        t.package_geometry = p.package;
        t.manifest.floorplan_source = Some(p.source);
        t.manifest.notes.push("floorplan: kiln-phys placement (dies, die-level macros, shoreline PHYs, memory stacks, packages side by side) and its layout inside macros; compute units are filled in by area inside their placed parent because kiln-phys arranges dies without unit area; wires are Manhattan routes between endpoint positions with kiln-phys link lengths and costs".into());
        t.run_scalars.push((
            "design_summary".into(),
            kiln_ir::common::canonical_json(&p.summary),
        ));
        if t.manifest.headline.area_mm2.is_none() {
            t.manifest.headline.area_mm2 = Some(Interval::point(p.package_mm2));
        }
    } else {
        t.floorplan = layout::unplaced_floorplan(&t);
        t.manifest.floorplan_source =
            (!t.floorplan.is_empty()).then(|| layout::UNPLACED.to_string());
    }
    if t.manifest.floorplan_source.as_deref() == Some(layout::UNPLACED) {
        t.manifest.notes.push("floorplan is unplaced: a hierarchy layout with nominal sizes per kind, not a physical placement (kiln-phys placement pending)".into());
    }
    if input.level >= TraceLevel::Ops {
        t.manifest.notes.push("spans are Tier A estimates (one per op on its binding resource, no contention); whole-step makespans extrapolate the middle iteration of the scheduled repeat window".into());
    }
    let area = t.manifest.headline.area_mm2;
    t.manifest.headline = headline(r);
    if t.manifest.headline.area_mm2.is_none() {
        t.manifest.headline.area_mm2 = area;
    }
    t.manifest.headline.floor_failures = t
        .diagnostics
        .iter()
        .filter(|d| d.code.starts_with("E-FLOOR"))
        .map(|d| d.code.clone())
        .collect();
    t
}

/// kiln-phys's model of `hw` placed into trace rows; a kiln-phys panic or a design without dies is an error.
fn place_phys(
    hw: &HwModel,
    index: &BTreeMap<&str, u32>,
    checks: &[ProfileCheck],
) -> Result<Placed, String> {
    let run = || {
        let ph = kiln_phys::Phys::new(hw);
        phys::place(hw, &ph, &|p: &str| index.get(p).copied(), checks)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err("no placed dies or packages".into()),
        Err(e) => Err(e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "kiln-phys panicked".into())),
    }
}

/// A structure-only trace of `hw` (no simulation, 05 §8 `kiln viz <design>`): resources, kiln-phys floorplan,
/// wires, package geometry, roofline ceilings and the design sheet.
pub fn structure(
    hw: &HwModel,
    design_json: Option<serde_json::Value>,
    design_name: Option<String>,
    checks: &[ProfileCheck],
) -> Trace {
    let mut prov = crate::Provenance::unknown(crate::Tier::A);
    prov.design_hash = hw.design_hash.clone();
    let r = EvalResult {
        schema: crate::RESULT_SCHEMA.into(),
        status: crate::result::Status::Ok,
        score: 0.0,
        score_interval: None,
        score_components: None,
        score_realistic: None,
        interval: Default::default(),
        stage_reached: crate::result::Stage::S0,
        tier: None,
        phases: vec![],
        ops: vec![],
        physical: None,
        features: BTreeMap::new(),
        violations: vec![],
        errors: vec![],
        warnings: vec![],
        audit: Default::default(),
        trace: None,
        provenance: prov,
        timing: Default::default(),
        calibration: None,
        invariants: None,
        sim: vec![],
    };
    let mut t = build(&BuildInput {
        result: &r,
        hw: Some(hw),
        design_json,
        design_name,
        workload_name: None,
        level: TraceLevel::Summary,
        floorplan: None,
        checks,
    });
    t.manifest
        .notes
        .push("structure only: no simulation (run kiln eval -o run.kiln for metrics)".into());
    t
}

fn level_name(res: &[Res], level: u8, offchip: Option<u8>) -> String {
    if Some(level) == offchip {
        return "DRAM".into();
    }
    let name = res
        .iter()
        .find(|x| x.mem_level == Some(level))
        .map(|x| x.path.rsplit('.').next().unwrap_or(&x.path).to_string());
    format!("L{level} {}", name.unwrap_or_default())
        .trim()
        .to_string()
}

/// Key of the off-chip level in `EnergyBreakdown::memory_j` (`l<level+1>` as 03 names levels).
fn offchip_name(_res: &[Res], offchip: Option<u8>) -> Option<String> {
    offchip.map(|l| format!("l{}", u32::from(l) + 1))
}

fn headline(r: &EvalResult) -> Headline {
    let first = r.phases.first();
    let sum = |f: &dyn Fn(&crate::result::PhaseResult) -> Interval| -> Option<Interval> {
        (!r.phases.is_empty()).then(|| {
            r.phases
                .iter()
                .map(f)
                .fold(Interval::point(0.0), |a, b| Interval {
                    low: a.low + b.low,
                    central: a.central + b.central,
                    high: a.high + b.high,
                })
        })
    };
    Headline {
        latency_s: first.map(|p| p.time_s),
        tokens_per_s: first.map(|p| p.tokens_per_s),
        energy_j: sum(&|p| p.energy_j),
        area_mm2: r.physical.as_ref().map(|p| p.package_mm2),
        power_w: r
            .physical
            .as_ref()
            .map(|p| p.peak_power_w)
            .or_else(|| first.map(|p| p.avg_power_w)),
        score: r.score_interval,
        floor_failures: vec![],
    }
}

/// A trace from a single `SimResult` (no evaluation wrapper).
pub fn from_sim(sim: &SimResult, hw: Option<&HwModel>, level: TraceLevel) -> Trace {
    let r = EvalResult {
        schema: crate::RESULT_SCHEMA.into(),
        status: crate::result::Status::Ok,
        score: 0.0,
        score_interval: None,
        score_components: None,
        score_realistic: None,
        interval: Default::default(),
        stage_reached: crate::result::Stage::S2,
        tier: Some(sim.tier),
        phases: vec![],
        ops: vec![],
        physical: None,
        features: BTreeMap::new(),
        violations: vec![],
        errors: vec![],
        warnings: vec![],
        audit: Default::default(),
        trace: None,
        provenance: sim.provenance.clone(),
        timing: Default::default(),
        calibration: None,
        invariants: None,
        sim: vec![SimResult {
            corner: Corner::Central,
            ..sim.clone()
        }],
    };
    build(&BuildInput {
        result: &r,
        hw,
        design_json: None,
        design_name: None,
        workload_name: None,
        level,
        floorplan: None,
        checks: &[],
    })
}

/// Greedy non-overlapping lanes per resource (Tier A spans of concurrent ops share a binding resource), then
/// sorts spans by `(resource, lane, t_start)` (05 §3.4) and records lane counts on the resources.
fn assign_lanes(t: &mut Trace) {
    t.spans.sort_by_key(|s| (s.resource, s.t_start, s.op));
    let mut ends: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    for s in &mut t.spans {
        let e = ends.entry(s.resource).or_default();
        let lane = e
            .iter()
            .position(|&end| end <= s.t_start)
            .unwrap_or(e.len());
        if lane == e.len() {
            e.push(0);
        }
        e[lane] = s.t_start + s.dur;
        s.lane = lane as u16;
    }
    for (r, e) in ends {
        t.resources[r as usize].lanes = e.len().min(usize::from(u16::MAX)) as u16;
    }
    t.spans
        .sort_by_key(|s| (s.resource, s.lane, s.t_start, s.op));
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;
    use crate::check::check_trace;
    use crate::container::{read_kiln, verify_members, write_kiln};
    use crate::perfetto::{PTrace, export_perfetto};

    #[test]
    fn build_write_read_validate_export() {
        let r = crate::check::tests::result();
        for level in [TraceLevel::Summary, TraceLevel::Ops] {
            let t = build(&BuildInput {
                result: &r,
                hw: None,
                design_json: None,
                design_name: Some("d".into()),
                workload_name: None,
                level,
                floorplan: None,
                checks: &[],
            });
            assert!(!t.resources.is_empty() && !t.ops.is_empty() && !t.phases.is_empty());
            assert_eq!(t.spans.is_empty(), level < TraceLevel::Ops);
            let diags = check_trace(&t);
            assert!(diags.is_empty(), "{diags:?}");
            let bytes = write_kiln(&t);
            assert_eq!(bytes, write_kiln(&t), "writer is deterministic");
            let back = read_kiln(&bytes).unwrap();
            assert!(verify_members(&bytes, &back.manifest).is_empty());
            let mut cmp = back.clone();
            cmp.manifest.tables.clear();
            assert_eq!(cmp, t);
            let p = PTrace::decode(export_perfetto(&back).as_slice()).unwrap();
            let begins = p
                .packet
                .iter()
                .filter(|x| x.track_event.as_ref().is_some_and(|e| e.r#type == Some(1)))
                .count();
            assert_eq!(begins, t.spans.len() + t.phases.len() + t.groups.len());
        }
    }

    #[test]
    fn op_kinds() {
        assert_eq!(op_kind_of("layers.attn", 10), "attention");
        assert_eq!(op_kind_of("layers.attn_norm", 0), "norm");
        assert_eq!(op_kind_of("layers.gate_up", 10), "matmul");
        assert_eq!(op_kind_of("layers.kv_append", 0), "memory");
        assert_eq!(op_kind_of("layers.qkv", 10), "matmul");
        assert_eq!(op_kind_of("layers.act", 0), "elementwise");
    }
}
