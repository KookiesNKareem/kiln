//! `kiln.hw/0` = the Python harness design JSON (`harness/design.py`), migrated by `v0_to_v1` (01 §19).

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::common::Diagnostic;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V0Design {
    name: String,
    clock_mhz: f64,
    tech_node: String,
    compute: Vec<V0Compute>,
    memory: Vec<V0Memory>,
    offchip: V0OffChip,
    #[serde(default)]
    links: Vec<V0Link>,
    #[serde(default)]
    notes: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V0Compute {
    name: String,
    kind: String,
    rows: u32,
    cols: u32,
    #[serde(default = "one")]
    count: u32,
    #[serde(default)]
    attach: String,
    #[serde(default = "bf16")]
    precision: String,
    #[serde(default = "fp32")]
    accumulator: String,
    #[serde(default = "buf_kib")]
    buffer_kib: f64,
    #[serde(default = "buf_gbps")]
    buffer_gbps: f64,
    #[serde(default)]
    regfile_kib: f64,
    #[serde(default)]
    regfile_gbps: f64,
    #[serde(default)]
    ops: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V0Memory {
    name: String,
    size_mib: f64,
    bandwidth_gbps: f64,
    #[serde(default = "one")]
    count: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V0OffChip {
    capacity_gib: f64,
    bandwidth_gbps: f64,
    attach: Vec<String>,
    #[serde(default = "hbm2e")]
    kind: String,
    #[serde(default = "one")]
    stacks: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct V0Link {
    name: String,
    endpoints: Vec<String>,
    bandwidth_gbps: f64,
    #[serde(default = "bus")]
    kind: String,
    #[serde(default = "global")]
    scope: String,
}

fn one() -> u32 {
    1
}
fn bf16() -> String {
    "bf16".into()
}
fn fp32() -> String {
    "fp32".into()
}
fn hbm2e() -> String {
    "HBM2e".into()
}
fn bus() -> String {
    "bus".into()
}
fn global() -> String {
    "global".into()
}
fn buf_kib() -> f64 {
    2048.0
}
fn buf_gbps() -> f64 {
    4000.0
}

fn fail(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error("E-IR-1901", msg).hint("fix the harness design (harness/design.py rules) and re-import")
}

fn ident(s: &str) -> String {
    let mut id: String =
        s.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    if !id.starts_with(|c: char| c.is_ascii_lowercase()) {
        id.insert(0, 'u');
    }
    id
}

/// Ids that end in a digit cannot be replicated (E-IR-0105), so such names get a trailing `_`.
fn rep_ident(s: &str, count: u32) -> String {
    let id = ident(s);
    if count > 1 && id.ends_with(|c: char| c.is_ascii_digit()) { format!("{id}_") } else { id }
}

fn bytes(v: f64, what: &str) -> Result<u64, Diagnostic> {
    if v >= 0.0 && v.fract() == 0.0 { Ok(v as u64) } else { Err(fail(format!("{what} = {v} B is not a whole byte count"))) }
}

fn precision(p: &str) -> &str {
    if p == "fp8" { "fp8_e4m3" } else { p }
}

fn op_class(op: &str) -> Option<&'static str> {
    Some(match op {
        "MatMul" | "Gemm" | "Einsum" => "matmul",
        "Conv" => "conv",
        "Add" | "Sub" | "Mul" | "Div" | "Relu" => "elementwise",
        "Exp" | "Pow" | "Silu" | "Sigmoid" | "Gelu" | "Tanh" => "transcendental",
        "Softmax" | "ReduceMax" | "ReduceSum" | "ReduceMean" | "MaxPool" | "AveragePool" | "GlobalAveragePool"
        | "GlobalMaxPool" => "reduction",
        _ => return None,
    })
}

/// Pure value-tree migration of a harness design to an authoring-form `kiln.hw/1.0` document.
pub fn migrate_v0(v: &Value) -> Result<Value, Diagnostic> {
    let mut v = v.clone();
    if let Value::Object(m) = &mut v {
        m.remove("schema");
    }
    let d: V0Design = serde_json::from_value(v).map_err(|e| fail(format!("not a harness design: {e}")))?;
    let clk = d.clock_mhz * 1e6;
    let bits_per_cycle = |gbps: f64| (gbps * 1e9 * 8.0 / clk).ceil() as u64;
    let tech = match d.tech_node.as_str() {
        "n3" => "tsmc_n3e".to_owned(),
        n => format!("tsmc_{n}"),
    };

    let mems: IndexMap<&str, &V0Memory> = d.memory.iter().map(|m| (m.name.as_str(), m)).collect();
    let units: IndexMap<&str, &V0Compute> = d.compute.iter().map(|c| (c.name.as_str(), c)).collect();
    let partitioned = |m: &str| {
        mems.get(m).is_some_and(|mm| mm.count > 1 && d.compute.iter().any(|c| c.attach == m))
    };
    let part_id = |m: &str| format!("{}_part", ident(m));
    let has_rf = |c: &V0Compute| c.kind == "vector" && c.regfile_kib > 0.0;
    let rf_id = |c: &V0Compute| format!("{}_rf", ident(&c.name));
    let mem_id = |m: &str| if partitioned(m) { ident(m) } else { rep_ident(m, mems[m].count) };
    let mem_sel = |m: &str| if partitioned(m) { format!("{}*.{}", part_id(m), mem_id(m)) } else { mem_id(m) };
    let unit_sels = |c: &V0Compute, local: bool| -> Vec<String> {
        let ids = if has_rf(c) { vec![rf_id(c)] } else { vec![ident(&c.name)] };
        ids.into_iter()
            .map(|id| if partitioned(&c.attach) && !local { format!("{}*.{id}*", part_id(&c.attach)) } else { format!("{id}*") })
            .collect()
    };
    let link_of = |c: &V0Compute| {
        d.links.iter().find(|l| l.endpoints.contains(&c.name) && l.endpoints.contains(&c.attach)).map(|l| ident(&l.name))
    };

    let unit_json = |c: &V0Compute, count: u32| -> Result<(Value, Option<Value>), Diagnostic> {
        if !mems.contains_key(c.attach.as_str()) {
            return Err(fail(format!("compute '{}' attaches to unknown memory {:?}", c.name, c.attach)));
        }
        let id = rep_ident(&c.name, count);
        let p = precision(&c.precision);
        let mut u = match c.kind.as_str() {
            "matrix" => json!({
                "id": id, "kind": "matrix",
                "geometry": { "systolic": { "rows": c.rows, "cols": c.cols } },
                "precisions": [format!("{p}*{p}+{}", precision(&c.accumulator))],
            }),
            "vector" => json!({ "id": id, "kind": "vector", "lanes": c.cols, "sublanes": c.rows, "precisions": [format!("{p}@1")] }),
            k => return Err(fail(format!("compute '{}': unknown kind {k:?}", c.name))),
        };
        if count > 1 {
            u["count"] = json!(count);
        }
        if let Some(ops) = &c.ops {
            let mut classes: Vec<&str> = ops.iter().filter_map(|o| op_class(o)).collect();
            classes.sort_unstable();
            classes.dedup();
            u["ops"] = json!(classes);
        }
        u["local"] = json!([{ "id": "buf", "holds": "any", "capacity": bytes(c.buffer_kib * 1024.0, "buffer_kib")? }]);
        let attach = &mems[c.attach.as_str()];
        let width = bits_per_cycle(c.buffer_gbps).min(bits_per_cycle(attach.bandwidth_gbps));
        let rf = if has_rf(c) {
            let rid = rf_id(c);
            u["feeds"] = json!({ "any": { "from": if count > 1 { format!("{rid}[{{i}}]") } else { rid.clone() } } });
            let mut m = json!({
                "id": rid, "kind": "register_file", "capacity": bytes(c.regfile_kib * 1024.0, "regfile_kib")?,
                "ports": [{ "dir": "rw", "width_bits": bits_per_cycle(c.regfile_gbps) }],
                "overrides": { "bandwidth": c.regfile_gbps * 1e9, "source": "harness" },
            });
            if count > 1 {
                m["count"] = json!(count);
            }
            Some(m)
        } else {
            let mut feed = json!({ "from": mem_id(&c.attach), "width_bits": width });
            if let Some(l) = link_of(c) {
                feed["via"] = json!(l);
            }
            u["feeds"] = json!({ "any": feed });
            None
        };
        Ok((u, rf))
    };

    let mem_json = |m: &V0Memory, count: u32| -> Result<Value, Diagnostic> {
        let mut j = json!({
            "id": rep_ident(&m.name, count), "kind": "scratchpad",
            "capacity": bytes(m.size_mib * 1048576.0, "size_mib")?,
            "ports": [{ "dir": "rw", "width_bits": bits_per_cycle(m.bandwidth_gbps) }],
            "overrides": { "bandwidth": m.bandwidth_gbps * 1e9, "source": "harness" },
        });
        if count > 1 {
            j["count"] = json!(count);
        }
        Ok(j)
    };

    let link_json = |l: &V0Link, local: bool| -> Result<Value, Diagnostic> {
        let mut eps = vec![];
        for e in &l.endpoints {
            if let Some(c) = units.get(e.as_str()) {
                eps.extend(unit_sels(c, local));
            } else if mems.contains_key(e.as_str()) {
                eps.push(if local { mem_id(e) } else { mem_sel(e) });
            } else if e != "offchip" {
                return Err(fail(format!("link '{}': unknown endpoint {e:?}", l.name)));
            }
        }
        let topology = if l.kind == "link" { "p2p" } else { "bus" };
        Ok(json!({
            "id": ident(&l.name), "topology": topology, "endpoints": eps,
            "link": { "width_bits": bits_per_cycle(l.bandwidth_gbps), "bandwidth": l.bandwidth_gbps * 1e9, "source": "harness" },
        }))
    };

    let mut clusters = vec![];
    let mut die_units = vec![];
    let mut die_mems = vec![];
    for m in &d.memory {
        if partitioned(&m.name) {
            let mut cu = vec![];
            let mut cm = vec![mem_json(m, 1)?];
            for c in d.compute.iter().filter(|c| c.attach == m.name) {
                let (u, rf) = unit_json(c, c.count / m.count)?;
                cu.push(u);
                cm.extend(rf);
            }
            let nets = d
                .links
                .iter()
                .filter(|l| l.scope == "per_memory" && l.endpoints.contains(&m.name))
                .map(|l| link_json(l, true))
                .collect::<Result<Vec<_>, _>>()?;
            clusters.push(json!({ "id": part_id(&m.name), "count": m.count, "units": cu, "memories": cm, "networks": nets }));
        } else {
            die_mems.push(mem_json(m, m.count)?);
        }
    }
    for c in d.compute.iter().filter(|c| !partitioned(&c.attach)) {
        let (u, rf) = unit_json(c, c.count)?;
        die_units.push(u);
        die_mems.extend(rf);
    }
    let mut nets = d
        .links
        .iter()
        .filter(|l| l.scope == "global" || !l.endpoints.iter().any(|e| partitioned(e)))
        .map(|l| link_json(l, false))
        .collect::<Result<Vec<_>, _>>()?;
    let o = &d.offchip;
    let offchip_net = match d.links.iter().find(|l| l.endpoints.iter().any(|e| e == "offchip")) {
        Some(l) => ident(&l.name),
        None => {
            let eps: Vec<String> = o.attach.iter().filter(|a| mems.contains_key(a.as_str())).map(|a| mem_sel(a)).collect();
            if eps.len() != o.attach.len() {
                return Err(fail(format!("offchip.attach {:?} names unknown memories", o.attach)));
            }
            nets.push(json!({
                "id": "offchip", "topology": "bus", "endpoints": eps,
                "link": { "width_bits": bits_per_cycle(o.bandwidth_gbps) },
            }));
            "offchip".to_owned()
        }
    };

    let stacks = o.stacks.max(1);
    let kind = match o.kind.to_lowercase().as_str() {
        k @ ("hbm2" | "hbm2e" | "hbm3" | "hbm3e" | "hbm4" | "lpddr5" | "lpddr5x" | "gddr6" | "gddr6x" | "gddr7" | "ddr5") => {
            k.to_owned()
        }
        _ => "custom".to_owned(),
    };
    let io_width: u32 = if kind.starts_with("hbm") { 1024 } else { 64 };
    let per_stack_bw = o.bandwidth_gbps * 1e9 / f64::from(stacks);
    let mut stack = json!({
        "id": "hbm", "kind": kind,
        "capacity": bytes(o.capacity_gib * 1073741824.0 / f64::from(stacks), "offchip.capacity_gib / stacks")?,
        "io_width_bits": io_width,
        "pin_rate_bits_per_s": per_stack_bw * 8.0 / f64::from(io_width),
        "overrides": { "bandwidth": per_stack_bw, "source": "harness" },
        "attach": { "network": format!("die.{offchip_net}") },
    });
    if stacks > 1 {
        stack["count"] = json!(stacks);
    }

    Ok(json!({
        "schema": "kiln.hw/1.0",
        "name": ident(&d.name).replace('_', "-"),
        "meta": {
            "description": d.notes,
            "citations": { "harness": format!("harness/designs ({}): kiln.hw/0 design, migrated by v0_to_v1", d.name) },
        },
        "tech": tech,
        "clocks": [{ "id": "core", "freq": clk }],
        "system": { "package": {
            "id": "chip",
            "dies": [{
                "id": "die", "default_clock": "core",
                "clusters": clusters, "units": die_units, "memories": die_mems, "networks": nets,
            }],
            "mem_stacks": [stack],
        } },
    }))
}
