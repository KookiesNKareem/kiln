//! T2 design transformations (06 §12.2) as edits of a design's canonical JSON (01 §17).

use serde_json::{Map, Value, json};

pub type Edit = Result<Value, String>;

const ENTITY_ARRAYS: [&str; 10] =
    ["boards", "packages", "dies", "clusters", "units", "memories", "blocks", "networks", "ports", "mem_stacks"];

/// Visits every object reachable from `v`, depth first, children in declaration order.
pub fn visit_mut(v: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    match v {
        Value::Object(o) => {
            f(o);
            for c in o.values_mut() {
                visit_mut(c, f);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|c| visit_mut(c, f)),
        _ => {}
    }
}

fn scale_num(v: &mut Value, k: f64) -> Result<(), String> {
    let x = v.as_f64().ok_or_else(|| format!("expected a number, got {v}"))?;
    *v = json!(x * k);
    Ok(())
}

fn count(o: &Map<String, Value>) -> u64 {
    o.get("count").and_then(Value::as_u64).unwrap_or(1)
}

/// M1: scales every off-chip stack's pin rate (and the stack clock, if one is named) by `k`.
pub fn scale_offchip_bandwidth(design: &Value, k: f64) -> Edit {
    let mut d = design.clone();
    let mut clocks = vec![];
    let mut err = None;
    visit_mut(&mut d, &mut |o| {
        if let Some(Value::Array(stacks)) = o.get_mut("mem_stacks") {
            for s in stacks.iter_mut().filter_map(Value::as_object_mut) {
                if let Some(c) = s.get("clock").and_then(Value::as_str) {
                    clocks.push(c.to_owned());
                }
                match s.get_mut("pin_rate_bits_per_s") {
                    Some(p) => err = err.take().or(scale_num(p, k).err()),
                    None => err = err.take().or(Some(format!("stack {:?} has no pin rate", s.get("id")))),
                }
            }
        }
    });
    if clocks.is_empty() && err.is_none() && !has_stacks(&d) {
        return Err("design has no off-chip stacks".into());
    }
    visit_mut(&mut d, &mut |o| {
        if let Some(Value::Array(cs)) = o.get_mut("clocks") {
            for c in cs.iter_mut().filter_map(Value::as_object_mut) {
                if c.get("id").and_then(Value::as_str).is_some_and(|id| clocks.iter().any(|x| x == id))
                    && let Some(f) = c.get_mut("freq")
                {
                    err = err.take().or(scale_num(f, k).err());
                }
            }
        }
    });
    err.map_or(Ok(d), Err)
}

fn has_stacks(d: &Value) -> bool {
    let mut found = false;
    let mut d = d.clone();
    visit_mut(&mut d, &mut |o| found |= o.get("mem_stacks").and_then(Value::as_array).is_some_and(|a| !a.is_empty()));
    found
}

/// M2/M3: multiplies the replication count of every compute unit whose kind is in `kinds`.
pub fn scale_unit_count(design: &Value, kinds: &[&str], k: u64) -> Edit {
    let mut d = design.clone();
    let mut n = 0;
    visit_mut(&mut d, &mut |o| {
        if let Some(Value::Array(units)) = o.get_mut("units") {
            for u in units.iter_mut().filter_map(Value::as_object_mut) {
                if u.get("kind").and_then(Value::as_str).is_some_and(|x| kinds.contains(&x)) {
                    let c = count(u) * k;
                    u.insert("count".into(), json!(c));
                    n += 1;
                }
            }
        }
    });
    if n == 0 { Err(format!("no units of kind {kinds:?}")) } else { Ok(d) }
}

/// M4: moves every placed die or chiplet halfway towards the package origin, halving inter-die wire spans.
pub fn halve_placement_offsets(design: &Value) -> Edit {
    let mut d = design.clone();
    let mut n = 0;
    visit_mut(&mut d, &mut |o| {
        if let Some(Value::Object(p)) = o.get_mut("placement") {
            for key in ["x", "y"] {
                if let Some(v) = p.get_mut(key).filter(|v| v.is_number()) {
                    let x = v.as_f64().unwrap_or(0.0);
                    *v = json!(x * 0.5);
                    n += 1;
                }
            }
        }
    });
    if n == 0 { Err("no fixed placements to move".into()) } else { Ok(d) }
}

pub const IDLE_UNIT: &str = "trust_idle";

/// M5: adds an fp64-only vector unit (no LLM op can use it) beside the first unit, fed from that unit's feeds.
pub fn add_idle_unit(design: &Value) -> Edit {
    let mut d = design.clone();
    let mut done = false;
    visit_mut(&mut d, &mut |o| {
        if done {
            return;
        }
        let Some(Value::Array(units)) = o.get_mut("units") else { return };
        let Some(feeds) = units.first().and_then(|u| u.get("feeds")).and_then(Value::as_object) else { return };
        let Some(from) = feeds.values().find_map(|f| f.get("from").cloned()) else { return };
        units.push(json!({
            "id": IDLE_UNIT, "kind": "vector", "lanes": 16, "precisions": ["fp64@1"],
            "feeds": { "any": { "from": from } },
        }));
        done = true;
    });
    if done { Ok(d) } else { Err("no unit with feeds to attach an idle unit to".into()) }
}

/// M6a: reverses every entity array and the citation map; order carries no meaning (01 §17).
pub fn permute(design: &Value) -> Value {
    let mut d = design.clone();
    visit_mut(&mut d, &mut |o| {
        for k in ENTITY_ARRAYS {
            if let Some(Value::Array(a)) = o.get_mut(k) {
                a.reverse();
            }
        }
    });
    if let Some(Value::Object(c)) = d.pointer_mut("/meta/citations") {
        let rev: Map<String, Value> = std::mem::take(c).into_iter().rev().collect();
        *c = rev;
    }
    d
}

const NON_REFERENCE_KEYS: [&str; 17] = [
    "kind", "type", "protocol", "dataflow", "policy", "edge", "harvest_power", "role", "precisions", "functions",
    "ops", "dir", "accumulate_in", "holds", "implementation", "mode", "for_kind",
];

fn id_chars(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// `tok` is `id` or a replicated instance name of it (`id3`, `id0_1`, `id0_`).
fn instance_of<'a>(tok: &'a str, id: &str) -> Option<&'a str> {
    let rest = tok.strip_prefix(id)?;
    let ok = rest.is_empty()
        || (rest.starts_with(|c: char| c.is_ascii_digit()) && rest.chars().all(|c| c.is_ascii_digit() || c == '_'));
    ok.then_some(rest)
}

fn rename_str(s: &str, renames: &[(String, String)]) -> String {
    let mut out = String::with_capacity(s.len());
    let mut tok = String::new();
    let flush = |tok: &mut String, out: &mut String| {
        let longest = renames.iter().filter_map(|(a, b)| instance_of(tok, a).map(|r| (a.len(), b, r))).max_by_key(|x| x.0);
        match longest {
            Some((_, b, r)) => {
                out.push_str(b);
                out.push_str(r);
            }
            None => out.push_str(tok),
        }
        tok.clear();
    };
    for c in s.chars() {
        if id_chars(c) {
            tok.push(c);
        } else {
            flush(&mut tok, &mut out);
            out.push(c);
        }
    }
    flush(&mut tok, &mut out);
    out
}

fn rename_refs(v: &mut Value, renames: &[(String, String)]) {
    match v {
        Value::String(s) => *s = rename_str(s, renames),
        Value::Array(a) => a.iter_mut().for_each(|c| rename_refs(c, renames)),
        Value::Object(o) => {
            for (k, c) in o.iter_mut() {
                if c.is_object() || !NON_REFERENCE_KEYS.contains(&k.as_str()) {
                    rename_refs(c, renames);
                }
            }
        }
        _ => {}
    }
}

/// Every entity and clock id below `system` plus top-level clocks and power domains, in visit order.
pub fn entity_ids(design: &Value) -> Vec<String> {
    let mut ids = vec![];
    let mut d = design.clone();
    for key in ["clocks", "power", "system"] {
        if let Some(v) = d.get_mut(key) {
            visit_mut(v, &mut |o| {
                if let Some(id) = o.get("id").and_then(Value::as_str)
                    && !ids.iter().any(|x| x == id)
                {
                    ids.push(id.to_owned());
                }
            });
        }
    }
    ids
}

/// M6b: renames every id (prefix `prefix`) and rewrites every reference to it, including claim scopes.
pub fn rename_all(design: &Value, prefix: &str) -> Value {
    let renames: Vec<(String, String)> = entity_ids(design).into_iter().map(|id| (id.clone(), format!("{prefix}{id}"))).collect();
    let mut d = design.clone();
    for key in ["clocks", "power", "system"] {
        if let Some(v) = d.get_mut(key) {
            rename_refs(v, &renames);
        }
    }
    if let Some(Value::Array(claims)) = d.pointer_mut("/meta/claims") {
        for c in claims.iter_mut() {
            if let Some(s) = c.get_mut("scope") {
                rename_refs(s, &renames);
            }
        }
    }
    d
}

/// M7: replaces memory `id` by twice as many instances with half of every capacity (incl. carveouts and pinnable
/// bytes) and half of each port's width, and widens exact references (`id` -> `id*`) so every consumer reaches both.
/// `id` must not end in a digit (01 E-IR-0105: replicated instance names append the index).
pub fn split_memory(design: &Value, id: &str) -> Edit {
    let mut d = design.clone();
    let mut n = 0;
    let mut err = None;
    visit_mut(&mut d, &mut |o| {
        let Some(Value::Array(mems)) = o.get_mut("memories") else { return };
        for m in mems.iter_mut().filter_map(Value::as_object_mut) {
            if m.get("id").and_then(Value::as_str) != Some(id) {
                continue;
            }
            let c = count(m) * 2;
            m.insert("count".into(), json!(c));
            for p in ["/capacity", "/word_bits", "/cache/pinnable"] {
                if let Some(v) = m_ptr(m, p) {
                    err = err.take().or(scale_num(v, 0.5).err());
                }
            }
            if let Some(Value::Array(opts)) = m_ptr(m, "/operands/options") {
                for o in opts.iter_mut().filter_map(Value::as_object_mut) {
                    for v in o.values_mut().filter(|v| v.is_number()) {
                        err = err.take().or(scale_num(v, 0.5).err());
                    }
                }
            }
            if let Some(Value::Array(ports)) = m.get_mut("ports") {
                for p in ports.iter_mut() {
                    if let Some(w) = p.get_mut("width_bits") {
                        err = err.take().or(scale_num(w, 0.5).err());
                    }
                }
            }
            n += 1;
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    if n == 0 {
        return Err(format!("no memory {id:?}"));
    }
    integerize(&mut d);
    visit_mut(&mut d, &mut |o| {
        let joins = o
            .get("endpoints")
            .and_then(Value::as_array)
            .is_some_and(|e| e.iter().any(|x| x.get("select").unwrap_or(x).as_str() == Some(id)));
        let topo = o.get("topology").map(|t| t.get("type").unwrap_or(t));
        if joins && topo.and_then(Value::as_str) == Some("p2p") {
            o.insert("topology".into(), json!({ "type": "bus" }));
        }
    });
    if let Some(sys) = d.get_mut("system") {
        widen_refs(sys, id);
    }
    Ok(d)
}

fn m_ptr<'a>(m: &'a mut Map<String, Value>, p: &str) -> Option<&'a mut Value> {
    let (head, rest) = p[1..].split_once('/').map_or((&p[1..], ""), |(h, r)| (h, r));
    let v = m.get_mut(head)?;
    if rest.is_empty() { Some(v) } else { v.pointer_mut(&format!("/{rest}")) }
}

fn integerize(v: &mut Value) {
    match v {
        Value::Number(n) => {
            if let Some(f) = n.as_f64().filter(|f| f.fract() == 0.0 && n.is_f64() && f.abs() < 9e15) {
                *v = json!(f as i64);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(integerize),
        Value::Object(o) => o.values_mut().for_each(integerize),
        _ => {}
    }
}

fn widen_refs(v: &mut Value, id: &str) {
    match v {
        Value::String(s) => {
            let segs: Vec<String> =
                s.split('.').map(|seg| if seg == id { format!("{id}*") } else { seg.to_owned() }).collect();
            *s = segs.join(".");
        }
        Value::Array(a) => a.iter_mut().for_each(|c| widen_refs(c, id)),
        Value::Object(o) => {
            for (k, c) in o.iter_mut() {
                if c.is_object() || (k != "id" && !NON_REFERENCE_KEYS.contains(&k.as_str())) {
                    widen_refs(c, id);
                }
            }
        }
        _ => {}
    }
}
