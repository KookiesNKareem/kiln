//! Persistent cost cache file (03 §2.8): JSON lines, a header naming the model version, then one entry per
//! [`CostKey`] with its last-use tick. Floats are stored as bit patterns so a loaded entry is bit-identical to
//! the computed one. Saving merges with entries other processes wrote meanwhile, keeps the most recently used
//! within a byte budget and replaces the file atomically.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

use kiln_ir::common::Diagnostic;
use serde_json::{Value, json};

use crate::{CostEntry, CostKey};

/// Version of the cost model: a hash of the sources that decide search results, fixed at build time.
pub const MODEL_HASH: &str = env!("KILN_COST_MODEL_HASH");

const FORMAT: u64 = 1;

pub(crate) type Loaded = (CostKey, Result<CostEntry, Diagnostic>, u64);

/// `f64` numbers become `"#<16 hex digits>"` strings (bits), everything else is kept.
fn encode(v: Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => Value::String(format!("#{:016x}", n.as_f64().unwrap_or(0.0).to_bits())),
        Value::Array(a) => Value::Array(a.into_iter().map(encode).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, encode(v))).collect()),
        v => v,
    }
}

fn decode(v: Value) -> Value {
    match v {
        Value::String(s) if s.len() == 17 && s.starts_with('#') => match u64::from_str_radix(&s[1..], 16) {
            Ok(bits) => serde_json::Number::from_f64(f64::from_bits(bits)).map_or(Value::Null, Value::Number),
            Err(_) => Value::String(s),
        },
        Value::Array(a) => Value::Array(a.into_iter().map(decode).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, decode(v))).collect()),
        v => v,
    }
}

pub(crate) fn line(key: &CostKey, slot: &Result<std::sync::Arc<CostEntry>, Diagnostic>, tick: u64) -> String {
    let v = match slot {
        Ok(e) => json!({ "ok": encode(serde_json::to_value(&**e).unwrap_or(Value::Null)) }),
        Err(d) => json!({ "err": d }),
    };
    json!({ "t": tick, "k": [key.unit_template, key.shape, key.objective, key.options], "v": v }).to_string()
}

fn parse(l: &str) -> Option<Loaded> {
    let mut v: Value = serde_json::from_str(l).ok()?;
    let k = v.get_mut("k")?.take();
    let (t, s, o, opts): (String, String, _, _) = serde_json::from_value(k).ok()?;
    let key = CostKey { unit_template: t, shape: s, objective: o, options: opts };
    let tick = v.get("t")?.as_u64()?;
    let body = v.get_mut("v")?;
    let r = if let Some(e) = body.get_mut("ok") {
        Ok(serde_json::from_value(decode(e.take())).ok()?)
    } else {
        Err(serde_json::from_value(body.get_mut("err")?.take()).ok()?)
    };
    Some((key, r, tick))
}

/// Entries of the cache file at `path`; nothing when it is missing, unreadable or of another model version.
pub(crate) fn read(path: &Path) -> Vec<Loaded> {
    read_lines(path).into_iter().filter_map(|(_, l, _)| parse(&l)).collect()
}

/// Key, tick and raw line of every entry of the file at `path` (same validity rules as [`read`]).
fn read_lines(path: &Path) -> Vec<(CostKey, String, u64)> {
    let Ok(f) = std::fs::File::open(path) else { return vec![] };
    let mut lines = std::io::BufReader::new(f).lines();
    let header: Option<Value> = lines.next().and_then(Result::ok).and_then(|h| serde_json::from_str(&h).ok());
    if !header.is_some_and(|h| h["format"].as_u64() == Some(FORMAT) && h["model"].as_str() == Some(MODEL_HASH)) {
        return vec![];
    }
    lines
        .map_while(Result::ok)
        .filter_map(|l| {
            let mut v: Value = serde_json::from_str(&l).ok()?;
            let (t, s, o, opts): (String, String, _, _) = serde_json::from_value(v.get_mut("k")?.take()).ok()?;
            let tick = v.get("t")?.as_u64()?;
            Some((CostKey { unit_template: t, shape: s, objective: o, options: opts }, l, tick))
        })
        .collect()
}

/// Writes `entries` (key, line, tick) merged with the file's current entries that `entries` lacks, most recent
/// first until `max_bytes`; atomic replace.
pub(crate) fn write(path: &Path, entries: Vec<(CostKey, String, u64)>, max_bytes: u64) -> std::io::Result<()> {
    let mut all: BTreeMap<CostKey, (String, u64)> = BTreeMap::new();
    for (k, l, t) in read_lines(path).into_iter().chain(entries) {
        all.insert(k, (l, t));
    }
    let mut v: Vec<(CostKey, (String, u64))> = all.into_iter().collect();
    v.sort_by(|a, b| b.1.1.cmp(&a.1.1).then(a.0.cmp(&b.0)));
    let header = json!({ "format": FORMAT, "model": MODEL_HASH }).to_string();
    let mut used = header.len() as u64 + 1;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        writeln!(f, "{header}")?;
        for (_, (l, _)) in v {
            used += l.len() as u64 + 1;
            if used > max_bytes {
                break;
            }
            writeln!(f, "{l}")?;
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}
