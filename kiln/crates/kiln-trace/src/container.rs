//! `.kiln` container (05 §3.5, §3.6): 64-byte header, 64-byte-aligned Arrow IPC file members (one per table),
//! JSON manifest written last, 64-byte trailer. A manifest-only container is valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema};
use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};

use sha2::{Digest, Sha256};

use crate::arrowx;
use crate::interval::Interval;
use crate::provenance::{Provenance, Tier, TraceLevel};
use crate::trace::Trace;

pub const MAGIC: [u8; 8] = *b"KILNTRC\0";
pub const CONTAINER_VERSION: u32 = 1;
pub const TRACE_SCHEMA_VERSION: &str = "1.1";
pub const BLOCK: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub schema_version: String,
    pub tick_s: f64,
    pub level: TraceLevel,
    pub tier: Tier,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub design: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<serde_json::Value>,
    /// Enum string tables, keyed `<table>.<column>`; codes index into the list.
    #[serde(default)]
    pub enums: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub tables: Vec<TableEntry>,
    #[serde(default)]
    pub headline: Headline,
    #[serde(default)]
    pub thumbnails: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Where `floorplan` came from: a kiln-phys placer id, or `unplaced` (hierarchy layout, not physical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floorplan_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub design_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload_name: Option<String>,
}

impl Manifest {
    pub fn new(level: TraceLevel, provenance: Provenance) -> Self {
        Self {
            format: "kiln-trace".into(),
            schema_version: TRACE_SCHEMA_VERSION.into(),
            tick_s: 1e-12,
            level,
            tier: provenance.tier,
            provenance,
            design: None,
            workload: None,
            enums: BTreeMap::new(),
            tables: Vec::new(),
            headline: Headline::default(),
            thumbnails: Vec::new(),
            notes: Vec::new(),
            floorplan_source: None,
            design_name: None,
            workload_name: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableEntry {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub rows: u64,
    #[serde(default)]
    pub compression: Option<String>,
    pub sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Headline {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_s: Option<Interval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_s: Option<Interval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_j: Option<Interval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area_mm2: Option<Interval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_w: Option<Interval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<Interval>,
    /// Failed floor/invariant codes; empty when all passed.
    #[serde(default)]
    pub floor_failures: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub flags: u32,
    pub manifest_offset: u64,
    pub manifest_len: u64,
}

impl Header {
    pub fn encode(&self) -> [u8; BLOCK] {
        let mut b = [0u8; BLOCK];
        b[..8].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&self.version.to_le_bytes());
        b[12..16].copy_from_slice(&self.flags.to_le_bytes());
        b[16..24].copy_from_slice(&self.manifest_offset.to_le_bytes());
        b[24..32].copy_from_slice(&self.manifest_len.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self, Diagnostic> {
        if b.len() < BLOCK || b[..8] != MAGIC {
            return Err(Diagnostic::error(
                "E-TRACE-CONTAINER",
                "not a .kiln container (bad magic)",
            ));
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
        Ok(Self {
            version: u32_at(8),
            flags: u32_at(12),
            manifest_offset: u64_at(16),
            manifest_len: u64_at(24),
        })
    }
}

/// Writes a container with no table members: header, manifest JSON, trailer.
pub fn write_manifest_only(manifest: &Manifest) -> Vec<u8> {
    let json = serde_json::to_vec(manifest).expect("manifest serializes");
    let header = Header {
        version: CONTAINER_VERSION,
        flags: 0,
        manifest_offset: BLOCK as u64,
        manifest_len: json.len() as u64,
    };
    let mut out = header.encode().to_vec();
    out.extend_from_slice(&json);
    out.resize(out.len().next_multiple_of(BLOCK), 0);
    let mut trailer = header.encode();
    trailer[8..16].fill(0);
    out.extend_from_slice(&trailer);
    out
}

/// Reads the manifest via the trailer (written last, 05 §3.6) and cross-checks the header.
pub fn read_manifest(bytes: &[u8]) -> Result<Manifest, Diagnostic> {
    let header = Header::decode(bytes)?;
    if header.version != CONTAINER_VERSION {
        return Err(Diagnostic::error(
            "E-TRACE-CONTAINER",
            format!("unsupported container version {}", header.version),
        ));
    }
    let trailer = Header::decode(&bytes[bytes.len().saturating_sub(BLOCK)..]).map_err(|_| {
        Diagnostic::error("E-TRACE-CONTAINER", "missing trailer").hint("run `kiln trace recover`")
    })?;
    let at = |h: &Header| (h.manifest_offset, h.manifest_len);
    if at(&header) != at(&trailer) {
        return Err(Diagnostic::error(
            "E-TRACE-CONTAINER",
            format!(
                "header manifest range {:?} != trailer {:?}",
                at(&header),
                at(&trailer)
            ),
        )
        .hint("run `kiln trace recover`"));
    }
    let (off, len) = (
        trailer.manifest_offset as usize,
        trailer.manifest_len as usize,
    );
    let slice = bytes
        .get(off..off.saturating_add(len))
        .filter(|_| off >= BLOCK && off.saturating_add(len) <= bytes.len() - BLOCK)
        .ok_or_else(|| {
            Diagnostic::error(
                "E-TRACE-CONTAINER",
                "manifest range outside the space between header and trailer",
            )
        })?;
    let m: Manifest = serde_json::from_slice(slice).map_err(|e| {
        Diagnostic::error(
            "E-TRACE-CONTAINER",
            format!("manifest is not valid JSON: {e}"),
        )
    })?;
    if m.format != "kiln-trace" {
        return Err(Diagnostic::error(
            "E-TRACE-CONTAINER",
            format!("manifest format {:?}", m.format),
        ));
    }
    let mut names = std::collections::BTreeSet::new();
    if let Some(e) = m.tables.iter().find(|e| !names.insert(e.name.as_str())) {
        return Err(Diagnostic::error(
            "E-TRACE-CONTAINER",
            format!("table {:?} has more than one member", e.name),
        ));
    }
    Ok(m)
}

/// Writes a full container: header, one Arrow IPC file member per table (64-byte aligned), manifest, trailer.
/// Byte-identical for identical traces.
pub fn write_kiln(t: &Trace) -> Vec<u8> {
    let mut out = vec![0u8; BLOCK];
    let mut m = t.manifest.clone();
    m.tables.clear();
    for (name, batch) in t.batches() {
        if name == "spans" && m.level < TraceLevel::Ops && t.spans.is_empty() {
            continue;
        }
        let bytes = arrowx::to_ipc_file(&batch);
        let offset = out.len() as u64;
        m.tables.push(TableEntry {
            name: name.into(),
            offset,
            length: bytes.len() as u64,
            rows: batch.num_rows() as u64,
            compression: None,
            sha256: hex::encode(Sha256::digest(&bytes)),
        });
        out.extend_from_slice(&bytes);
        out.resize(out.len().next_multiple_of(BLOCK), 0);
    }
    let json = serde_json::to_vec(&m).expect("manifest serializes");
    let header = Header {
        version: CONTAINER_VERSION,
        flags: 0,
        manifest_offset: out.len() as u64,
        manifest_len: json.len() as u64,
    };
    out[..BLOCK].copy_from_slice(&header.encode());
    out.extend_from_slice(&json);
    out.resize(out.len().next_multiple_of(BLOCK), 0);
    let mut trailer = header.encode();
    trailer[8..16].fill(0);
    out.extend_from_slice(&trailer);
    out
}

/// Reads every known table; unknown tables are skipped (05 §3.10).
pub fn read_kiln(bytes: &[u8]) -> Result<Trace, Diagnostic> {
    let manifest = read_manifest(bytes)?;
    let entries = manifest.tables.clone();
    let mut t = Trace::empty(manifest);
    for e in &entries {
        let member = member(bytes, e)?;
        let (batches, partial) =
            arrowx::read_ipc(member).map_err(|d| d.at(format!("table {}", e.name)))?;
        if partial {
            return Err(Diagnostic::error(
                "E-TRACE-CONTAINER",
                format!("table {} is truncated", e.name),
            )
            .hint("run `kiln trace recover`"));
        }
        t.set_table(&e.name, &batches)?;
    }
    Ok(t)
}

pub fn read_kiln_file(path: &std::path::Path) -> Result<Trace, Diagnostic> {
    let bytes = std::fs::read(path).map_err(|e| {
        Diagnostic::error(
            "E-TRACE-CONTAINER",
            format!("cannot read {}: {e}", path.display()),
        )
    })?;
    read_kiln(&bytes).map_err(|d| {
        if d.path.is_none() {
            d.at(path.display().to_string())
        } else {
            d
        }
    })
}

fn member<'a>(bytes: &'a [u8], e: &TableEntry) -> Result<&'a [u8], Diagnostic> {
    let range = usize::try_from(e.offset).ok().zip(
        e.offset
            .checked_add(e.length)
            .and_then(|end| usize::try_from(end).ok()),
    );
    range
        .and_then(|(start, end)| bytes.get(start..end))
        .ok_or_else(|| {
            Diagnostic::error(
                "E-TRACE-CONTAINER",
                format!("table {} lies outside the file", e.name),
            )
        })
}

/// Member checksums against the manifest's table directory.
pub fn verify_members(bytes: &[u8], m: &Manifest) -> Vec<Diagnostic> {
    m.tables
        .iter()
        .filter_map(|e| match member(bytes, e) {
            Err(d) => Some(d),
            Ok(b) if hex::encode(Sha256::digest(b)) != e.sha256 => Some(Diagnostic::error(
                "E-TRACE-CONTAINER",
                format!("table {} fails its sha256", e.name),
            )),
            Ok(_) => None,
        })
        .collect()
}

/// Tables written at each level (05 §3.3); a level includes everything below it.
pub fn tables_for(level: TraceLevel) -> Vec<&'static str> {
    const SUMMARY: &[&str] = &[
        "resources",
        "ops",
        "phases",
        "ceilings",
        "run_scalars",
        "aggregates_resource",
        "aggregates_op_resource",
        "limiters",
        "bottleneck",
        "groups",
        "collectives",
        "mapping",
        "floorplan",
        "wires",
        "diagnostics",
    ];
    const FULL: &[&str] = &[
        "transfers",
        "routes",
        "critical_path",
        "thermal_frames",
        "counters",
    ];
    match level {
        TraceLevel::None => vec![],
        TraceLevel::Summary => SUMMARY.to_vec(),
        TraceLevel::Ops => [SUMMARY, &["spans"]].concat(),
        TraceLevel::Full => [SUMMARY, &["spans"], FULL].concat(),
    }
}

fn f(name: &str, dt: DataType, nullable: bool) -> Field {
    Field::new(name, dt, nullable)
}

fn list(dt: DataType) -> DataType {
    DataType::List(Arc::new(Field::new("item", dt, true)))
}

fn xy() -> DataType {
    DataType::Struct(Fields::from(vec![
        f("x", DataType::Float64, false),
        f("y", DataType::Float64, false),
    ]))
}

/// Arrow schema of a 05 §3.4 table, with `kiln.table` / `kiln.schema_version` metadata.
pub fn table_schema(name: &str) -> Option<Schema> {
    use DataType::*;
    if let Some(s) = crate::trace::implemented_schema(name) {
        return Some(s);
    }
    let dict = || Dictionary(Box::new(UInt32), Box::new(Utf8));
    let fields = match name {
        "resources" => vec![
            f("idx", UInt32, false),
            f("path", dict(), false),
            f("kind", UInt16, false),
            f("class", UInt8, false),
            f("parent", UInt32, true),
            f("chip", UInt32, false),
            f("array_id", UInt32, true),
            f("array_pos", UInt32, true),
            f("mem_level", UInt8, true),
            f("capacity_b", Float64, true),
            f("peak_bw_bps", Float64, true),
            f(
                "peak_flops",
                Map(
                    Arc::new(f(
                        "entries",
                        Struct(Fields::from(vec![
                            f("key", UInt8, false),
                            f("value", Float64, true),
                        ])),
                        false,
                    )),
                    false,
                ),
                true,
            ),
            f("lanes", UInt16, false),
        ],
        "ops" => vec![
            f("idx", UInt32, false),
            f("path", Utf8, false),
            f("kind", UInt16, false),
            f("phase", UInt8, false),
            f("layer", Int32, true),
            f("flops", Float64, false),
            f("precision", UInt8, false),
            f("bytes_by_level", list(Float64), false),
            f("link_bytes", Float64, false),
            f("t_start", Int64, false),
            f("t_end", Int64, false),
            f("chips", list(UInt32), false),
            f("mapping", UInt32, false),
            f("macs_useful", UInt64, false),
            f("macs_issued", UInt64, false),
            f("target", UInt8, false),
            f(
                "host_vs_nmp_s",
                FixedSizeList(Arc::new(Field::new("item", Float64, false)), 2),
                true,
            ),
            f("group", UInt32, false),
        ],
        "spans" => vec![
            f("resource", UInt32, false),
            f("lane", UInt16, false),
            f("kind", UInt8, false),
            f("flags", UInt8, false),
            f("op", UInt32, false),
            f("task", UInt32, false),
            f("slice", UInt32, false),
            f("t_start", Int64, false),
            f("dur", Int64, false),
            f("bytes", Float64, false),
            f("energy_j", Float32, false),
        ],
        "transfers" => vec![
            f("id", UInt64, false),
            f("op", UInt32, false),
            f("src", UInt32, false),
            f("dst", UInt32, false),
            f("route", UInt32, false),
            f("bytes", Float64, false),
            f("t_start", Int64, false),
            f("dur", Int64, false),
            f("collective", UInt32, true),
            f("step", UInt16, true),
        ],
        "routes" => vec![
            f("idx", UInt32, false),
            f("links", list(UInt32), false),
            f("wire_len_um", Float64, false),
        ],
        "counters" => vec![
            f("metric", UInt8, false),
            f("resource", UInt32, false),
            f("t", Int64, false),
            f("value", Float64, false),
        ],
        "thermal_frames" => vec![
            f("die", UInt32, false),
            f("t", Int64, false),
            f("nx", UInt16, false),
            f("ny", UInt16, false),
            f("origin_um", xy(), false),
            f("cell_um", Float64, false),
            f("temp_k", list(Float32), false),
        ],
        "aggregates_resource" => vec![
            f("resource", UInt32, false),
            f("busy_s", Float64, false),
            f("stall_s", Float64, false),
            f("bytes", Float64, false),
            f("flops", Float64, false),
            f("energy_dyn_j", Float64, false),
            f("energy_leak_j", Float64, false),
            f("avg_power_w", Float64, false),
            f("peak_power_w", Float64, false),
            f("power_density_w_mm2", Float64, true),
            f("peak_temp_k", Float64, true),
        ],
        "aggregates_op_resource" => vec![
            f("op", UInt32, false),
            f("resource", UInt32, false),
            f("time_s", Float64, false),
            f("energy_component", UInt8, false),
            f("energy_j", Float64, false),
        ],
        "limiters" => vec![
            f("op", UInt32, false),
            f("group", UInt32, false),
            f("binding", UInt8, false),
            f("resource", UInt32, true),
            f("time_s", Float64, false),
            f("attained_frac", Float64, false),
            f("rank", UInt8, false),
            f("share", Float64, false),
        ],
        "bottleneck" => vec![
            f("section", UInt8, false),
            f("binding", UInt8, true),
            f("resource", UInt32, true),
            f("time_s", Float64, true),
            f("utilization", Float64, true),
            f("shadow_price", Float64, true),
            f("slack", Float64, true),
        ],
        "critical_path" => vec![
            f("task", UInt32, false),
            f("span_row", UInt64, true),
            f("transfer", UInt64, true),
            f("reason", UInt8, false),
        ],
        "groups" => vec![
            f("group", UInt32, false),
            f("ops", list(UInt32), false),
            f("kind", UInt8, false),
            f("t_start", Int64, false),
            f("t_end", Int64, false),
            f("bubble_s", Float64, false),
            f("exposed_overhead_s", Float64, false),
        ],
        "collectives" => vec![
            f("collective", UInt32, false),
            f("op", UInt32, false),
            f("algorithm", Utf8, false),
            f("group_chips", list(UInt32), false),
            f("steps", UInt16, false),
            f("bytes", Float64, false),
            f("t_start", Int64, false),
            f("t_end", Int64, false),
            f("link_bytes_by_tier", list(Float64), false),
        ],
        "mapping" => vec![
            f("op", UInt32, false),
            f("tiles", list(UInt32), false),
            f("chips", list(UInt32), false),
            f("parallelism", Utf8, false),
            f("loop_nest", Utf8, false),
            f("tiling_by_level", list(Utf8), false),
        ],
        "floorplan" => vec![
            f("resource", UInt32, false),
            f("die", UInt32, false),
            f("layer", UInt8, false),
            f("x_um", Float64, false),
            f("y_um", Float64, false),
            f("w_um", Float64, false),
            f("h_um", Float64, false),
            f("poly", list(xy()), true),
            f("rotation", UInt8, false),
        ],
        "wires" => vec![
            f("link", UInt32, false),
            f("polyline", list(xy()), false),
            f("layer", UInt8, false),
            f("length_um", Float64, false),
        ],
        "diagnostics" => vec![
            f("code", Utf8, false),
            f("severity", UInt8, false),
            f("message", Utf8, false),
            f("path", Utf8, true),
            f("hint", Utf8, true),
        ],
        _ => return None,
    };
    Some(Schema::new(fields).with_metadata([
        ("kiln.table", name),
        ("kiln.schema_version", TRACE_SCHEMA_VERSION),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::tests::provenance;
    use proptest::prelude::*;

    #[test]
    fn every_level_table_has_a_schema() {
        for t in tables_for(TraceLevel::Full) {
            let s = table_schema(t).unwrap_or_else(|| panic!("no schema for {t}"));
            assert_eq!(s.metadata()["kiln.table"], t);
        }
        assert!(tables_for(TraceLevel::Summary).len() < tables_for(TraceLevel::Ops).len());
        assert!(table_schema("nope").is_none());
    }

    #[test]
    fn manifest_only_container_round_trip() {
        let mut m = Manifest::new(TraceLevel::Summary, provenance());
        m.headline.latency_s = Some(Interval::new(0.01, 0.012, 0.015).unwrap());
        let bytes = write_manifest_only(&m);
        assert_eq!(bytes.len() % BLOCK, 0);
        assert_eq!(read_manifest(&bytes).unwrap(), m);
        assert_eq!(
            read_manifest(&bytes[..BLOCK]).unwrap_err().code,
            "E-TRACE-CONTAINER"
        );
    }

    #[test]
    fn duplicate_table_members_are_rejected() {
        let t = crate::trace::Trace::empty(Manifest::new(TraceLevel::Ops, provenance()));
        let bytes = write_kiln(&t);
        let mut m = read_manifest(&bytes).unwrap();
        let ops = m.tables.iter().find(|e| e.name == "ops").unwrap().clone();
        m.tables.push(ops);
        let dup = write_manifest_only(&m);
        assert_eq!(read_manifest(&dup).unwrap_err().code, "E-TRACE-CONTAINER");
    }

    #[test]
    fn header_and_trailer_must_agree() {
        let bytes = write_manifest_only(&Manifest::new(TraceLevel::Summary, provenance()));
        let mut zeroed = bytes.clone();
        zeroed[16..32].fill(0);
        assert_eq!(
            read_manifest(&zeroed).unwrap_err().code,
            "E-TRACE-CONTAINER"
        );
        let n = bytes.len();
        let mut overlapping = bytes.clone();
        for at in [16, n - BLOCK + 16] {
            overlapping[at..at + 8].fill(0);
        }
        assert!(read_manifest(&overlapping).is_err());
    }

    #[test]
    fn member_range_overflow_is_an_error() {
        let mut m = Manifest::new(TraceLevel::Summary, provenance());
        for (offset, length) in [(u64::MAX, 1), (1, u64::MAX)] {
            m.tables = vec![TableEntry {
                name: "ops".into(),
                offset,
                length,
                rows: 0,
                compression: None,
                sha256: String::new(),
            }];
            let bytes = write_manifest_only(&m);
            let d = verify_members(&bytes, &m);
            assert_eq!(d.len(), 1);
            assert_eq!(d[0].code, "E-TRACE-CONTAINER");
        }
    }

    proptest! {
        #[test]
        fn header_round_trip(version: u32, flags: u32, off: u64, len: u64) {
            let h = Header { version, flags, manifest_offset: off, manifest_len: len };
            prop_assert_eq!(Header::decode(&h.encode()).unwrap(), h);
        }
    }
}
