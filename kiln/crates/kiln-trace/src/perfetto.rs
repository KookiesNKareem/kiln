//! Perfetto export (05 §3.11): native protobuf `Trace` with nested track descriptors following the resource
//! hierarchy, slices per span (interned names and categories), an execution-group track and a phase track;
//! plus the legacy Chrome JSON format.
//!
//! The protobuf messages are hand-declared `prost` structs carrying only the fields used, with the field
//! numbers of Perfetto's `trace_packet.proto`, `track_descriptor.proto`, `track_event.proto`,
//! `debug_annotation.proto` and `interned_data.proto` (v58.2); no `protoc` at build time.

use std::collections::{BTreeMap, BTreeSet};

use prost::Message;
use serde_json::json;

use crate::trace::{NONE_U32, Trace};

#[derive(Clone, PartialEq, Message)]
pub struct PTrace {
    #[prost(message, repeated, tag = "1")]
    pub packet: Vec<TracePacket>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TracePacket {
    #[prost(uint64, optional, tag = "8")]
    pub timestamp: Option<u64>,
    #[prost(uint32, optional, tag = "10")]
    pub trusted_packet_sequence_id: Option<u32>,
    #[prost(message, optional, tag = "11")]
    pub track_event: Option<TrackEvent>,
    #[prost(message, optional, tag = "12")]
    pub interned_data: Option<InternedData>,
    #[prost(uint32, optional, tag = "13")]
    pub sequence_flags: Option<u32>,
    #[prost(message, optional, tag = "60")]
    pub track_descriptor: Option<TrackDescriptor>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TrackDescriptor {
    #[prost(uint64, optional, tag = "1")]
    pub uuid: Option<u64>,
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
    #[prost(uint64, optional, tag = "5")]
    pub parent_uuid: Option<u64>,
    #[prost(int32, optional, tag = "11")]
    pub child_ordering: Option<i32>,
    #[prost(int32, optional, tag = "12")]
    pub sibling_order_rank: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TrackEvent {
    #[prost(uint64, repeated, packed = "false", tag = "3")]
    pub category_iids: Vec<u64>,
    #[prost(message, repeated, tag = "4")]
    pub debug_annotations: Vec<DebugAnnotation>,
    #[prost(int32, optional, tag = "9")]
    pub r#type: Option<i32>,
    #[prost(uint64, optional, tag = "10")]
    pub name_iid: Option<u64>,
    #[prost(uint64, optional, tag = "11")]
    pub track_uuid: Option<u64>,
    #[prost(string, optional, tag = "23")]
    pub name: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct DebugAnnotation {
    #[prost(uint64, optional, tag = "3")]
    pub uint_value: Option<u64>,
    #[prost(double, optional, tag = "5")]
    pub double_value: Option<f64>,
    #[prost(string, optional, tag = "6")]
    pub string_value: Option<String>,
    #[prost(string, optional, tag = "10")]
    pub name: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct InternedData {
    #[prost(message, repeated, tag = "1")]
    pub event_categories: Vec<InternedString>,
    #[prost(message, repeated, tag = "2")]
    pub event_names: Vec<InternedString>,
}

#[derive(Clone, PartialEq, Message)]
pub struct InternedString {
    #[prost(uint64, optional, tag = "1")]
    pub iid: Option<u64>,
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,
}

const SEQ: u32 = 1;
const SEQ_INCREMENTAL_STATE_CLEARED: u32 = 1;
const SEQ_NEEDS_INCREMENTAL_STATE: u32 = 2;
const SLICE_BEGIN: i32 = 1;
const SLICE_END: i32 = 2;
const EXPLICIT: i32 = 3;
const PHASE_TRACK: u64 = 1;
const GROUP_TRACK: u64 = 2;
const RES_BASE: u64 = 16;

/// One drawable slice before encoding.
struct Slice {
    track: u64,
    start_ns: u64,
    end_ns: u64,
    name: String,
    cat: String,
    args: Vec<DebugAnnotation>,
}

fn ann_s(name: &str, v: impl Into<String>) -> DebugAnnotation {
    DebugAnnotation {
        name: Some(name.into()),
        string_value: Some(v.into()),
        ..Default::default()
    }
}

fn ann_f(name: &str, v: f64) -> DebugAnnotation {
    DebugAnnotation {
        name: Some(name.into()),
        double_value: Some(v),
        ..Default::default()
    }
}

/// Greedy non-overlapping lane per span within its resource, so slices on one track always nest.
fn lanes(t: &Trace) -> Vec<u16> {
    let mut order: Vec<usize> = (0..t.spans.len()).collect();
    order.sort_by_key(|&i| (t.spans[i].resource, t.spans[i].t_start, i));
    let mut out = vec![0u16; t.spans.len()];
    let mut ends: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    for i in order {
        let s = &t.spans[i];
        let e = ends.entry(s.resource).or_default();
        let lane = e
            .iter()
            .position(|&end| end <= s.t_start)
            .unwrap_or(e.len());
        if lane == e.len() {
            e.push(0);
        }
        e[lane] = s.t_start + s.dur;
        out[i] = lane as u16;
    }
    out
}

pub fn export_perfetto(t: &Trace) -> Vec<u8> {
    let tick_ns = t.tick_s() * 1e9;
    let ns = |ticks: i64| (ticks.max(0) as f64 * tick_ns).round() as u64;
    let lane = lanes(t);
    // Tracks: every resource with spans, its ancestors, and one child track per extra lane.
    let mut used: BTreeSet<u32> = BTreeSet::new();
    for s in &t.spans {
        let mut cur = Some(s.resource);
        while let Some(c) = cur {
            if !used.insert(c) {
                break;
            }
            cur = t.resources.get(c as usize).and_then(|r| r.parent);
        }
    }
    let res_uuid = |r: u32| RES_BASE + u64::from(r) * 256;
    let mut packets: Vec<TracePacket> = Vec::new();
    let desc = |uuid: u64, name: String, parent: Option<u64>, rank: i32| TracePacket {
        trusted_packet_sequence_id: Some(SEQ),
        track_descriptor: Some(TrackDescriptor {
            uuid: Some(uuid),
            name: Some(name),
            parent_uuid: parent,
            child_ordering: Some(EXPLICIT),
            sibling_order_rank: Some(rank),
        }),
        ..Default::default()
    };
    packets.push(desc(PHASE_TRACK, "phases".into(), None, -2));
    packets.push(desc(GROUP_TRACK, "execution groups".into(), None, -1));
    let mut max_lane: BTreeMap<u32, u16> = BTreeMap::new();
    for (s, &l) in t.spans.iter().zip(&lane) {
        let e = max_lane.entry(s.resource).or_default();
        *e = (*e).max(l);
    }
    for &r in &used {
        let row = &t.resources[r as usize];
        let parent = row.parent.filter(|p| used.contains(p)).map(res_uuid);
        let name = match parent {
            Some(_) => row.path.rsplit('.').next().unwrap_or(&row.path).to_string(),
            None => row.path.clone(),
        };
        packets.push(desc(res_uuid(r), name.clone(), parent, r as i32));
        for l in 1..=max_lane.get(&r).copied().unwrap_or(0) {
            packets.push(desc(
                res_uuid(r) + u64::from(l),
                format!("{name} #{l}"),
                parent,
                r as i32,
            ));
        }
    }
    let mut slices: Vec<Slice> = Vec::new();
    for p in &t.phases {
        slices.push(Slice {
            track: PHASE_TRACK,
            start_ns: ns(p.t_offset),
            end_ns: ns(p.t_end),
            name: p.id.clone(),
            cat: "phase".into(),
            args: vec![
                ann_f("makespan_s", p.makespan_s),
                ann_f("energy_j", p.energy_j),
                ann_s("summary", p.summary.clone()),
            ],
        });
    }
    for g in &t.groups {
        slices.push(Slice {
            track: GROUP_TRACK,
            start_ns: ns(g.t_start),
            end_ns: ns(g.t_end),
            name: format!("group {}", g.group),
            cat: t.enum_name("groups.kind", u32::from(g.kind)).into(),
            args: vec![
                ann_s("binding", t.binding_name(g.binding)),
                ann_f("bubble_s", g.bubble_s),
                ann_f("exposed_overhead_s", g.exposed_overhead_s),
            ],
        });
    }
    for (s, &l) in t.spans.iter().zip(&lane) {
        let op = (s.op != NONE_U32).then(|| &t.ops[s.op as usize]);
        slices.push(Slice {
            track: res_uuid(s.resource) + u64::from(l),
            start_ns: ns(s.t_start),
            end_ns: ns(s.t_start + s.dur),
            name: op.map_or_else(
                || t.enum_name("spans.kind", u32::from(s.kind)).to_string(),
                |o| o.path.clone(),
            ),
            cat: op.map_or("span", |o| t.op_kind(o)).to_string(),
            args: vec![
                ann_s("kind", t.enum_name("spans.kind", u32::from(s.kind))),
                ann_f("bytes", s.bytes),
                ann_f("energy_j", f64::from(s.energy_j)),
                ann_s(
                    "estimated",
                    if s.flags & crate::trace::span_flags::ESTIMATED != 0 {
                        "yes"
                    } else {
                        "no"
                    },
                ),
            ],
        });
    }
    // Events in time order; an end sorts before a begin at the same time on the same track.
    let mut events: Vec<(u64, u8, usize)> = Vec::with_capacity(slices.len() * 2);
    for (i, s) in slices.iter().enumerate() {
        events.push((s.start_ns, 1, i));
        events.push((s.end_ns.max(s.start_ns), 0, i));
    }
    // Same time: ends first, inner (later-started) ends before outer; outer (later-ending) begins first.
    events.sort_by_key(|&(ts, kind, i)| {
        let s = &slices[i];
        (
            ts,
            kind,
            u64::MAX - if kind == 0 { s.start_ns } else { s.end_ns },
            i,
        )
    });
    let mut names: BTreeMap<String, u64> = BTreeMap::new();
    let mut cats: BTreeMap<String, u64> = BTreeMap::new();
    let mut first = true;
    for (ts, kind, i) in events {
        let s = &slices[i];
        let mut interned = InternedData::default();
        let mut ev = TrackEvent {
            track_uuid: Some(s.track),
            ..Default::default()
        };
        if kind == 1 {
            let n = names.len() as u64 + 1;
            let iid = *names.entry(s.name.clone()).or_insert_with(|| {
                interned.event_names.push(InternedString {
                    iid: Some(n),
                    name: Some(s.name.clone()),
                });
                n
            });
            let c = cats.len() as u64 + 1;
            let cid = *cats.entry(s.cat.clone()).or_insert_with(|| {
                interned.event_categories.push(InternedString {
                    iid: Some(c),
                    name: Some(s.cat.clone()),
                });
                c
            });
            ev.r#type = Some(SLICE_BEGIN);
            ev.name_iid = Some(iid);
            ev.category_iids = vec![cid];
            ev.debug_annotations = s.args.clone();
        } else {
            ev.r#type = Some(SLICE_END);
        }
        let has_interned =
            !interned.event_names.is_empty() || !interned.event_categories.is_empty();
        packets.push(TracePacket {
            timestamp: Some(ts),
            trusted_packet_sequence_id: Some(SEQ),
            track_event: Some(ev),
            interned_data: has_interned.then_some(interned),
            sequence_flags: Some(if first {
                SEQ_INCREMENTAL_STATE_CLEARED | SEQ_NEEDS_INCREMENTAL_STATE
            } else {
                SEQ_NEEDS_INCREMENTAL_STATE
            }),
            ..Default::default()
        });
        first = false;
    }
    PTrace { packet: packets }.encode_to_vec()
}

/// Legacy Chrome JSON (`traceEvents`, complete events in microseconds); one thread per resource lane.
pub fn export_chrome_json(t: &Trace) -> serde_json::Value {
    let us = |ticks: i64| ticks as f64 * t.tick_s() * 1e6;
    let lane = lanes(t);
    let mut ev = vec![];
    let mut tids: BTreeMap<(u32, u16), u64> = BTreeMap::new();
    for (s, &l) in t.spans.iter().zip(&lane) {
        let n = tids.len() as u64 + 10;
        let tid = *tids.entry((s.resource, l)).or_insert(n);
        let op = (s.op != NONE_U32).then(|| &t.ops[s.op as usize]);
        ev.push(json!({
            "name": op.map_or("span", |o| o.path.as_str()), "cat": op.map_or("span", |o| t.op_kind(o)),
            "ph": "X", "ts": us(s.t_start), "dur": us(s.dur), "pid": 1, "tid": tid,
            "args": {"bytes": s.bytes, "energy_j": s.energy_j, "kind": t.enum_name("spans.kind", u32::from(s.kind))},
        }));
    }
    for p in &t.phases {
        ev.push(json!({"name": p.id, "cat": "phase", "ph": "X", "ts": us(p.t_offset), "dur": us(p.t_end - p.t_offset), "pid": 1, "tid": 1}));
    }
    for g in &t.groups {
        ev.push(json!({"name": format!("group {}", g.group), "cat": t.enum_name("groups.kind", u32::from(g.kind)), "ph": "X",
            "ts": us(g.t_start), "dur": us(g.t_end - g.t_start), "pid": 1, "tid": 2}));
    }
    let mut meta = vec![
        json!({"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "phases"}}),
        json!({"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "execution groups"}}),
    ];
    for ((r, l), tid) in &tids {
        let p = &t.resources[*r as usize].path;
        let name = if *l == 0 {
            p.clone()
        } else {
            format!("{p} #{l}")
        };
        meta.push(
            json!({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}}),
        );
        meta.push(json!({"name": "thread_sort_index", "ph": "M", "pid": 1, "tid": tid, "args": {"sort_index": tid}}));
    }
    meta.extend(ev);
    json!({"traceEvents": meta, "displayTimeUnit": "ns"})
}
