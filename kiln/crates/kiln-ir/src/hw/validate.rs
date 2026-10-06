//! Structural validation (01 §18, pipeline step 12) and validation profiles (§18.3).

use serde::{Deserialize, Serialize};

use super::compute::{
    AccumulateIn, CimSpec, CimStyle, ComputeKind, ComputeUnit, Dataflow, Geometry, MemKind, MemStack, Memory, NearAccessMode,
    NearGranularity, OperandPolicy, OperandRole, PrecisionMode, StackAttach,
};
use super::diag::Diags;
use super::model::{ChannelKind, ContainerKind, HwModel, MemSpec, NodeIx, UnitInst};
use super::net::{LinkPhys, LinkSpec, Routing, Topology};
use super::phys::{KNOWN_TECH, Placement, PowerOverride};
use super::quantity::Hz;
use super::types::{Contents, HwDoc, Layout};
use crate::common::Diagnostic;
use crate::precision::{Precision, PrecisionKind};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    #[default]
    Full,
    Reference,
    Search,
    StreamCompat,
}

/// Largest per-unit base operation count per cycle (2^32): far beyond any datapath, and products stay exact in f64.
const MAX_OPS_PER_CYCLE: u64 = 1 << 32;

pub fn valid_id(s: &str) -> bool {
    s.starts_with(|c: char| c.is_ascii_lowercase())
        && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

struct V<'a> {
    doc: &'a HwDoc,
    m: &'a HwModel,
    d: Diags,
    profile: Profile,
    pricer: Option<&'a dyn Pricer>,
}

/// A "less is better" performance override (energy, area, leakage, latency, power) whose derived value only the
/// physical model (04) can compute; `search` allows it at >= derived (01 §18.3, 00 decision 3).
#[derive(Clone, Debug, PartialEq)]
pub struct PricedField {
    /// Arena node of the overriding entity (network node for link and router overrides).
    pub node: usize,
    /// Field name as E-IR-1101 reports it (`power.area`, `read_energy`, `link.latency`, ...).
    pub field: &'static str,
    /// Sub-key (`power.energy_per_op` op name).
    pub key: Option<String>,
    /// Override in base units: um^2, J, J/B, W, s, cycles or gate equivalents.
    pub value: f64,
}

/// Derived values for [`PricedField`]s (implemented by kiln-phys); `None` = cannot derive, stays unverifiable.
pub trait Pricer {
    fn derived(&self, f: &PricedField) -> Option<f64>;
}

/// Runs every structural (S) check plus the profile's extra rules.
pub fn validate(doc: &HwDoc, m: &HwModel, profile: Profile) -> Vec<Diagnostic> {
    validate_priced(doc, m, profile, None)
}

/// [`validate`] with a pricer: under `search`, priced "less is better" overrides are compared with the derived
/// value instead of being rejected as unverifiable.
pub fn validate_priced(doc: &HwDoc, m: &HwModel, profile: Profile, pricer: Option<&dyn Pricer>) -> Vec<Diagnostic> {
    let mut v = V { doc, m, d: Diags::default(), profile, pricer };
    v.ids_and_tech();
    v.clocks_and_power();
    v.units();
    v.memories();
    v.stacks();
    v.networks();
    v.staging();
    v.floorplan();
    v.system();
    v.claims();
    v.profiles();
    v.d.into_vec()
}

fn err(code: &str, msg: impl Into<String>, at: &str) -> Diagnostic {
    Diagnostic::error(code, msg).at(at)
}

impl V<'_> {
    fn push(&mut self, entity: &str, d: Diagnostic) {
        self.d.push_inst(entity, d);
    }

    fn ids_and_tech(&mut self) {
        for n in self.m.nodes.iter().skip(1) {
            if !matches!(n.ix, NodeIx::Router(_)) && !valid_id(&n.entity_id) {
                let fixed = n.entity_id.to_lowercase().replace('.', "_");
                self.push(
                    &n.entity,
                    err("E-IR-0104", format!("id '{}' invalid: ids match ^[a-z][a-z0-9_-]*$", n.entity_id), &n.path)
                        .hint(format!("use '{fixed}'")),
                );
            }
        }
        let mut techs = vec![(self.doc.tech.node().to_owned(), String::new())];
        for c in &self.m.tree {
            if let Some(t) = &c.tech {
                techs.push((t.clone(), self.m.nodes[c.node].entity.clone()));
            }
        }
        for mem in &self.m.memories {
            if let MemSpec::Stack(s) = &mem.spec
                && let Some(t) = s.logic_die.as_ref().and_then(|l| l.tech.as_ref())
            {
                techs.push((t.node().to_owned(), self.m.nodes[mem.node].entity.clone()));
            }
        }
        for (t, at) in techs {
            if !KNOWN_TECH.contains(&t.as_str()) {
                self.push(
                    &format!("tech:{t}"),
                    err("E-IR-1001", format!("unknown tech node '{t}'"), &at).hint(format!("known nodes: {KNOWN_TECH:?}")),
                );
            }
        }
    }

    fn clocks_and_power(&mut self) {
        for c in &self.m.clocks {
            let s = &c.spec;
            for (what, f) in [("freq", Some(s.freq)), ("base", s.base)] {
                if let Some(Hz(f)) = f
                    && !(1e6..=20e9).contains(&f)
                {
                    self.push(
                        &c.path,
                        err("E-IR-0109", format!("clock {what} {} GHz outside [1 MHz, 20 GHz]", f / 1e9), &c.path)
                            .hint("did you mean a value in MHz/GHz? e.g. \"1.41GHz\""),
                    );
                }
            }
            if s.vf.windows(2).any(|w| w[1].freq.0 <= w[0].freq.0 || w[1].voltage.0 < w[0].voltage.0) {
                self.push(&c.path, err("E-IR-0902", "vf points must ascend in freq with non-decreasing voltage", &c.path));
            }
            if let (Some(lo), Some(hi)) = (s.vf.first(), s.vf.last()) {
                let range = lo.freq.0..=hi.freq.0;
                for (what, f) in [("freq", Some(s.freq)), ("base", s.base)] {
                    if let Some(f) = f
                        && !range.contains(&f.0)
                    {
                        self.push(
                            &c.path,
                            err("E-IR-0903", format!("{what} {} MHz outside the vf curve", f.0 / 1e6), &c.path)
                                .hint("extend `vf` or move the operating point inside it"),
                        );
                    }
                }
            }
        }
        for pd in &self.m.power_domains {
            if let Some(r) = &pd.assumed
                && !(r.lo.0 > 0.0 && r.lo.0 <= pd.cap.0 && pd.cap.0 <= r.hi.0)
            {
                self.push(
                    &pd.path,
                    err("E-IR-0908", format!("assumed cap {} W outside its range [{}, {}] W", pd.cap.0, r.lo.0, r.hi.0), &pd.path)
                        .hint("an assumed cap is a nominal value inside the plausible range: 0 < lo <= cap <= hi"),
                );
            }
        }
        for c in &self.m.tree {
            if c.kind != ContainerKind::Die {
                continue;
            }
            let covering: Vec<&str> = self
                .m
                .power_domains
                .iter()
                .filter(|pd| pd.members.iter().any(|&mc| self.ancestor_or_self(self.m.tree[mc].node, c.node)))
                .map(|pd| pd.path.as_str())
                .collect();
            if covering.len() > 1 {
                let at = &self.m.nodes[c.node].path;
                self.push(
                    &self.m.nodes[c.node].entity.clone(),
                    err("E-IR-0904", format!("die in power domains {covering:?}"), at)
                        .hint("an entity belongs to at most one power domain; drop one cap or narrow `members`"),
                );
            }
        }
    }

    fn ancestor_or_self(&self, anc: usize, mut n: usize) -> bool {
        loop {
            if n == anc {
                return true;
            }
            match self.m.nodes[n].parent {
                Some(p) => n = p,
                None => return false,
            }
        }
    }

    fn die_of(&self, mut n: usize) -> Option<usize> {
        loop {
            if let NodeIx::Container(c) = self.m.nodes[n].ix
                && matches!(self.m.tree[c].kind, ContainerKind::Die | ContainerKind::Package)
            {
                return Some(n);
            }
            n = self.m.nodes[n].parent?;
        }
    }

    fn units(&mut self) {
        for (ui, u) in self.m.units.iter().enumerate() {
            let n = &self.m.nodes[u.node];
            if !n.enabled {
                continue;
            }
            let (key, at) = (n.entity.clone(), n.path.clone());
            let s = &u.spec;
            self.unit_spec(s, &key, &at);
            let mut covered: Vec<OperandRole> = u.feeds.keys().copied().collect();
            for &li in &u.local {
                if let MemSpec::Local { buffer, .. } = &self.m.memories[li].spec
                    && buffer.refill_from.is_some()
                {
                    covered.push(buffer.holds);
                }
            }
            let needed: Vec<OperandRole> = match &s.kind {
                _ if u.near.is_some() => vec![],
                ComputeKind::Matrix(_) => vec![OperandRole::A, OperandRole::B, OperandRole::O],
                ComputeKind::Cim(_) => vec![OperandRole::A, OperandRole::O],
                _ => vec![OperandRole::In, OperandRole::Out],
            };
            for role in needed {
                let ok = covered.iter().any(|&c| c == role || c == OperandRole::Any)
                    || (role == OperandRole::In && covered.contains(&OperandRole::A))
                    || (role == OperandRole::Out && covered.contains(&OperandRole::O));
                if !ok {
                    let r = serde_json::to_value(role).unwrap_or_default();
                    let r = r.as_str().unwrap_or("?");
                    self.push(
                        &key,
                        err("E-IR-0304", format!("unit '{}' reads operand '{r}' but has no feed for '{r}'", s.id), &at)
                            .hint(format!("add feeds: {{ {r}: \"<memory>\" }} (or `any`)")),
                    );
                }
            }
            for (role, f) in &u.feeds {
                let mem = &self.m.memories[f.mem];
                if let (Some(w), MemSpec::OnChip(m)) = (s.feeds[role].width_bits, &mem.spec)
                    && w > m.widest_port_bits()
                {
                    self.push(
                        &key,
                        err(
                            "E-IR-0310",
                            format!("feed '{role:?}' of '{}' wants {w} b/cycle; memory '{}' ports are {} b", s.id, m.id, m.widest_port_bits()),
                            &at,
                        )
                        .hint("widen the memory ports or lower width_bits"),
                    );
                }
                if let Some(near) = &u.near
                    && near.mem != f.mem
                {
                    self.push(&key, err("E-IR-0603", "near unit declares a feed from a memory other than its bound one", &at));
                }
                if self.die_of(u.node) != self.die_of(mem.node) && !mem.is_stack() {
                    self.push(
                        &key,
                        err("E-IR-0722", format!("feed from '{}' crosses dies; no channel can be built", self.m.nodes[mem.node].path), &at)
                            .hint("stage through a local memory reached over a d2d network"),
                    );
                }
                if let Some(ni) = f.via {
                    let eps = &self.m.networks[ni].endpoints;
                    if !(eps.contains(&NodeIx::Unit(ui)) && eps.contains(&NodeIx::Mem(f.mem))) {
                        self.push(
                            &key,
                            err("E-IR-0714", format!("feed via '{}' but unit and memory are not both its endpoints", self.m.nodes[self.m.networks[ni].node].path), &at),
                        );
                    }
                }
                if f.via.is_none()
                    && u.clock != mem.clock
                    && mem.clock.is_some()
                    && mem.clock.is_some_and(|c| self.m.clocks[c].spec.crossing_latency.is_none())
                {
                    self.push(
                        &format!("{key}#0906"),
                        Diagnostic::warning("W-IR-0906", "feed crosses clock domains with no crossing_latency (default applied)")
                            .at(&at),
                    );
                }
            }
            if let (Some(near), Some(ni)) = (&s.near, &u.near) {
                let mem = &self.m.memories[ni.mem];
                let bad = match (&mem.spec, near.granularity) {
                    (MemSpec::OnChip(_), NearGranularity::PerPseudoChannel | NearGranularity::PerChannel) => true,
                    (MemSpec::OnChip(m), NearGranularity::PerBank) => m.banks == 1 && n.count > 1,
                    _ => false,
                };
                if bad {
                    self.push(&key, err("E-IR-0602", format!("granularity {:?} incompatible with '{}'", near.granularity, self.m.nodes[mem.node].path), &at));
                }
                if near.access_mode == NearAccessMode::ExclusiveAllBank
                    && let MemSpec::Stack(st) = &mem.spec
                    && matches!(st.attach, None | Some(StackAttach::Vertical { .. }))
                {
                    self.push(&key, err("E-IR-0604", "exclusive_all_bank stack has no host-side controller to issue commands", &at));
                }
            }
            self.local_buffers(s, &key, &at);
        }
    }

    fn local_buffers(&mut self, s: &ComputeUnit, key: &str, at: &str) {
        let ComputeKind::Matrix(mx) = &s.kind else { return };
        let Geometry::Systolic { rows, cols } = mx.geometry else { return };
        let pes = f64::from(rows) * f64::from(cols);
        let widest = |f: fn(&PrecisionMode) -> Option<Precision>| {
            s.precisions.iter().filter_map(f).map(|p| p.storage_bits() / 8.0).fold(0.0, f64::max)
        };
        let holds = |role: OperandRole| -> f64 {
            s.local
                .iter()
                .filter(|l| l.holds == role || l.holds == OperandRole::Any)
                .map(|l| l.capacity.as_f64())
                .fold(0.0, f64::max)
        };
        let flows = mx.dataflow.as_ref().map_or_else(|| vec![mx.geometry.default_dataflow()], |d| d.all());
        if flows.contains(&Dataflow::WeightStationary) {
            let need = pes * widest(|m| match m {
                PrecisionMode::Mac { b, .. } => Some(b.precision),
                _ => None,
            });
            if holds(OperandRole::B) < need {
                self.push(
                    key,
                    err("E-IR-0311", format!("weight-stationary '{}' needs a 'b' local buffer >= {need} B", s.id), at)
                        .hint("add local: [{ id: \"w\", holds: \"b\", capacity: ... }]"),
                );
            }
        }
        if flows.contains(&Dataflow::OutputStationary) && mx.accumulate_in != AccumulateIn::Feed {
            let need = pes * widest(|m| match m {
                PrecisionMode::Mac { acc, .. } => Some(acc.precision),
                _ => None,
            });
            if holds(OperandRole::O).max(holds(OperandRole::C)) < need {
                self.push(
                    key,
                    err("E-IR-0311", format!("output-stationary '{}' needs a 'c'/'o' local buffer >= {need} B", s.id), at)
                        .hint("add an accumulator local buffer or set accumulate_in: \"feed\""),
                );
            }
        }
    }

    fn unit_spec(&mut self, s: &ComputeUnit, key: &str, at: &str) {
        if s.precisions.is_empty() {
            self.push(key, err("E-IR-0301", format!("unit '{}' has no precision modes", s.id), at).hint("add e.g. precisions: [\"bf16*bf16+fp32\"]"));
        }
        let mut keys = vec![];
        for m in &s.precisions {
            if m.rate() <= 0.0 || !m.rate().is_finite() {
                self.push(key, err("E-IR-0303", format!("mode {m} has rate <= 0"), at));
            }
            for p in m.operands() {
                if !p.is_compute_name() {
                    self.push(
                        key,
                        err("E-IR-0302", format!("'{p}' is a storage/scale name, not a compute precision"), at)
                            .hint(format!("hardware modes use compute names; '{p}' computes as '{}'", p.compute())),
                    );
                }
            }
            if let PrecisionMode::Mac { a, b, acc, .. } = m {
                let (a, b, acc) = (a.precision, b.precision, acc.precision);
                let ok = if a.is_integer() && b.is_integer() {
                    matches!(acc, Precision::Int32 | Precision::Int16)
                } else {
                    let need = a.exponent_bits().unwrap_or(0).max(b.exponent_bits().unwrap_or(0));
                    acc.is_float() && acc.exponent_bits().unwrap_or(0) >= need && !matches!(acc.kind(), PrecisionKind::Mx)
                };
                if !ok {
                    self.push(
                        key,
                        err("E-IR-0303", format!("accumulator '{acc}' incompatible with '{a}*{b}'"), at)
                            .hint("integer inputs need int32/int16 accumulators; float inputs need a float accumulator with at least their exponent range"),
                    );
                }
                if let Some(k) = match &s.kind {
                    ComputeKind::Matrix(mx) => mx.geometry.reduction_extent(),
                    ComputeKind::Cim(c) => Some(c.rows),
                    _ => None,
                } {
                    for p in [m_a(m), m_b(m)].into_iter().flatten() {
                        if let Some(bs) = p.block_size()
                            && matches!(p.precision.kind(), PrecisionKind::Mx | PrecisionKind::BlockFloat)
                            && k % bs != 0
                        {
                            self.push(
                                &format!("{key}#0312"),
                                Diagnostic::warning("W-IR-0312", format!("MX block {bs} of '{p}' does not divide reduction extent {k}")).at(at),
                            );
                        }
                    }
                }
            }
            if keys.contains(&m.key()) {
                self.push(&format!("{key}#0313"), Diagnostic::warning("W-IR-0313", format!("duplicate precision mode {}", m.key())).at(at));
            }
            keys.push(m.key());
        }
        for (name, r, _) in rate_multipliers(s) {
            if r <= 0.0 || !r.is_finite() {
                self.push(key, err("E-IR-0303", format!("{name} rate {r} is not finite and > 0"), at));
            }
        }
        if let Some(ops) = &s.ops {
            let legal = s.kind.legal_ops();
            for op in ops.iter().filter(|o| !legal.contains(o)) {
                self.push(key, err("E-IR-0306", format!("op class {op:?} not legal on a {} unit", s.kind.name()), at).hint(format!("legal: {legal:?}")));
            }
        }
        let zero = match &s.kind {
            ComputeKind::Matrix(mx) => mx.geometry.dims().contains(&0),
            ComputeKind::Vector(v) => v.lanes == 0 || v.sublanes == 0,
            ComputeKind::Scalar(sc) => sc.issue_width == 0,
            ComputeKind::Special(sp) => sp.lanes == 0,
            ComputeKind::Cim(c) => {
                [c.rows, c.cols, c.cell_bits, c.input_bits_per_cycle, c.weight_sets, c.active_rows()].contains(&0)
            }
        };
        if zero {
            self.push(key, err("E-IR-0307", format!("unit '{}' has a zero geometry dimension or lane count", s.id), at));
        }
        if s.kind.checked_base_ops_per_cycle().is_none_or(|n| n > MAX_OPS_PER_CYCLE) {
            self.push(
                key,
                err("E-IR-0109", format!("unit '{}' geometry exceeds {MAX_OPS_PER_CYCLE} operations per cycle", s.id), at)
                    .hint("one unit is one datapath; replicate it with `count` instead"),
            );
        }
        if let ComputeKind::Matrix(mx) = &s.kind {
            for df in mx.dataflow.as_ref().map(|d| d.all()).unwrap_or_default() {
                let ok = match (&mx.geometry, df) {
                    (_, Dataflow::Any) | (Geometry::Spatial { .. }, _) => true,
                    (_, Dataflow::RowStationary) => false,
                    (Geometry::OuterProduct { .. }, Dataflow::InputStationary) => false,
                    _ => true,
                };
                if !ok {
                    self.push(key, err("E-IR-0308", format!("dataflow {df:?} not supported by this geometry"), at));
                }
            }
            for sp in &mx.sparsity {
                let bad_pattern = match sp.pattern.split_once(':') {
                    Some(("block", n)) => n.parse::<u32>().map_or(true, |n| n == 0),
                    Some((n, m)) => match (n.parse::<u32>(), m.parse::<u32>()) {
                        (Ok(n), Ok(m)) => n == 0 || n >= m,
                        _ => true,
                    },
                    None => sp.pattern != "unstructured",
                };
                if !sp.speedup.is_finite() || sp.speedup <= 1.0 || bad_pattern || !matches!(sp.operand, OperandRole::A | OperandRole::B) {
                    self.push(key, err("E-IR-0309", format!("invalid sparsity {:?} (speedup {}) on operand {:?}", sp.pattern, sp.speedup, sp.operand), at));
                } else if let Some(max) = sp.max_speedup().filter(|&max| sp.speedup > max * (1.0 + 1e-9)) {
                    self.push(
                        key,
                        err("E-IR-0309", format!("sparsity {:?} speedup {} exceeds the {max} its structure allows", sp.pattern, sp.speedup), at)
                            .hint("an n:m pattern skips at most the zeros: speedup <= m/n"),
                    );
                }
            }
        }
        if let ComputeKind::Cim(c) = &s.kind {
            self.cim(s, c, key, at);
        }
    }

    fn cim(&mut self, s: &ComputeUnit, c: &CimSpec, key: &str, at: &str) {
        let Some(derived) = c.derived_capacity() else {
            self.push(key, err("E-IR-0109", format!("CIM '{}' stores more than 2^64 bytes", s.id), at).hint("replicate the array with `count` instead"));
            return;
        };
        if let Some(cap) = c.weight_capacity.filter(|&cap| cap != derived) {
            self.push(
                key,
                err("E-IR-0606", format!("CIM weight_capacity {} B != rows*cols*cell_bits*weight_sets/8 = {} B", cap.0, derived.0), at)
                    .hint("the array stores cell_bits per cell; omit weight_capacity or change rows/cols/cell_bits/weight_sets"),
            );
        }
        for m in s.precisions.iter().filter(|m| m.rate() > 1.0) {
            self.push(
                &format!("{key}#0608"),
                err("E-IR-0608", format!("CIM mode {m}: @rate > 1 claims throughput beyond the array-derived rate"), at)
                    .hint("CIM throughput is derived from parallel_rows, cols, cell_bits and input_bits_per_cycle (01 §9.3); raise those instead"),
            );
        }
        let mut bad = vec![];
        if c.active_rows() > c.rows {
            bad.push(format!("parallel_rows {} > rows {}", c.active_rows(), c.rows));
        }
        if c.cell_bits > 8 || c.input_bits_per_cycle > 16 {
            bad.push(format!("cell_bits {} (max 8) / input_bits_per_cycle {} (max 16)", c.cell_bits, c.input_bits_per_cycle));
        }
        match (c.style, c.adc_bits) {
            (CimStyle::Analog, None) => bad.push("analog CIM without adc_bits cannot be priced".into()),
            (CimStyle::Digital, Some(_)) => bad.push("adc_bits on a digital CIM".into()),
            _ => {}
        }
        for b in bad {
            self.push(key, err("E-IR-0609", format!("CIM parameter out of range: {b}"), at));
        }
        if let (CimStyle::Analog, Some(adc)) = (c.style, c.adc_bits)
            && adc < c.boundary_adc_bits()
        {
            self.push(
                &format!("{key}#0610"),
                Diagnostic::warning("W-IR-0610", format!("adc_bits {adc} below the lossless {} bits for this parallel_rows; accuracy loss is not modeled", c.boundary_adc_bits()))
                    .at(at),
            );
        }
    }

    fn memories(&mut self) {
        let mut fed_by: Vec<Vec<usize>> = vec![vec![]; self.m.memories.len()];
        for u in &self.m.units {
            for f in u.feeds.values() {
                if let Some(p) = self.m.nodes[u.node].parent {
                    fed_by[f.mem].push(p);
                }
            }
        }
        let mut referenced = vec![false; self.m.memories.len()];
        for u in &self.m.units {
            u.feeds.values().for_each(|f| referenced[f.mem] = true);
            if let Some(n) = &u.near {
                referenced[n.mem] = true;
            }
        }
        for (mi, mem) in self.m.memories.iter().enumerate() {
            mem.backing.iter().for_each(|&b| referenced[b] = true);
            if !mem.backing.is_empty() {
                referenced[mi] = true;
            }
        }
        for n in &self.m.networks {
            for e in &n.endpoints {
                if let NodeIx::Mem(mi) = e {
                    referenced[*mi] = true;
                }
            }
        }
        for c in &self.m.channels {
            for e in [c.src, c.dst] {
                if let NodeIx::Mem(mi) = e {
                    referenced[mi] = true;
                }
            }
        }
        for (mi, mem) in self.m.memories.iter().enumerate() {
            let n = &self.m.nodes[mem.node];
            if !n.enabled {
                continue;
            }
            let (key, at) = (n.entity.clone(), n.path.clone());
            let MemSpec::OnChip(m) = &mem.spec else { continue };
            self.memory_spec(m, &key, &at);
            if let (Some(bw), Some(f)) = (m.overrides.bandwidth, self.m.clock_hz(mem.clock)) {
                let derived = m.port_bits_per_cycle() as f64 * f.0 / 8.0;
                if bw.0 > derived * (1.0 + 1e-9) {
                    self.push(
                        &key,
                        err("E-IR-0409", format!("bandwidth override {} GB/s exceeds port bandwidth {} GB/s", bw.0 / 1e9, derived / 1e9), &at)
                            .hint("widen ports or lower the override"),
                    );
                }
            }
            if m.kind == MemKind::RegisterFile && !m.shared_rf {
                let mut parents = fed_by[mi].clone();
                parents.sort_unstable();
                parents.dedup();
                if parents.len() > 1 {
                    self.push(
                        &format!("{key}#0410"),
                        Diagnostic::warning("W-IR-0410", "register file fed by units of several cluster instances").at(&at).hint("set shared_rf: true if intended"),
                    );
                }
            }
            if !referenced[mi] {
                self.push(
                    &format!("{key}#0407"),
                    Diagnostic::warning("W-IR-0407", format!("dead memory '{}': no feed, network or backing reaches it", m.id)).at(&at),
                );
            }
            let tech = mem_tech(self.m, mem.node);
            let unavailable = match m.implementation {
                super::compute::MemImpl::Edram => {
                    ["tsmc_n5", "tsmc_n4", "tsmc_n4p", "nvidia_4n", "tsmc_n3e", "tsmc_n2", "asap7"].contains(&tech.as_str())
                }
                super::compute::MemImpl::Mram => !["tsmc_n16", "tsmc_n12", "tsmc_n7"].contains(&tech.as_str()),
                _ => false,
            };
            if unavailable {
                self.push(&key, err("E-IR-1002", format!("{:?} memory not available at {tech}", m.implementation), &at));
            }
        }
        for (mi, mem) in self.m.memories.iter().enumerate() {
            let mut seen = vec![mi];
            let mut cur = mem.backing.clone();
            while !cur.is_empty() {
                if cur.contains(&mi) {
                    let n = &self.m.nodes[mem.node];
                    self.push(&n.entity.clone(), err("E-IR-0404", "backing cycle", &n.path.clone()));
                    break;
                }
                let next: Vec<usize> = cur
                    .iter()
                    .filter(|&&c| !seen.contains(&c))
                    .flat_map(|&c| self.m.memories[c].backing.clone())
                    .collect();
                seen.extend(cur);
                cur = next;
            }
        }
    }

    fn memory_spec(&mut self, m: &Memory, key: &str, at: &str) {
        if m.ports.is_empty() {
            self.push(key, err("E-IR-0402", format!("memory '{}' has no ports", m.id), at).hint("add ports: [{ dir: \"rw\", width_bits: 512 }]"));
        }
        if m.word_bits < 8 || !m.word_bits.is_power_of_two() {
            self.push(key, err("E-IR-0408", format!("word_bits {} is not a power-of-two multiple of 8", m.word_bits), at));
        } else {
            let gran = u64::from(m.banks) * u64::from(m.word_bits / 8);
            if gran == 0 || !m.capacity.0.is_multiple_of(gran) {
                self.push(key, err("E-IR-0401", format!("capacity {} B not divisible by banks*word ({gran} B)", m.capacity.0), at));
            }
        }
        let carve_cache = matches!(&m.operands, OperandPolicy::Carveout { options } if options.iter().any(|o| o.cache.0 > 0));
        let bad_cache = match (m.kind, &m.cache) {
            (MemKind::Cache, None) => true,
            (k, Some(_)) if k != MemKind::Cache => !carve_cache,
            (_, None) => carve_cache,
            _ => false,
        };
        if bad_cache {
            self.push(key, err("E-IR-0403", "cache spec missing or misplaced (kind: cache or a carveout with cache > 0 needs `cache`)", at));
        }
        match &m.operands {
            OperandPolicy::Carveout { options } => {
                for o in options.iter().filter(|o| u128::from(o.scratch.0) + u128::from(o.cache.0) > u128::from(m.capacity.0)) {
                    self.push(key, err("E-IR-0405", format!("carveout {} + {} B exceeds capacity {} B", o.scratch.0, o.cache.0, m.capacity.0), at));
                }
            }
            OperandPolicy::Partitioned { parts } => {
                let sum: u128 = parts.values().map(|b| u128::from(b.0)).sum();
                if sum > u128::from(m.capacity.0) {
                    self.push(key, err("E-IR-0406", format!("partitions sum {sum} B exceed capacity {} B", m.capacity.0), at));
                }
            }
            OperandPolicy::Unified => {}
        }
        if let Some(c) = &m.cache {
            let word = u64::from(m.word_bits / 8).max(1);
            if c.line.0 == 0 || c.line.0 % word != 0 || u128::from(c.ways) * u128::from(c.line.0) > u128::from(m.capacity.0) {
                self.push(key, err("E-IR-0411", "cache line not a multiple of the word, or ways*line > capacity", at));
            }
        }
    }

    fn stacks(&mut self) {
        for (mi, mem) in self.m.memories.iter().enumerate() {
            let MemSpec::Stack(s) = &mem.spec else { continue };
            let n = &self.m.nodes[mem.node];
            if !n.enabled {
                continue;
            }
            let (key, at) = (n.entity.clone(), n.path.clone());
            self.stack_spec(s, &key, &at);
            if let Some(StackAttach::Phys { phys }) = &s.attach {
                let phy_lanes: Vec<Option<u32>> = self
                    .m
                    .channels
                    .iter()
                    .filter(|c| c.src == NodeIx::Mem(mi))
                    .filter_map(|c| match c.dst {
                        NodeIx::Block(b) => match &self.m.blocks[b].spec.kind {
                            super::types::BlockKind::Phy(p) => Some(p.lanes),
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect();
                let width: u32 = phy_lanes.iter().map(|l| l.unwrap_or(s.io_width_bits)).sum();
                if phy_lanes.len() != phys.len() || (width != s.io_width_bits && !phy_lanes.is_empty()) {
                    self.push(
                        &key,
                        err("E-IR-0505", format!("stack '{}' needs PHYs totalling {} b; bound {} PHYs / {width} b", s.id, s.io_width_bits, phy_lanes.len()), &at),
                    );
                }
            }
            if let Some(StackAttach::Vertical { .. }) = &s.attach {
                let pkg = self.package_of(mem.node);
                let foreign = self.m.channels.iter().any(|c| {
                    c.src == NodeIx::Mem(mi) && c.kind == super::model::ChannelKind::Vertical && self.package_of(self.m.node_of(c.dst)) != pkg
                });
                if foreign {
                    self.push(&key, err("E-IR-0507", "stacked DRAM vertically attached to a die in another package", &at));
                }
            }
        }
    }

    fn stack_spec(&mut self, s: &MemStack, key: &str, at: &str) {
        if s.attach.is_none() {
            self.push(key, err("E-IR-0501", format!("mem stack '{}' has no attach", s.id), at).hint("set attach: { network: \"die.<noc>\" } or { phys: [...] }"));
        }
        if let Some((lo, hi, pin)) = s.kind.sanity() {
            if s.capacity.0 < lo || s.capacity.0 > hi {
                self.push(&format!("{key}#0502"), Diagnostic::warning("W-IR-0502", format!("capacity {} GiB outside the known range for {:?}", s.capacity.0 as f64 / 1073741824.0, s.kind)).at(at));
            }
            if s.pin_rate_bits_per_s.0 > pin * 1.05 {
                self.push(&format!("{key}#0503"), Diagnostic::warning("W-IR-0503", format!("pin rate {} Gb/s beyond {:?} maximum {} Gb/s", s.pin_rate_bits_per_s.0 / 1e9, s.kind, pin / 1e9)).at(at));
            }
        }
        if let Some(bw) = s.overrides.bandwidth
            && bw.0 > s.derived_bandwidth().0 * (1.0 + 1e-9)
        {
            self.push(key, err("E-IR-0506", format!("bandwidth override {} GB/s exceeds io_width*pin_rate/8 = {} GB/s", bw.0 / 1e9, s.derived_bandwidth().0 / 1e9), at));
        }
        if s.io_width_bits == 0 || s.pin_rate_bits_per_s.0 <= 0.0 {
            self.push(key, err("E-IR-0706", "mem stack io width or pin rate is zero", at));
        }
    }

    fn link_checks(&mut self, l: &LinkSpec, clock: Option<Hz>, key: &str, at: &str) {
        let zero = l.width_bits == Some(0)
            || match &l.phys {
                LinkPhys::D2d(d) => d.modules == 0 || d.pin_rate_bits_per_s.0 <= 0.0,
                LinkPhys::Serdes(s) => s.lanes == 0 || s.lane_rate_bits_per_s.0 <= 0.0,
                LinkPhys::Optical(o) => o.lanes == 0 || o.lane_rate_bits_per_s.0 <= 0.0,
                LinkPhys::OnDie { .. } => l.width_bits.is_none() && l.bandwidth.is_none(),
                LinkPhys::Vertical(_) => false,
            };
        if zero {
            self.push(key, err("E-IR-0706", "link has zero (or missing on-die) width, lanes or rate", at).hint("give width_bits for on-die links"));
            return;
        }
        let phy = match &l.phys {
            LinkPhys::OnDie { .. } | LinkPhys::Vertical(_) => None,
            _ => LinkSpec { bandwidth: None, ..l.clone() }.derived_bandwidth(clock),
        };
        if let (Some(w), Some(f), Some(p)) = (l.width_bits, clock, phy) {
            let wb = f64::from(w) * f.0 / 8.0;
            if (wb - p.0).abs() > 0.01 * p.0 {
                self.push(key, err("E-IR-0713", format!("width_bits {w} gives {} GB/s, PHY gives {} GB/s", wb / 1e9, p.0 / 1e9), at));
            }
        }
        if let (Some(bw), Some(d)) = (l.bandwidth, LinkSpec { bandwidth: None, ..l.clone() }.derived_bandwidth(clock))
            && bw.0 > d.0 * (1.0 + 1e-9)
        {
            self.push(key, err("E-IR-0712", format!("bandwidth override {} GB/s exceeds derived {} GB/s", bw.0 / 1e9, d.0 / 1e9), at));
        }
    }

    fn networks(&mut self) {
        for net in &self.m.networks {
            let n = &self.m.nodes[net.node];
            if !n.enabled {
                continue;
            }
            let (key, at) = (n.entity.clone(), n.path.clone());
            let s = net.spec.clone();
            let clock = self.m.clock_hz(net.clock);
            for l in s.link_specs() {
                self.link_checks(l, clock, &key, &at);
            }
            let dor = matches!(s.routing, Routing::Default | Routing::DimensionOrder { .. });
            if matches!(s.topology, Topology::Torus { .. } | Topology::Ring { .. }) && dor && s.router.vcs == Some(1) && !net.direct {
                self.push(&key, err("E-IR-0707", "dimension-order routing on a torus/ring needs >= 2 VCs (dateline)", &at).hint("set router.vcs: 2"));
            }
            if matches!(s.topology, Topology::Bus { .. }) && (s.features.multicast || s.features.broadcast) {
                self.push(&format!("{key}#0724"), Diagnostic::warning("W-IR-0724", "multicast/broadcast on a bus is redundant").at(&at));
            }
            if !s.features.in_network_reduce.is_empty()
                && let Topology::Star { .. } = s.topology
            {
                let switches: Vec<_> = self
                    .m
                    .tree
                    .iter()
                    .filter_map(|c| c.switch.as_ref())
                    .filter(|sw| s.features.in_network_reduce.iter().any(|p| !sw.in_network_reduce.contains(p)))
                    .collect();
                if !switches.is_empty() {
                    self.push(&key, err("E-IR-0711", "in_network_reduce precision unsupported by the switch", &at));
                }
            }
            if matches!(s.link.phys, LinkPhys::D2d(_)) {
                for c in self.m.channels.iter().filter(|c| c.network == Some(net_ix(self.m, net.node))) {
                    let pa = self.package_of(self.m.node_of(c.src));
                    let pb = self.package_of(self.m.node_of(c.dst));
                    if pa != pb {
                        self.push(&key, err("E-IR-0708", "d2d link between dies of different packages (use serdes)", &at));
                        break;
                    }
                }
            }
        }
        for p in &self.m.ports {
            let n = &self.m.nodes[p.node];
            if let Some(l) = &p.spec.link {
                let clock = self.m.tree[p.container].clock;
                self.link_checks(l, self.m.clock_hz(clock), &n.entity.clone(), &n.path.clone());
            }
            if let Some(phy) = &p.spec.phy {
                let target = self.m.blocks.iter().find(|b| {
                    let bn = &self.m.nodes[b.node];
                    bn.parent == n.parent && bn.entity_id == *phy
                });
                if let Some(b) = target
                    && let super::types::BlockKind::Phy(ps) = &b.spec.kind
                    && ps.for_kind != p.spec.kind
                {
                    self.push(&n.entity.clone(), err("E-IR-0709", format!("port phy '{phy}' is for {:?}, port is {:?}", ps.for_kind, p.spec.kind), &n.path.clone()));
                }
            }
        }
    }

    fn package_of(&self, mut n: usize) -> Option<usize> {
        loop {
            if let NodeIx::Container(c) = self.m.nodes[n].ix
                && self.m.tree[c].kind == ContainerKind::Package
            {
                return Some(n);
            }
            n = self.m.nodes[n].parent?;
        }
    }

    fn staging(&mut self) {
        let sources: Vec<usize> = self
            .m
            .memories
            .iter()
            .filter(|m| m.is_stack() && self.m.nodes[m.node].enabled)
            .map(|m| m.node)
            .chain(self.m.tree.iter().filter(|c| c.kind == ContainerKind::Host).map(|c| c.node))
            .collect();
        let fill = self.m.reachable_mems(&sources);
        // A near unit reads and writes its bound memory in place: it stages through it both ways.
        let staged = |u: &UnitInst| -> Vec<(OperandRole, usize, bool)> {
            let near = u.near.iter().flat_map(|n| [(OperandRole::In, n.mem, true), (OperandRole::Out, n.mem, true)]);
            u.feeds.iter().map(|(r, f)| (*r, f.mem, false)).chain(near).collect()
        };
        let readers: Vec<(usize, usize)> = (0..self.m.units.len())
            .flat_map(|ui| staged(&self.m.units[ui]).into_iter().filter(|(r, ..)| r.is_input()).map(move |(_, m, _)| (ui, m)))
            .collect();
        for (ui, u) in self.m.units.iter().enumerate() {
            let n = &self.m.nodes[u.node];
            if !n.enabled {
                continue;
            }
            for (role, fmem, near) in staged(u) {
                let mem = &self.m.memories[fmem];
                let mpath = self.m.nodes[mem.node].path.clone();
                if role.is_input() && !fill[mem.node] {
                    self.push(
                        &n.entity.clone(),
                        err(
                            "E-IR-0720",
                            format!("unit '{}' stages operand '{role:?}' through '{mpath}', which no network connects to off-chip memory or a host", n.path),
                            &n.path.clone(),
                        )
                        .hint(format!("add '{}' to the endpoints of a network that reaches a mem stack controller, or set its backing", self.m.nodes[mem.node].entity_id)),
                    );
                }
                if role.is_output() && !role.is_input() {
                    let reach = self.m.reachable_mems(&[mem.node]);
                    let drains = sources.iter().any(|&s| reach[s]) || readers.iter().any(|&(r, m)| m == fmem && !(near && r == ui));
                    if !drains {
                        self.push(&n.entity.clone(), err("E-IR-0721", format!("output memory '{mpath}' cannot drain to off-chip memory or another unit"), &n.path.clone()));
                    }
                }
            }
        }
    }

    fn floorplan(&mut self) {
        for (pkg, p) in self.packages() {
            for d in &p.dies {
                if let Some(l) = &d.layer {
                    let idx = p.layers.iter().find(|x| x.id.as_str() == l).map(|x| x.index);
                    if idx.is_some_and(|i| i > 0) && d.over.is_none() {
                        self.push(&format!("{pkg}.{}", d.id), err("E-IR-0809", format!("die '{}' on an upper layer needs `over`", d.id), &format!("{pkg}.{}", d.id)));
                    }
                }
                if matches!(d.placement, Placement::Array) && !matches!(d.rep.layout, Layout::Grid { .. }) {
                    self.push(&format!("{pkg}.{}", d.id), err("E-IR-0812", "placement: array needs a grid layout", &format!("{pkg}.{}", d.id)));
                }
                self.contents_placement(&d.contents, &format!("{pkg}.{}", d.id));
            }
            for s in &p.mem_stacks {
                if let Placement::Site { site } = &s.site
                    && !p.substrate.sites.iter().any(|x| x.id.as_str() == site)
                {
                    self.push(&format!("{pkg}.{}", s.id), err("E-IR-0504", format!("stack site '{site}' is not a package site"), &format!("{pkg}.{}", s.id)));
                }
            }
            for l in &p.links {
                if let LinkPhys::Vertical(_) = l.link.phys {
                    let layer_of = |r: &str| {
                        let die = r.split('[').next().unwrap_or(r).split('.').next().unwrap_or(r);
                        p.dies.iter().find(|d| d.id.as_str() == die).and_then(|d| d.layer.as_ref()).and_then(|ly| p.layers.iter().find(|x| x.id.as_str() == ly)).map(|x| x.index)
                    };
                    if let (Some(a), Some(b)) = (layer_of(&l.a), layer_of(&l.b))
                        && (a - b).abs() != 1
                    {
                        self.push(&format!("{pkg}.{}", l.id), err("E-IR-0810", "vertical link between non-adjacent layers", &format!("{pkg}.{}", l.id)));
                    }
                }
            }
        }
    }

    fn contents_placement(&mut self, c: &Contents, at: &str) {
        let check = |p: &Placement, layout: &Layout| -> Option<(&'static str, &'static str)> {
            match p {
                Placement::Pinned { x, y, .. } if x.0 < 0.0 || y.0 < 0.0 => Some(("E-IR-0811", "pinned coordinates are negative")),
                Placement::Region { x0, y0, x1, y1 } if x0.0 < 0.0 || y0.0 < 0.0 || x1.0 < x0.0 || y1.0 < y0.0 => {
                    Some(("E-IR-0811", "region box is negative or inverted"))
                }
                Placement::Array if !matches!(layout, Layout::Grid { .. }) => Some(("E-IR-0812", "placement: array needs a grid layout")),
                _ => None,
            }
        };
        let mut found = vec![];
        for cl in &c.clusters {
            found.extend(check(&cl.placement, &cl.rep.layout).map(|e| (e, cl.id.to_string())));
        }
        for u in &c.units {
            found.extend(check(&u.placement, &u.rep.layout).map(|e| (e, u.id.to_string())));
        }
        for m in &c.memories {
            found.extend(check(&m.placement, &m.rep.layout).map(|e| (e, m.id.to_string())));
        }
        for b in &c.blocks {
            found.extend(check(&b.placement, &b.rep.layout).map(|e| (e, b.id.to_string())));
        }
        for ((code, msg), id) in found {
            let p = format!("{at}.{id}");
            self.push(&p, err(code, msg, &p));
        }
        for cl in &c.clusters {
            self.contents_placement(&cl.contents, &format!("{at}.{}", cl.id));
        }
    }

    fn packages(&self) -> Vec<(String, super::types::Package)> {
        let mut v = vec![];
        for b in &self.doc.system.boards {
            for p in &b.packages {
                v.push((format!("{}.{}", b.id, p.id), p.clone()));
            }
        }
        v
    }

    fn system(&mut self) {
        let pkgs: Vec<usize> = self
            .m
            .tree
            .iter()
            .filter(|c| c.kind == ContainerKind::Package && self.m.nodes[c.node].enabled)
            .map(|c| c.node)
            .collect();
        if pkgs.len() > 1 {
            let connected = self.m.networks.iter().any(|n| {
                let mut ps: Vec<Option<usize>> = n.endpoints.iter().map(|&e| self.package_of(self.m.node_of(e))).collect();
                ps.sort_unstable();
                ps.dedup();
                n.direct || ps.len() > 1
            });
            if !connected {
                self.push("system#0723", Diagnostic::warning("W-IR-0723", format!("{} chips but no chip-to-chip network", pkgs.len())).hint("add an ICI/NVLink network over package ports"));
            }
        }
        let hosts = self.m.tree.iter().any(|c| c.kind == ContainerKind::Host);
        if hosts {
            for &p in &pkgs {
                let linked = self.m.channels.iter().any(|c| c.kind == super::model::ChannelKind::Host && self.ancestor_or_self(p, self.m.node_of(c.dst)));
                if !linked {
                    let path = self.m.nodes[p].path.clone();
                    self.push("system#0725", Diagnostic::warning("W-IR-0725", format!("chip '{path}' has no host link")).at(&path));
                }
            }
        }
    }

    fn claims(&mut self) {
        for c in &self.doc.meta.claims {
            let scope: Option<Vec<usize>> = if c.scope.is_empty() {
                None
            } else {
                match self.m.instances(&c.scope) {
                    Ok(v) if !v.is_empty() => Some(v.into_iter().map(|ix| self.m.node_of(ix)).collect()),
                    _ => {
                        self.push(&format!("claim:{}", c.metric), err("E-IR-0205", format!("claim scope {:?} matches nothing", c.scope), "meta.claims"));
                        continue;
                    }
                }
            };
            let derived = match c.metric.split_once('.') {
                Some(("peak_ops", p)) => Precision::from_name(p).map(|p| self.m.peak_ops_for(p, scope.as_deref())),
                Some(("onchip_bytes", k)) => {
                    let kind = serde_json::from_value::<MemKind>(serde_json::Value::from(k)).ok();
                    Some(self.m.onchip_capacity(scope.as_deref(), kind).as_f64())
                }
                None if c.metric == "offchip_bw" => Some(self.m.offchip_bandwidth(scope.as_deref()).0),
                None if c.metric == "offchip_bytes" => Some(self.m.offchip_capacity(scope.as_deref()).as_f64()),
                _ => None,
            };
            if let Some(v) = derived
                && (v - c.value).abs() > c.rel_tol * c.value.abs()
            {
                self.push(
                    &format!("claim:{}:{}", c.metric, c.scope),
                    Diagnostic::warning("W-IR-1801", format!("{} derived {v:.4e} vs claimed {:.4e} (src:{})", c.metric, c.value, c.source))
                        .at("meta.claims")
                        .hint("check count/geometry/clock/pin rate against the source"),
                );
            }
        }
    }

    fn profiles(&mut self) {
        match self.profile {
            Profile::Full => {}
            Profile::Reference => self.reference(),
            Profile::Search => self.search(),
            Profile::StreamCompat => self.stream_compat(),
        }
    }

    /// Every override-bearing entity, as (entity key, path, overridden fields, source, bandwidth-only?).
    fn overrides(&self) -> Vec<(String, String, Vec<&'static str>, Option<String>)> {
        let mut out = vec![];
        let po = |p: &PowerOverride| -> Vec<&'static str> {
            let mut v = vec![];
            if p.area.is_some() {
                v.push("power.area");
            }
            if !p.energy_per_op.is_empty() {
                v.push("power.energy_per_op");
            }
            if p.leakage.is_some() {
                v.push("power.leakage");
            }
            if p.ctrl_ge.is_some() {
                v.push("power.ctrl_ge");
            }
            v
        };
        let link = link_overrides;
        for u in &self.m.units {
            let n = &self.m.nodes[u.node];
            out.push((n.entity.clone(), n.path.clone(), po(&u.spec.power), u.spec.power.source.clone()));
        }
        for m in &self.m.memories {
            let n = &self.m.nodes[m.node];
            match &m.spec {
                MemSpec::OnChip(s) => out.push((n.entity.clone(), n.path.clone(), s.overrides.performance_fields(), s.overrides.source.clone())),
                MemSpec::Stack(s) => {
                    let o = &s.overrides;
                    let mut f = vec![];
                    if o.bandwidth.is_some() {
                        f.push("overrides.bandwidth");
                    }
                    if o.energy_per_byte.is_some() {
                        f.push("overrides.energy_per_byte");
                    }
                    if o.latency.is_some() {
                        f.push("overrides.latency");
                    }
                    if o.power.is_some() {
                        f.push("overrides.power");
                    }
                    out.push((n.entity.clone(), n.path.clone(), f, o.source.clone()));
                }
                MemSpec::Local { .. } => {}
            }
        }
        for b in &self.m.blocks {
            let n = &self.m.nodes[b.node];
            out.push((n.entity.clone(), n.path.clone(), po(&b.spec.power), b.spec.power.source.clone()));
        }
        for net in &self.m.networks {
            let n = &self.m.nodes[net.node];
            for l in net.spec.link_specs() {
                out.push((n.entity.clone(), n.path.clone(), link(l), l.source.clone()));
            }
            out.push((n.entity.clone(), n.path.clone(), po(&net.spec.router.power), net.spec.router.power.source.clone()));
        }
        for p in &self.m.ports {
            let n = &self.m.nodes[p.node];
            if let Some(l) = &p.spec.link {
                out.push((n.entity.clone(), n.path.clone(), link(l), l.source.clone()));
            }
        }
        for (pkg, p) in self.packages() {
            for l in &p.links {
                out.push((format!("{pkg}.{}", l.id), format!("{pkg}.{}", l.id), link(&l.link), l.link.source.clone()));
            }
            for d in &p.dies {
                out.push((format!("{pkg}.{}", d.id), format!("{pkg}.{}", d.id), po(&d.power), d.power.source.clone()));
            }
        }
        out.retain(|(_, _, f, _)| !f.is_empty());
        out
    }

    fn reference(&mut self) {
        let citations = &self.doc.meta.citations;
        for (key, at, fields, source) in self.overrides() {
            if source.as_ref().is_none_or(|s| !citations.contains_key(s)) {
                self.push(
                    &format!("{key}#1103"),
                    err("E-IR-1103", format!("override of {fields:?} without a `source` citation"), &at)
                        .hint("add source: \"<meta.citations id>\" next to the override"),
                );
            }
        }
        let claims = &self.doc.meta.claims;
        let has_peak = claims.iter().any(|c| c.metric.starts_with("peak_ops."));
        let has_bw = claims.iter().any(|c| c.metric == "offchip_bw");
        if !has_peak || !has_bw {
            self.push(
                "meta#1103",
                err("E-IR-1103", "reference profile needs meta.claims for peak_ops.<precision> and offchip_bw", "meta.claims")
                    .hint("add claims with published values and their citation ids"),
            );
        }
    }

    fn search(&mut self) {
        if let Some(f) = &self.doc.family {
            self.push(
                "family",
                err("E-IR-1102", format!("family '{f}' set; platform calibration residuals cannot be inherited by novel designs"), "family")
                    .hint("remove `family`"),
            );
        }
        for pd in self.m.power_domains.iter().filter(|p| p.assumed.is_some()) {
            self.push(
                &pd.path,
                err("E-IR-1106", "assumed (unpublished) power cap: reference designs only; a novel design states its cap", &pd.path)
                    .hint("remove `assumed` and give the cap the design is held to"),
            );
        }
        let mut priced: Vec<(String, &'static str)> = vec![];
        if let Some(pr) = self.pricer {
            for f in self.priced_fields() {
                let Some(d) = pr.derived(&f) else { continue };
                let n = &self.m.nodes[f.node];
                priced.push((n.path.clone(), f.field));
                if f.value < d * (1.0 - 1e-9) {
                    let name = f.key.as_ref().map_or_else(|| f.field.to_string(), |k| format!("{}.{k}", f.field));
                    self.push(
                        &format!("{}#1101{}", n.entity, f.field),
                        err("E-IR-1101", format!("{name} override {:.4e} below derived {d:.4e} (ratio {:.2})", f.value, f.value / d), &n.path)
                            .hint("energy, area, leakage, power and latency below the kiln-phys derived value are unpriced; values >= derived are allowed"),
                    );
                }
            }
        }
        for (key, at, fields, _) in self.overrides() {
            let fields: Vec<&str> =
                fields.into_iter().filter(|f| !f.ends_with("bandwidth") && !priced.iter().any(|(p, x)| *p == at && x == f)).collect();
            if !fields.is_empty() {
                self.push(&format!("{key}#1101"), unverifiable(&fields, &at));
            }
        }
        self.bandwidth_overrides();
        for u in &self.m.units {
            let n = &self.m.nodes[u.node];
            let s = &u.spec;
            let mut f = vec![];
            if let Some(near) = &s.near {
                if near.internal_bandwidth.is_some() {
                    f.push("near.internal_bandwidth");
                }
                if near.command_latency.is_some() {
                    f.push("near.command_latency");
                }
            }
            let derived_fill = match &s.kind {
                ComputeKind::Matrix(mx) => match mx.geometry {
                    Geometry::Systolic { rows, cols } => Some(f64::from(rows + cols) - 1.0),
                    _ => None,
                },
                _ => None,
            };
            for (name, v) in [("pipeline.fill", s.pipeline.fill), ("pipeline.drain", s.pipeline.drain)] {
                match (v, derived_fill) {
                    (Some(v), Some(d)) if v.0 < d => self.push(
                        &format!("{}#1101p", n.entity),
                        err("E-IR-1101", format!("{name} override {} cycles below derived {d} (ratio {:.2})", v.0, v.0 / d), &n.path)
                            .hint("latency below the structure-derived value is unpriced; values >= derived are allowed"),
                    ),
                    (Some(_), None) => f.push(name),
                    _ => {}
                }
            }
            for (name, r, d) in rate_multipliers(s) {
                if r > d * (1.0 + 1e-9) {
                    self.push(
                        &format!("{}#1101r{name}", n.entity),
                        err("E-IR-1101", format!("{name} override {r} exceeds the priced default {d} (ratio {:.2})", r / d), &n.path)
                            .hint("the datapath is priced for the default per-class rate; raise lanes or the mode @rate instead (de-rates are allowed)"),
                    );
                }
            }
            if s.feeds.values().any(|fd| fd.latency.is_some()) {
                f.push("feeds.latency");
            }
            if matches!(&s.kind, ComputeKind::Cim(c) if c.weight_write.is_some()) {
                f.push("weight_write");
            }
            if let ComputeKind::Cim(c) = &s.kind
                && let (CimStyle::Analog, Some(adc)) = (c.style, c.adc_bits)
                && adc < c.boundary_adc_bits()
            {
                self.push(
                    &format!("{}#1107", n.entity),
                    err("E-IR-1107", format!("analog CIM adc_bits {adc} below the lossless {} bits: accuracy loss is unmodeled", c.boundary_adc_bits()), &n.path)
                        .hint("raise adc_bits to the lossless boundary or lower parallel_rows / input_bits_per_cycle / cell_bits"),
                );
            }
            if !f.is_empty() {
                self.push(&format!("{}#1101", n.entity), unverifiable(&f, &n.path));
            }
        }
        let mut extra: Vec<(usize, Vec<&'static str>)> = vec![];
        for net in &self.m.networks {
            extra.push((net.node, net.spec.link_specs().flat_map(phys_overrides).collect()));
        }
        extra.extend(self.m.ports.iter().filter_map(|p| p.spec.link.as_ref().map(|l| (p.node, phys_overrides(l)))));
        for c in &self.m.tree {
            if let Some(sw) = &c.switch {
                let mut f = phys_overrides(&sw.port);
                f.extend(link_overrides(&sw.port).into_iter().filter(|f| !f.ends_with("bandwidth")));
                f.extend([("latency", sw.latency.is_some()), ("reduce_bandwidth", sw.reduce_bandwidth.is_some()), ("power", sw.power.is_some())].into_iter().filter(|x| x.1).map(|x| x.0));
                extra.push((c.node, f));
            }
        }
        for b in &self.m.blocks {
            if let super::types::BlockKind::Dma(d) = &b.spec.kind
                && d.bandwidth.is_some()
            {
                extra.push((b.node, vec!["bandwidth"]));
            }
        }
        for (node, f) in extra.into_iter().filter(|(_, f)| !f.is_empty()) {
            let n = &self.m.nodes[node];
            self.push(&format!("{}#1101x", n.entity), unverifiable(&f, &n.path));
        }
        for c in self.m.clocks.iter().filter(|c| c.spec.crossing_latency.is_some()) {
            self.push(&format!("{}#1101", c.path), unverifiable(&["crossing_latency"], &c.path));
        }
        for net in &self.m.networks {
            if net.spec.router.pipeline.is_some_and(|p| p.0 < 2.0) {
                let n = &self.m.nodes[net.node];
                self.push(&format!("{}#1101", n.entity), err("E-IR-1101", "router.pipeline below the 2-cycle default is unpriced", &n.path));
            }
        }
        for u in &self.m.units {
            let ComputeKind::Matrix(mx) = &u.spec.kind else { continue };
            let n = &self.m.nodes[u.node];
            for sp in &mx.sparsity {
                let unpriced = match sp.min_metadata_bits() {
                    None => Some(format!("sparsity {:?} has no structural speedup bound kiln-phys can price", sp.pattern)),
                    Some(b) if sp.metadata_bits_per_nz < b => {
                        Some(format!("sparsity {:?} declares {} metadata bits per nonzero; its index needs {b}", sp.pattern, sp.metadata_bits_per_nz))
                    }
                    Some(_) => None,
                };
                if let Some(msg) = unpriced {
                    self.push(
                        &format!("{}#1104sparsity", n.entity),
                        err("E-IR-1104", msg, &n.path).hint("use an n:m pattern with speedup <= m/n and metadata_bits_per_nz >= log2(m)"),
                    );
                }
            }
        }
        for m in &self.m.memories {
            if let MemSpec::Stack(s) = &m.spec
                && s.kind == super::compute::DramKind::Custom
                && s.timing.is_none()
            {
                let n = &self.m.nodes[m.node];
                self.push(&n.entity.clone(), err("E-IR-1104", "custom DRAM kind without timing data cannot be priced by kiln-phys", &n.path.clone()).hint("use a JEDEC kind or give `timing`"));
            }
        }
    }

    /// Every "less is better" override kiln-phys can price, with its value in base units.
    fn priced_fields(&self) -> Vec<PricedField> {
        let mut out = vec![];
        let mut po = |node: usize, p: &PowerOverride| {
            let mut push = |field, key: Option<String>, value: f64| out.push(PricedField { node, field, key, value });
            if let Some(a) = p.area {
                push("power.area", None, a.0 * 1e6);
            }
            for (k, e) in &p.energy_per_op {
                push("power.energy_per_op", Some(k.clone()), e.0);
            }
            if let Some(l) = p.leakage {
                push("power.leakage", None, l.0);
            }
            if let Some(g) = p.ctrl_ge {
                push("power.ctrl_ge", None, g);
            }
        };
        for u in &self.m.units {
            po(u.node, &u.spec.power);
        }
        for b in &self.m.blocks {
            po(b.node, &b.spec.power);
        }
        for net in &self.m.networks {
            po(net.node, &net.spec.router.power);
        }
        for c in &self.m.tree {
            if let Some(d) = &c.die {
                po(c.node, &d.power);
            }
        }
        for m in &self.m.memories {
            let mut push = |field, value: f64| out.push(PricedField { node: m.node, field, key: None, value });
            match &m.spec {
                MemSpec::OnChip(s) => {
                    let o = &s.overrides;
                    o.read_energy.into_iter().for_each(|e| push("read_energy", e.0));
                    o.write_energy.into_iter().for_each(|e| push("write_energy", e.0));
                    o.latency.into_iter().for_each(|l| push("latency", l.0));
                    o.leakage.into_iter().for_each(|l| push("leakage", l.0));
                    o.area.into_iter().for_each(|a| push("area", a.0 * 1e6));
                }
                MemSpec::Stack(s) => {
                    let o = &s.overrides;
                    o.energy_per_byte.into_iter().for_each(|e| push("overrides.energy_per_byte", e.0));
                    o.power.into_iter().for_each(|p| push("overrides.power", p.0));
                }
                MemSpec::Local { .. } => {}
            }
        }
        let mut link = |node: usize, l: &LinkSpec| {
            l.latency.into_iter().for_each(|x| out.push(PricedField { node, field: "link.latency", key: None, value: x.0 }));
            l.energy.into_iter().for_each(|x| out.push(PricedField { node, field: "link.energy", key: None, value: x.0 }));
        };
        for net in &self.m.networks {
            net.spec.link_specs().for_each(|l| link(net.node, l));
        }
        for p in &self.m.ports {
            if let Some(l) = &p.spec.link {
                link(p.node, l);
            }
        }
        out
    }

    /// 00 decision 3: a bandwidth override at or below the derived value is a cost-neutral de-rate; above it, or
    /// where kiln-ir cannot derive the value, it is unpriced.
    fn bandwidth_overrides(&mut self) {
        let mut found: Vec<(usize, &'static str, f64, Option<f64>)> = vec![];
        for m in &self.m.memories {
            let o = match &m.spec {
                MemSpec::OnChip(s) => s.overrides.bandwidth.map(|b| ("bandwidth", b)),
                MemSpec::Stack(s) => s.overrides.bandwidth.map(|b| ("overrides.bandwidth", b)),
                MemSpec::Local { .. } => None,
            };
            if let Some((field, b)) = o {
                found.push((m.node, field, b.0, m.bandwidth_derived.map(|d| d.0)));
            }
        }
        let is_stack = |ix: NodeIx| matches!(ix, NodeIx::Mem(i) if self.m.memories[i].is_stack());
        for c in &self.m.channels {
            let stack_link = is_stack(c.src)
                || is_stack(c.dst)
                || (c.kind == ChannelKind::MemPort && matches!((c.src, c.dst), (NodeIx::Block(_), NodeIx::Block(_))));
            let Some(bw) = c.bandwidth else { continue };
            if c.kind == ChannelKind::Near || stack_link || c.bandwidth_derived.is_some_and(|d| bw.0 == d.0) {
                continue;
            }
            let owner = c.network.map_or(self.m.node_of(c.src), |n| self.m.networks[n].node);
            found.push((owner, "link.bandwidth", bw.0, c.bandwidth_derived.map(|d| d.0)));
        }
        for (node, field, o, d) in found {
            let n = &self.m.nodes[node];
            let (key, at) = (format!("{}#1101b", n.entity), n.path.clone());
            match d {
                None => self.push(&key, unverifiable(&[field], &at)),
                Some(d) if o > d * (1.0 + 1e-9) => self.push(
                    &key,
                    err("E-IR-1101", format!("{field} override {} GB/s exceeds derived {} GB/s (ratio {:.2})", o / 1e9, d / 1e9, o / d), &at)
                        .hint("performance above the structure-derived value is unpriced; widen ports/links or raise pin rate instead (de-rates <= derived are allowed)"),
                ),
                Some(_) => {}
            }
        }
    }

    fn stream_compat(&mut self) {
        let mut first: Option<(String, String)> = None;
        let mut note = |at: &str, why: String| {
            if first.is_none() {
                first = Some((at.to_owned(), why));
            }
        };
        let pkgs = self.m.tree.iter().filter(|c| c.kind == ContainerKind::Package).count();
        let dies = self.m.tree.iter().filter(|c| c.kind == ContainerKind::Die).count();
        if pkgs != 1 || dies != 1 {
            note("system", format!("{pkgs} packages / {dies} dies (stream_compat is 1 chip, 1 die)"));
        }
        for u in &self.m.units {
            let n = &self.m.nodes[u.node];
            let s = &u.spec;
            let ok_kind = matches!(s.kind, ComputeKind::Matrix(_) | ComputeKind::Vector(_));
            let mut mems: Vec<usize> = u.feeds.values().map(|f| f.mem).collect();
            mems.dedup();
            let ok_mode = match (&s.kind, s.precisions.as_slice()) {
                (ComputeKind::Matrix(_), [PrecisionMode::Mac { a, b, acc, .. }]) => {
                    a.precision == Precision::Bf16 && b.precision == Precision::Bf16 && acc.precision == Precision::Fp32
                }
                (ComputeKind::Vector(_), [PrecisionMode::Elem { .. }]) => true,
                _ => false,
            };
            if !ok_kind || !ok_mode || mems.len() != 1 || u.near.is_some() {
                note(&n.path, format!("unit '{}' is not a single-precision matrix/vector unit fed by one memory (bf16*bf16+fp32 matrix modes only)", s.id));
            }
        }
        let max_level = self.m.memories.iter().enumerate().filter(|(_, m)| !m.is_stack() && !m.is_local()).map(|(i, _)| self.m.levels[i]).filter(|&l| l != u8::MAX).max().unwrap_or(0);
        if max_level > 2 {
            note("memories", format!("{max_level} on-chip memory levels (stream_compat allows 2)"));
        }
        for net in &self.m.networks {
            if !matches!(net.spec.topology, Topology::Bus { .. } | Topology::P2p) {
                note(&self.m.nodes[net.node].path, format!("network topology {} (stream_compat allows bus/p2p)", net.spec.topology.name()));
            }
        }
        if let Some((at, why)) = first {
            self.push("stream_compat", err("E-IR-1105", format!("not inside the stream_compat subset: {why}"), &at));
        }
    }
}

/// A performance field kiln-ir cannot derive yet, so the <=/>= derived rule (01 §18.3) cannot be checked.
fn unverifiable(fields: &[&str], at: &str) -> Diagnostic {
    err("E-IR-1101", format!("unverifiable performance override {fields:?}: no derived value to compare against"), at)
        .hint("remove it; it becomes allowed once kiln-phys derives it (throughput <= derived, latency/energy/area/power >= derived)")
}

fn link_overrides(l: &LinkSpec) -> Vec<&'static str> {
    [("link.latency", l.latency.is_some()), ("link.energy", l.energy.is_some()), ("link.bandwidth", l.bandwidth.is_some())]
        .into_iter()
        .filter(|x| x.1)
        .map(|x| x.0)
        .collect()
}

/// PHY constants 04 tabulates (§4.6, §4.10); an explicit value is an unpriced override.
fn phys_overrides(l: &LinkSpec) -> Vec<&'static str> {
    match &l.phys {
        LinkPhys::Serdes(s) => [("phys.encoding_efficiency", s.encoding_efficiency.is_some()), ("phys.fec_latency", s.fec_latency.is_some())]
            .into_iter()
            .filter(|x| x.1)
            .map(|x| x.0)
            .collect(),
        LinkPhys::Optical(o) if o.switch_latency.is_some() => vec!["phys.switch_latency"],
        _ => vec![],
    }
}

/// Per-class (vector `class_rates`) and per-function (special `fn_rates`) multipliers as (field, value, priced
/// default): kiln-phys prices the datapath at lanes x mode rate, so a multiplier above its default is unpriced.
fn rate_multipliers(s: &ComputeUnit) -> Vec<(String, f64, f64)> {
    match &s.kind {
        ComputeKind::Vector(v) => v.class_rates.iter().map(|(c, &r)| (format!("class_rates.{}", snake(c)), r, c.default_vector_rate())).collect(),
        ComputeKind::Special(sp) => sp.fn_rates.iter().map(|(f, &r)| (format!("fn_rates.{}", snake(f)), r, 1.0)).collect(),
        _ => vec![],
    }
}

fn snake<T: Serialize>(t: &T) -> String {
    serde_json::to_value(t).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default()
}

fn m_a(m: &PrecisionMode) -> Option<crate::precision::PrecisionSpec> {
    match m {
        PrecisionMode::Mac { a, .. } => Some(*a),
        _ => None,
    }
}

fn m_b(m: &PrecisionMode) -> Option<crate::precision::PrecisionSpec> {
    match m {
        PrecisionMode::Mac { b, .. } => Some(*b),
        _ => None,
    }
}

fn net_ix(m: &HwModel, node: usize) -> usize {
    match m.nodes[node].ix {
        NodeIx::Net(n) => n,
        _ => usize::MAX,
    }
}

fn mem_tech(m: &HwModel, mut n: usize) -> String {
    loop {
        if let NodeIx::Container(c) = m.nodes[n].ix
            && let Some(t) = &m.tree[c].tech
        {
            return t.clone();
        }
        match m.nodes[n].parent {
            Some(p) => n = p,
            None => return String::new(),
        }
    }
}
