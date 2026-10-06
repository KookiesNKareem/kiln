//! Temporal search (03 §2.4): loop orders built innermost-first as a DFS over multiset-permutation prefixes,
//! with LOMA-style greedy bottom-up allocation, capacity-share and double-buffer variants, and exact
//! prunings: (1) once every non-top level boundary of every stream is fixed, the remaining loops matter only
//! through stationarity, so one order per arrangement of the remaining dim groups is costed; (2) a subtree
//! whose lower bound (compute, or the port traffic of its fixed boundaries under the most favorable
//! stationarity) exceeds the best cost found is skipped; (3) a prefix whose signature matches a fully explored
//! one is skipped: every leaf below it is an equivalence-key duplicate, never evaluated or counted against the
//! budget, so the result (truncation included) is unchanged.

use std::collections::BTreeSet;

use kiln_ir::common::Diagnostic;

use crate::model::{ClassEval, MAX_DIMS, OrderCtx, Prep, SpatialCtx, evaluate_with, infeasible};
use crate::search::{better, objective_value};
use crate::types::*;

pub(crate) struct SearchCfg {
    pub obj: Objective,
    pub budget: SearchBudget,
    /// Best objective so far over earlier spatial candidates (prunes when the class is the whole nest).
    pub bound: Option<f64>,
    /// (compute-cycle floor, MAC-energy floor) used to normalize weighted objectives.
    pub floors: (f64, f64),
    /// Latency floor at which the search stops (`SearchBudget::stop_at_floor`).
    pub stop: Option<f64>,
}

#[derive(Clone)]
pub(crate) struct ClassResult {
    pub eval: ClassEval,
    pub tm: TemporalMapping,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Share {
    Even,
    Proportional,
    OutputFirst,
    /// At the output's outermost on-chip level, the output gets half of what is left (all of it when no input
    /// shares the level) and the inputs split the rest evenly: room for an output-stationary tile next to streamed
    /// input tiles. Other levels split evenly.
    OutputHalf,
}

/// A port as the bound sees it: a key unique per (level, port) and its width in bits per cycle.
#[derive(Clone, Copy)]
struct PortRef {
    key: u32,
    bits_int: u64,
    bits_f: f64,
}

impl PortRef {
    const NONE: PortRef = PortRef { key: u32::MAX, bits_int: 0, bits_f: 0.0 };

    fn new(level: LevelIx, port: usize, bytes_per_cycle: f64) -> PortRef {
        let b = bytes_per_cycle * 8.0;
        let bits_int = if b.fract() == 0.0 && (1.0..1.8e19).contains(&b) { b as u64 } else { 0 };
        PortRef { key: (level * 64 + port) as u32, bits_int, bits_f: b }
    }

    fn cycles(&self, amount: u64, bits: u32) -> u64 {
        if self.key == u32::MAX {
            return 0;
        }
        let num = amount * u64::from(bits);
        if self.bits_int > 0 { num.div_ceil(self.bits_int) } else { (num as f64 / self.bits_f).ceil() as u64 }
    }
}

struct Variant {
    /// Per stream, per chain level: byte budget per instance (non-top levels).
    budget: Vec<Vec<u64>>,
    /// Per stream, per chain level: 2 when double buffered.
    mult: Vec<Vec<u64>>,
}

const OPEN: usize = usize::MAX;
/// Distinct loop items: at most the coarsened prime-factor count, or one per dim when more dims than that.
const LPF_KIDS: usize = if crate::search::LPF_LIMIT > MAX_DIMS { crate::search::LPF_LIMIT } else { MAX_DIMS };

struct Dfs<'a> {
    p: &'a Prep<'a>,
    sc: &'a SpatialCtx,
    sizes: &'a [u64],
    cfg: &'a SearchCfg,
    nl: usize,
    ttot: u64,
    issue: f64,
    e_floor: f64,
    types: Vec<TemporalLoop>,
    left: Vec<usize>,
    prefix: Vec<TemporalLoop>,
    tp: Vec<[u64; MAX_DIMS]>,
    cyc: Vec<u64>,
    /// Offset of stream `si`'s chain levels in per-(stream, level) rows; `nsl` slots per row.
    slot: Vec<usize>,
    nsl: usize,
    /// Row `k`: single-buffered tile bytes per (stream, level) holding loops `[0, k)`.
    tiles: Vec<u64>,
    /// Same layout: tile footprints in elements.
    el: Vec<u64>,
    full: Vec<u64>,
    full_el: Vec<u64>,
    /// Per (stream, level) slot, per direction: the port the transfer uses.
    pref: Vec<[PortRef; 4]>,
    /// Row `k`: greedy boundaries fixed by the length-`k` prefix (`OPEN` where later loops still decide).
    bb: Vec<usize>,
    /// Per stream: product of unplaced loops irrelevant to it.
    ir_left: Vec<u64>,
    var: Variant,
    best: Option<(ClassEval, TemporalMapping)>,
    /// Best objective among this pass's leaves.
    best_val: Option<f64>,
    /// The current variant double buffers wherever it can.
    buffered: bool,
    seen: BTreeSet<Vec<u64>>,
    /// Signatures of fully explored prefixes (current variant).
    done: BTreeSet<Vec<u64>>,
    keybuf: Vec<u64>,
    evals: u64,
    stats: SearchStats,
    first_err: Option<Diagnostic>,
    /// Output-stationary pass ([`os_items`]): the output's tile at its outermost on-chip level, per dim.
    os: Option<OsTile>,
}

impl Dfs<'_> {
    fn fill_tiles(&mut self, row: usize, tp: &[u64; MAX_DIMS]) {
        for si in 0..self.p.streams.len() {
            let s = &self.p.streams[si];
            let n = self.p.chains[si].len();
            for j in 0..n {
                let mut e = *tp;
                for (x, y) in e.iter_mut().zip(&self.sc.sp[si][j]) {
                    *x *= y;
                }
                let el = s.footprint(&e);
                let bits = if s.is_output && j + 1 < n { s.acc_bits.max(s.bits) } else { s.bits };
                let ix = row * self.nsl + self.slot[si] + j;
                self.el[ix] = el;
                self.tiles[ix] = (el * u64::from(bits)).div_ceil(8);
            }
        }
    }

    /// Row `row` from row `row - 1` after appending loop `l`: simple streams scale by the factor when the loop
    /// is relevant; others recompute.
    fn step_tiles(&mut self, row: usize, l: TemporalLoop) {
        let tp = self.tp[row];
        for si in 0..self.p.streams.len() {
            let s = &self.p.streams[si];
            let n = self.p.chains[si].len();
            let f = if s.relevant(l.dim) { l.factor } else { 1 };
            for j in 0..n {
                let ix = row * self.nsl + self.slot[si] + j;
                let el = if s.simple {
                    self.el[ix - self.nsl] * f
                } else {
                    let mut e = tp;
                    for (x, y) in e.iter_mut().zip(&self.sc.sp[si][j]) {
                        *x *= y;
                    }
                    s.footprint(&e)
                };
                let bits = if s.is_output && j + 1 < n { s.acc_bits.max(s.bits) } else { s.bits };
                self.el[ix] = el;
                self.tiles[ix] = (el * u64::from(bits)).div_ceil(8);
            }
        }
    }

    fn tile(&self, k: usize, si: usize, j: usize) -> u64 {
        self.tiles[k * self.nsl + self.slot[si] + j]
    }

    fn full(&self, si: usize, j: usize) -> u64 {
        self.full[self.slot[si] + j]
    }

    fn b(&self, row: usize, si: usize, j: usize) -> usize {
        self.bb[row * self.nsl + self.slot[si] + j]
    }

    fn variant(&self, share: Share, dbl: bool) -> Result<Variant, Diagnostic> {
        let p = self.p;
        let unit = p.unit;
        let ns = p.streams.len();
        let mult: Vec<Vec<u64>> = p
            .chains
            .iter()
            .map(|c| {
                let n = c.len();
                c.iter().enumerate().map(|(j, &l)| if dbl && unit.levels[l].double_buffer && j + 1 < n { 2 } else { 1 }).collect()
            })
            .collect();
        let mut budget: Vec<Vec<u64>> = p.chains.iter().map(|c| vec![u64::MAX; c.len()]).collect();
        for l in 0..unit.levels.len() {
            let lv = &unit.levels[l];
            let n_inst: u64 = lv.instance_axes.iter().map(|&a| u64::from(unit.axes[a].size)).product::<u64>().max(1);
            let mut shares: Vec<(usize, u64)> = (0..ns).filter(|&si| p.chains[si].contains(&l)).map(|si| lv.share(p.streams[si].role)).collect();
            shares.sort_unstable();
            shares.dedup();
            for (key, cap) in shares {
                let cap = cap / n_inst;
                let mut fixed = 0u64;
                let mut users: Vec<(usize, usize)> = vec![];
                for si in 0..ns {
                    for (j, _) in p.chains[si].iter().enumerate().filter(|&(_, &x)| x == l && lv.share(p.streams[si].role).0 == key) {
                        if j + 1 == p.chains[si].len() {
                            fixed += self.full(si, j);
                        } else {
                            users.push((si, j));
                        }
                    }
                }
                if fixed > cap {
                    return Err(infeasible(lv, fixed, cap, None));
                }
                let rem = cap - fixed;
                if users.is_empty() {
                    continue;
                }
                let full: Vec<u64> = users.iter().map(|&(si, j)| (self.full(si, j) * mult[si][j]).max(1)).collect();
                let n = users.len() as u64;
                match share {
                    Share::Even => users.iter().for_each(|&(si, j)| budget[si][j] = rem / n),
                    Share::Proportional => {
                        let tot: u128 = full.iter().map(|&x| u128::from(x)).sum();
                        for (i, &(si, j)) in users.iter().enumerate() {
                            budget[si][j] = (u128::from(rem) * u128::from(full[i]) / tot) as u64;
                        }
                    }
                    Share::OutputHalf if !users.iter().any(|&(si, j)| p.streams[si].is_output && j + 2 == p.chains[si].len()) => {
                        users.iter().for_each(|&(si, j)| budget[si][j] = rem / n)
                    }
                    Share::OutputHalf => {
                        let ins = users.iter().filter(|u| !p.streams[u.0].is_output).count() as u64;
                        let out_budget = if ins == 0 { rem } else { rem / 2 };
                        let outs = users.iter().filter(|u| p.streams[u.0].is_output).count() as u64;
                        for (i, &(si, j)) in users.iter().enumerate() {
                            budget[si][j] = if p.streams[si].is_output {
                                full[i].min(out_budget / outs)
                            } else {
                                (rem - out_budget) / ins
                            };
                        }
                    }
                    Share::OutputFirst => {
                        let outs: Vec<usize> = (0..users.len()).filter(|&i| p.streams[users[i].0].is_output).collect();
                        let mut left = rem;
                        for &i in &outs {
                            let g = full[i].min(left);
                            budget[users[i].0][users[i].1] = g;
                            left -= g;
                        }
                        let ins = (users.len() - outs.len()).max(1) as u64;
                        for (i, &(si, j)) in users.iter().enumerate() {
                            if !outs.contains(&i) {
                                budget[si][j] = left / ins;
                            }
                        }
                    }
                }
            }
        }
        for (si, row) in budget.iter().enumerate() {
            if self.full(si, 0) * mult[si][0] > row[0] && self.tile(0, si, 0) * mult[si][0] > row[0] {
                let l = p.chains[si][0];
                return Err(infeasible(&unit.levels[l], self.tile(0, si, 0) * mult[si][0], row[0], None));
            }
        }
        Ok(Variant { budget, mult })
    }

    /// Whether the variant's budgets can bind at all (otherwise other variants allocate identically or worse).
    fn binds(&self) -> bool {
        (0..self.p.streams.len())
            .any(|si| (0..self.p.chains[si].len() - 1).any(|j| self.full(si, j) * self.var.mult[si][j] > self.var.budget[si][j]))
    }

    /// Fixes the greedy boundaries implied by the current prefix into row `prefix.len()` of `bb`; returns
    /// whether every boundary is fixed.
    fn bounds(&mut self) -> bool {
        let k = self.prefix.len();
        let nl = self.nl;
        let mut all = true;
        for si in 0..self.p.streams.len() {
            let n = self.p.chains[si].len();
            let base = k * self.nsl + self.slot[si];
            self.bb[base..base + n].fill(OPEN);
            self.bb[base + n - 1] = nl;
            let mut lo = 0;
            for j in 0..n - 1 {
                let (bud, mult) = (self.var.budget[si][j], self.var.mult[si][j]);
                let fixed = if lo == nl || self.full(si, j) * mult <= bud {
                    lo = nl;
                    nl
                } else if lo > k {
                    OPEN
                } else if self.tile(lo, si, j) * mult > bud {
                    lo
                } else {
                    match (lo + 1..=k).find(|&m| self.tile(m, si, j) * mult > bud) {
                        Some(m) => {
                            lo = m - 1;
                            m - 1
                        }
                        None => OPEN,
                    }
                };
                if fixed == OPEN {
                    all = false;
                    break;
                }
                self.bb[base + j] = fixed;
            }
        }
        all
    }

    /// Lower bound of the objective over every completion of the current prefix.
    fn lower_bound(&self) -> f64 {
        let p = self.p;
        let unit = p.unit;
        let k = self.prefix.len();
        let want_e = self.cfg.obj != Objective::Latency;
        let want_c = self.cfg.obj != Objective::Energy;
        let mut busy = [(u32::MAX, 0u64); 32];
        let mut nb = 0;
        let mut energy = self.e_floor;
        let mut add = |pr: &PortRef, level: LevelIx, amount: u64, bits: u32, count: u64, inst: u64, read: bool| {
            if want_e {
                let lv = &unit.levels[level];
                energy += (amount * inst * count * u64::from(bits)).div_ceil(8) as f64
                    * if read { lv.e_read_j_per_b } else { lv.e_write_j_per_b };
            }
            if want_c && pr.key != u32::MAX {
                let r = pr.cycles(amount, bits) * count;
                match busy[..nb].iter().position(|x| x.0 == pr.key) {
                    Some(i) => busy[i].1 += r,
                    None if nb < 32 => {
                        busy[nb] = (pr.key, r);
                        nb += 1;
                    }
                    None => {}
                }
            }
        };
        let (mut onload, mut offload) = (0u64, 0u64);
        for (si, s) in p.streams.iter().enumerate() {
            let chain = &p.chains[si];
            let base = self.slot[si];
            let el = |kk: usize, j: usize| -> u64 {
                if kk <= k { self.el[kk * self.nsl + base + j] } else { self.full_el[base + j] }
            };
            let ext = |x: usize| -> u64 {
                let mut f = 1;
                let mut y = x;
                while y < k && !s.relevant(self.prefix[y].dim) {
                    f *= self.prefix[y].factor;
                    y += 1;
                }
                if y == k { f * self.ir_left[si] } else { f }
            };
            let pr = &self.pref[base..base + chain.len()];
            let c0 = if s.is_output { self.ttot } else { self.ttot / ext(0).min(s.run).max(1) };
            let d0 = if s.is_output { 3 } else { 0 };
            add(&pr[0][d0], chain[0], el(0, 0), s.bits, c0, self.sc.inst[si][0], !s.is_output);
            let (up, dn) = if s.is_output { (2, 3) } else { (1, 0) };
            for j in 0..chain.len() - 1 {
                let bj = self.b(k, si, j);
                if bj == OPEN {
                    break;
                }
                let count = if bj >= self.nl { 1 } else { self.ttot / (self.cyc_at(bj) * ext(bj)).max(1) };
                add(&pr[j][up], chain[j], el(bj, j), s.bits, count, self.sc.inst[si][j], s.is_output);
                add(&pr[j + 1][dn], chain[j + 1], el(bj, j + 1), s.bits, count, self.sc.inst[si][j + 1], !s.is_output);
            }
            // Startup and final drain move the level-0 tile through every boundary (03 §2.5); known once b_0 is.
            let b0 = self.b(k, si, 0);
            if want_c && b0 != OPEN && chain.len() > 1 {
                let chunk = el(b0, 1);
                let t: u64 = (0..chain.len() - 1)
                    .map(|j| pr[j][up].cycles(chunk, s.bits).max(pr[j + 1][dn].cycles(chunk, s.bits)) + unit.levels[chain[j + 1]].latency_cycles)
                    .sum();
                if s.is_output {
                    offload = offload.max(t);
                } else {
                    onload = onload.max(t);
                }
            }
        }
        let pl = unit.pipeline;
        let cycles = busy[..nb].iter().map(|x| x.1 as f64).fold(self.issue, f64::max) + (onload + offload + pl.fill + pl.drain) as f64;
        objective_value(self.cfg.obj, cycles, energy, self.cfg.floors.0, self.cfg.floors.1)
    }

    fn cyc_at(&self, k: usize) -> u64 {
        if k < self.cyc.len() { self.cyc[k] } else { self.ttot }
    }

    fn push(&mut self, t: usize) {
        let l = self.types[t];
        self.left[t] -= 1;
        self.prefix.push(l);
        let mut tp = *self.tp.last().expect("root");
        tp[l.dim] *= l.factor;
        self.cyc.push(self.cyc.last().expect("root") * l.factor);
        self.tp.push(tp);
        let row = self.prefix.len();
        self.step_tiles(row, l);
        for (si, s) in self.p.streams.iter().enumerate() {
            if !s.relevant(l.dim) {
                self.ir_left[si] /= l.factor;
            }
        }
    }

    fn pop(&mut self, t: usize) {
        let l = self.prefix.pop().expect("non-empty");
        self.left[t] += 1;
        self.tp.pop();
        self.cyc.pop();
        for (si, s) in self.p.streams.iter().enumerate() {
            if !s.relevant(l.dim) {
                self.ir_left[si] *= l.factor;
            }
        }
    }

    /// Explores the subtree of the current prefix; `bounded`: the caller already bounded it and fixed its
    /// boundaries (whether all are fixed).
    fn run(&mut self, bounded: Option<bool>) {
        if self.stats.truncated || self.at_floor() {
            return;
        }
        let all = match bounded {
            Some(all) => all,
            None => {
                let all = self.bounds();
                if let Some(bv) = self.prune_val()
                    && self.lower_bound() > bv
                {
                    self.stats.pruned += 1;
                    return;
                }
                all
            }
        };
        let mut sig = std::mem::take(&mut self.keybuf);
        self.signature(&mut sig);
        if self.done.contains(&sig[..]) {
            self.keybuf = sig;
            self.stats.pruned += 1;
            return;
        }
        let owned = sig.clone();
        self.keybuf = sig;
        let sig = owned;
        if all || self.prefix.len() == self.nl {
            self.leaves();
        } else {
            self.expand();
        }
        if !self.stats.truncated && !self.at_floor() {
            self.done.insert(sig);
        }
    }

    fn expand(&mut self) {
        let mut kids = [(0.0f64, 0usize); LPF_KIDS];
        let mut nk = 0;
        for t in 0..self.types.len() {
            if self.left[t] > 0 {
                self.push(t);
                self.bounds();
                kids[nk] = (self.lower_bound(), t);
                nk += 1;
                self.pop(t);
            }
        }
        kids[..nk].sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(lb, t) in &kids[..nk] {
            if let Some(bv) = self.prune_val()
                && lb > bv
            {
                self.stats.pruned += 1;
                continue;
            }
            self.push(t);
            let all = self.bounds();
            self.run(Some(all));
            self.pop(t);
            if self.stats.truncated || self.at_floor() {
                return;
            }
        }
    }

    /// Everything the subtree below the current prefix depends on (fixed boundaries with the extents and
    /// stationary runs at them, canonical boundaries already decided inside the prefix, its leading run, the
    /// loops left): two prefixes with equal signatures have subtrees with equal lower bounds and equal leaf
    /// equivalence keys, so once one is fully explored every leaf of the other is a `seen` duplicate.
    fn signature(&self, sig: &mut Vec<u64>) {
        let k = self.prefix.len();
        let nd = self.p.nest.dims.len();
        sig.clear();
        sig.push(k as u64);
        sig.extend(self.left.iter().map(|&c| c as u64));
        for (si, s) in self.p.streams.iter().enumerate() {
            let run = |x: usize| {
                let (mut f, mut y) = (1u64, x);
                while y < k && !s.relevant(self.prefix[y].dim) {
                    f *= self.prefix[y].factor;
                    y += 1;
                }
                [f, u64::from(y == k)]
            };
            sig.extend_from_slice(&run(0));
            if s.run != u64::MAX {
                // A capped run depends on the order of its factors, not only their product.
                sig.push(crate::model::stationary_run(s, &self.prefix) as u64);
            }
            let n = self.p.chains[si].len();
            let mut canon = Some(0usize);
            for j in 0..n {
                let bj = self.b(k, si, j);
                sig.push(bj as u64);
                if bj == OPEN {
                    canon = None;
                    continue;
                }
                if bj < self.nl {
                    sig.extend_from_slice(&self.tp[bj][..nd]);
                    sig.extend_from_slice(&run(bj));
                }
                if j + 1 < n
                    && let Some(c) = canon
                {
                    let mut x = if j == 0 { bj } else { bj.max(c) };
                    while x < k && !s.relevant(self.prefix[x].dim) {
                        x += 1;
                    }
                    if x >= self.nl {
                        sig.push(self.nl as u64);
                        canon = Some(self.nl);
                    } else if x < k {
                        sig.push(x as u64);
                        sig.extend_from_slice(&self.tp[x][..nd]);
                        canon = Some(x);
                    } else {
                        sig.push(u64::MAX);
                        canon = None;
                    }
                }
            }
        }
    }

    /// The DFS over every capacity-share and double-buffer variant that can matter.
    fn explore(&mut self) {
        let p = self.p;
        let shared = (0..p.unit.levels.len()).any(|l| p.chains.iter().filter(|c| c[..c.len() - 1].contains(&l)).count() > 1);
        if self.os.is_some() {
            self.os_explore();
            return;
        }
        let mut variants = vec![(Share::Even, true)];
        let mut i = 0;
        while i < variants.len() {
            let (share, dbl) = variants[i];
            i += 1;
            self.buffered = dbl;
            match self.variant(share, dbl) {
                Ok(v) => {
                    self.var = v;
                    if i == 1 && self.binds() {
                        variants.push((Share::Even, false));
                        if shared {
                            variants.extend_from_slice(&[
                                (Share::Proportional, true),
                                (Share::Proportional, false),
                                (Share::OutputFirst, true),
                                (Share::OutputFirst, false),
                            ]);
                        }
                    }
                    self.done.clear();
                    self.run(None);
                }
                Err(e) => {
                    if i == 1 {
                        variants.push((Share::Even, false));
                    }
                    self.first_err.get_or_insert(e);
                }
            }
        }
    }

    /// The output-stationary pass: its loop items are cut so the output tile fits half the outermost on-chip
    /// level, only the output-half capacity split is tried, and only nests holding that tile are costed (the
    /// template gives a unit the whole off-chip boundary, so latency alone would accept nests re-reading inputs
    /// from it many times over while the units sharing it contend).
    fn os_explore(&mut self) {
        for dbl in [true, false] {
            self.buffered = dbl;
            match self.variant(Share::OutputHalf, dbl) {
                Ok(v) => {
                    self.var = v;
                    self.done.clear();
                    self.run(None);
                }
                Err(e) => {
                    self.first_err.get_or_insert(e);
                }
            }
            if self.stats.truncated || self.at_floor() {
                return;
            }
        }
    }

    /// Subtrees above this objective are pruned. Earlier spatial candidates' best (`SearchCfg::bound`) is
    /// compared after their double-buffer upgrade, so it prunes only variants that already buffer: a leaf of an
    /// unbuffered variant may still beat it once upgraded.
    fn prune_val(&self) -> Option<f64> {
        let ext = self.cfg.bound.filter(|_| self.buffered);
        match (self.best_val, ext) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// The incumbent's issue + stall cycles reach the floor: only startup and fill/drain remain above it.
    fn at_floor(&self) -> bool {
        matches!((self.cfg.stop, &self.best), (Some(f), Some((ev, _))) if (ev.issue + ev.stall) as f64 <= f)
    }

    /// Costs the completions of a prefix whose boundaries are all fixed: one per arrangement of the remaining
    /// loops grouped by dim (within a group, order affects nothing).
    fn leaves(&mut self) {
        let p = self.p;
        let k = self.prefix.len();
        let alloc: Vec<Vec<usize>> = (0..p.streams.len()).map(|si| (0..p.chains[si].len()).map(|j| self.b(k, si, j).min(self.nl)).collect()).collect();
        let double_buffer: Vec<Vec<bool>> = alloc
            .iter()
            .enumerate()
            .map(|(si, a)| (0..a.len()).map(|j| self.var.mult[si][j] == 2 && !(j > 0 && a[j] == a[j - 1])).collect())
            .collect();
        let mut groups: Vec<Vec<TemporalLoop>> = vec![];
        for (t, &n) in self.types.iter().zip(&self.left) {
            for _ in 0..n {
                match groups.iter_mut().find(|g| g[0].dim == t.dim) {
                    Some(g) => g.push(*t),
                    None => groups.push(vec![*t]),
                }
            }
        }
        let mut order: Vec<usize> = (0..groups.len()).collect();
        let many = groups.len() > 4;
        let mut loops = self.prefix.clone();
        let mut key = std::mem::take(&mut self.keybuf);
        loop {
            loops.truncate(k);
            for &g in &order {
                loops.extend_from_slice(&groups[g]);
            }
            equivalence_key(p, &loops, &alloc, &double_buffer, &mut key);
            if self.seen.contains(&key[..]) {
                self.stats.pruned += 1;
            } else {
                self.seen.insert(key.clone());
                let tm = TemporalMapping { loops: loops.clone(), alloc: alloc.clone(), double_buffer: double_buffer.clone() };
                self.leaf(tm);
            }
            if many || self.stats.truncated || self.at_floor() || !next_index_perm(&mut order) {
                break;
            }
        }
        self.keybuf = key;
    }

    /// Costs a loop order whose equivalence key was not seen before.
    fn leaf(&mut self, tm: TemporalMapping) {
        let p = self.p;
        if let Some(tile) = &self.os
            && !holds_tile(p, &tm, tile)
        {
            self.stats.pruned += 1;
            return;
        }
        if self.evals >= self.cfg.budget.max_evals_per_spatial {
            self.stats.truncated = true;
            return;
        }
        self.evals += 1;
        self.stats.evaluated += 1;
        let oc = match OrderCtx::new(p, self.sc, &tm.loops) {
            Ok(oc) => oc,
            Err(e) => {
                self.first_err.get_or_insert(e);
                return;
            }
        };
        match evaluate_with(p, self.sizes, self.sc, &oc, &tm, true, false) {
            Ok(ev) => {
                let (fc, fe) = self.cfg.floors;
                if self.best.as_ref().is_none_or(|b| better(self.cfg.obj, &ev, &b.0, fc, fe).then(tm.cmp(&b.1)).is_lt()) {
                    let v = objective_value(self.cfg.obj, ev.total_f, ev.energy.total_j, fc, fe);
                    self.best_val = Some(self.best_val.map_or(v, |b| b.min(v)));
                    self.best = Some((ev, tm));
                }
            }
            Err(e) => {
                self.first_err.get_or_insert(e);
            }
        }
    }
}

fn next_index_perm(v: &mut [usize]) -> bool {
    let n = v.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && v[i - 1] >= v[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while v[j] <= v[i - 1] {
        j -= 1;
    }
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}

/// Native costs depend on a loop order only through each stream's per-level temporal extents (after the
/// stationarity merge), its stationary prefix at level 0, and buffering; orders agreeing on these are equivalent.
/// Written into `key` (cleared first).
fn equivalence_key(p: &Prep, loops: &[TemporalLoop], alloc: &[Vec<usize>], double_buffer: &[Vec<bool>], key: &mut Vec<u64>) {
    let nd = p.nest.dims.len();
    let nl = loops.len();
    key.clear();
    for (si, s) in p.streams.iter().enumerate() {
        // `canonical_bounds` without allocating: boundaries after the stationarity merge, then the stationary prefix.
        let a = &alloc[si];
        let n = a.len();
        let mut prev = 0usize;
        let mut b0 = 0usize;
        for j in 0..n {
            let bj = if j + 1 < n {
                let mut x = if j > 0 { a[j].max(prev) } else { a[j] };
                while x < nl && !s.relevant(loops[x].dim) {
                    x += 1;
                }
                x
            } else {
                a[j]
            };
            if j == 0 {
                b0 = bj;
            }
            prev = bj;
            let mut ext = [1u64; MAX_DIMS];
            for l in &loops[..bj] {
                ext[l.dim] *= l.factor;
            }
            key.extend_from_slice(&ext[..nd]);
            key.push(u64::from(double_buffer[si][j]));
        }
        let s0 = if s.is_output { 0 } else { crate::model::stationary_run(s, &loops[..b0]) };
        key.push(loops[..s0].iter().map(|l| l.factor).product());
    }
}

/// Per dim, the output's tile extent (temporal iterations) at its outermost on-chip level.
type OsTile = Vec<(usize, u64)>;

/// Outcome of one DFS pass over a loop-item set.
struct Pass {
    best: Option<(ClassEval, TemporalMapping)>,
    first_err: Option<Diagnostic>,
    evals: u64,
    stats: SearchStats,
}

#[allow(clippy::too_many_arguments)]
fn dfs_pass(p: &Prep, sizes: &[u64], sc: &SpatialCtx, cfg: &SearchCfg, items: Vec<TemporalLoop>, os: Option<OsTile>, issue: f64, e_floor: f64) -> Pass {
    let nd = p.nest.dims.len();
    let ttot: u64 = sc.t[..nd].iter().product();
    let mut types: Vec<TemporalLoop> = items.clone();
    types.dedup();
    let left = types.iter().map(|t| items.iter().filter(|x| *x == t).count()).collect();
    let mut dfs = Dfs {
        p,
        sc,
        sizes,
        cfg,
        nl: items.len(),
        ttot,
        issue,
        e_floor,
        types,
        left,
        prefix: vec![],
        tp: vec![[1; MAX_DIMS]],
        cyc: vec![1],
        slot: vec![],
        nsl: 0,
        tiles: vec![],
        el: vec![],
        full: vec![],
        full_el: vec![],
        pref: vec![],
        bb: vec![],
        ir_left: p.streams.iter().map(|s| items.iter().filter(|l| !s.relevant(l.dim)).map(|l| l.factor).product()).collect(),
        var: Variant { budget: vec![], mult: vec![] },
        best: None,
        best_val: None,
        buffered: true,
        seen: BTreeSet::new(),
        done: BTreeSet::new(),
        keybuf: vec![],
        evals: 0,
        stats: SearchStats::default(),
        first_err: None,
        os,
    };
    let mut off = 0;
    for c in &p.chains {
        dfs.slot.push(off);
        off += c.len();
    }
    dfs.nsl = off;
    dfs.tiles = vec![0; (dfs.nl + 1) * off];
    dfs.el = vec![0; (dfs.nl + 1) * off];
    for (si, ch) in p.chains.iter().enumerate() {
        for (j, &l) in ch.iter().enumerate() {
            let mut row = [PortRef::NONE; 4];
            for (d, r) in row.iter_mut().enumerate() {
                if let Some(pt) = p.ports[si][j][d] {
                    *r = PortRef::new(l, pt, p.unit.levels[l].ports[pt].bytes_per_cycle);
                }
            }
            dfs.pref.push(row);
        }
    }
    dfs.bb = vec![OPEN; (dfs.nl + 1) * off];
    dfs.fill_tiles(0, &[1; MAX_DIMS]);
    let mut full_tp = [1u64; MAX_DIMS];
    full_tp[..nd].copy_from_slice(&sc.t[..nd]);
    dfs.fill_tiles(dfs.nl, &full_tp);
    dfs.full = dfs.tiles[dfs.nl * off..(dfs.nl + 1) * off].to_vec();
    dfs.full_el = dfs.el[dfs.nl * off..(dfs.nl + 1) * off].to_vec();
    dfs.explore();
    Pass { best: dfs.best, first_err: dfs.first_err, evals: dfs.evals, stats: dfs.stats }
}

/// Loop items for the output-stationary pass, or None when the nest has no partial sums to keep on chip.
/// Partial sums leave the chip when a reduction loop sits above the output's outermost on-chip boundary; the
/// standard items (prime factors coarsened to `LPF_LIMIT`) often cannot express an output tile that both fits
/// that level and reuses the inputs well (2048 = 16 x 128 reaches 16, 128 or 2048 rows, never 512). Here each
/// parallel output dim is cut at the tile `T` that minimizes input re-reads from above the level with the
/// output tile in half of it (the [`Share::OutputHalf`] budget), so the nest `[inner tile loops, reductions,
/// outer loops]` is reachable. Returns the items and the tile.
fn os_items(p: &Prep, sc: &SpatialCtx) -> Option<(Vec<TemporalLoop>, OsTile)> {
    const MAX_ITEMS: usize = crate::search::LPF_LIMIT + 3;
    let nd = p.nest.dims.len();
    let t = &sc.t[..nd];
    let oi = p.streams.iter().position(|s| s.is_output)?;
    let out = &p.streams[oi];
    let chain = &p.chains[oi];
    if chain.len() < 2 || !(0..nd).any(|d| !out.relevant(d) && t[d] > 1) {
        return None;
    }
    let j = chain.len() - 2;
    let lv = &p.unit.levels[chain[j]];
    let n_inst: u64 = lv.instance_axes.iter().map(|&a| u64::from(p.unit.axes[a].size)).product::<u64>().max(1);
    let budget = lv.share(out.role).1 / n_inst / 2;
    let par: Vec<usize> = (0..nd).filter(|&d| out.relevant(d) && t[d] > 1).collect();
    if par.is_empty() {
        return None;
    }
    let divisors = |x: u64| (1..=x).filter(|v| x.is_multiple_of(*v)).collect::<Vec<u64>>();
    let divs: Vec<Vec<u64>> = par.iter().map(|&d| if t[d] > 1 << 16 { vec![1, t[d]] } else { divisors(t[d]) }).collect();
    if divs.iter().map(|v| v.len() as f64).product::<f64>() > 1e5 {
        return None;
    }
    let bits = u64::from(out.acc_bits.max(out.bits));
    let mut full = [1u64; MAX_DIMS];
    full[..nd].copy_from_slice(&sc.full_ext[..nd]);
    let ins: Vec<(f64, u64)> = p
        .streams
        .iter()
        .filter(|s| !s.is_output)
        .map(|s| ((s.footprint(&full) * u64::from(s.bits)) as f64 / 8.0, par.iter().enumerate().filter(|&(_, &d)| !s.relevant(d)).fold(0u64, |m, (i, _)| m | 1 << i)))
        .collect();
    let sp = &sc.sp[oi][j];
    let tile = |c: &[u64]| {
        let mut e = [1u64; MAX_DIMS];
        e[..nd].copy_from_slice(&sp[..nd]);
        for (x, &d) in par.iter().enumerate() {
            e[d] *= c[x];
        }
        (out.footprint(&e) * bits).div_ceil(8)
    };
    // Every combination of per-dim divisors (an odometer over `divs`): fewest input re-reads, then the smaller tile.
    let mut best: Option<(f64, u64, Vec<u64>)> = None;
    let mut ix = vec![0usize; par.len()];
    loop {
        let cur: Vec<u64> = ix.iter().zip(&divs).map(|(&i, v)| v[i]).collect();
        let bytes = tile(&cur);
        if bytes <= budget {
            let traffic: f64 = ins
                .iter()
                .map(|&(b, mask)| b * (0..par.len()).filter(|&x| mask & 1 << x != 0).map(|x| (t[par[x]] / cur[x]) as f64).product::<f64>())
                .sum();
            if best.as_ref().is_none_or(|b| traffic.total_cmp(&b.0).then(bytes.cmp(&b.1)).then(cur.cmp(&b.2)).is_lt()) {
                best = Some((traffic, bytes, cur));
            }
        }
        let Some(x) = (0..ix.len()).find(|&x| ix[x] + 1 < divs[x].len()) else { break };
        ix[x] += 1;
        ix[..x].fill(0);
    }
    let (_, _, tsz) = best?;
    // Pieces per dim: (dim, extent, items allowed); inner tile pieces may take two items while the total allows.
    let mut pieces: Vec<(usize, u64, usize)> = vec![];
    for (d, &td) in t.iter().enumerate() {
        match par.iter().position(|&x| x == d) {
            Some(x) => {
                pieces.push((d, tsz[x], 1));
                pieces.push((d, td / tsz[x], 1));
            }
            None => pieces.push((d, td, 1)),
        }
    }
    pieces.retain(|x| x.1 > 1);
    let mut total = pieces.len();
    if total > MAX_ITEMS {
        return None;
    }
    let mut order: Vec<usize> = (0..pieces.len()).filter(|&i| crate::search::loop_items(&[pieces[i].1]).len() > 1).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(pieces[i].1), i));
    for i in order {
        if total < MAX_ITEMS {
            pieces[i].2 = 2;
            total += 1;
        }
    }
    let mut items = vec![];
    for (d, ext, n) in pieces {
        for f in crate::search::coarsen(ext, n) {
            items.push(TemporalLoop { dim: d, factor: f });
        }
    }
    items.sort();
    Some((items, par.iter().copied().zip(tsz).collect()))
}

/// Best temporal mapping of one tile class under one spatial mapping; `evals` receives the loop orders the DFS
/// evaluated against the budget. When the best nest of the standard items reads partial sums back from off
/// chip, the output-stationary items ([`os_items`]) are searched too and the better result wins.
pub(crate) fn search_class(
    p: &Prep,
    sizes: &[u64],
    sp: &SpatialMapping,
    cfg: &SearchCfg,
    stats: &mut SearchStats,
    evals: &mut u64,
) -> Result<Option<ClassResult>, Diagnostic> {
    let sc = SpatialCtx::new(p, sizes, sp)?;
    let nd = p.nest.dims.len();
    let ttot: u64 = sc.t[..nd].iter().product();
    let issue = (ttot as f64 * p.ops_per_point as f64 / p.lane_rate).ceil();
    let m = &p.unit.modes[p.mode];
    let useful = crate::model::class_points(p.nest, sizes) * p.ops_per_point;
    let issued = (issue * m.macs_per_cycle).max(useful as f64);
    let e_floor = useful as f64 * m.e_mac_j + (issued - useful as f64) * m.e_mac_j * p.unit.e_mac_idle_ratio;
    if cfg.obj == Objective::Latency
        && let Some(b) = cfg.bound
        && issue > b
    {
        stats.pruned += 1;
        return Ok(None);
    }
    let (fc, fe) = cfg.floors;
    let mut best: Option<ClassResult> = None;
    let mut first_err = None;
    *evals = 0;
    let mut finish = |x: Pass, best: &mut Option<ClassResult>, stats: &mut SearchStats| -> Result<(), Diagnostic> {
        *evals += x.evals;
        stats.evaluated += x.stats.evaluated;
        stats.pruned += x.stats.pruned;
        stats.truncated |= x.stats.truncated;
        if first_err.is_none() {
            first_err = x.first_err;
        }
        let Some((ev, tm)) = x.best else { return Ok(()) };
        // Each pass's best is upgraded before the comparison: buffering can reorder them.
        let tm = upgrade_double_buffer(p, sizes, &sc, cfg, ev, tm, stats);
        let oc = OrderCtx::new(p, &sc, &tm.loops)?;
        let eval = evaluate_with(p, sizes, &sc, &oc, &tm, true, true)?;
        if best.as_ref().is_none_or(|b| better(cfg.obj, &eval, &b.eval, fc, fe).then(tm.cmp(&b.tm)).is_lt()) {
            *best = Some(ClassResult { eval, tm });
        }
        Ok(())
    };
    let standard = dfs_pass(p, sizes, &sc, cfg, crate::search::loop_items(&sc.t[..nd]), None, issue, e_floor);
    finish(standard, &mut best, stats)?;
    if !p.compat
        && best.as_ref().is_some_and(|b| spills_offchip(p, &b.tm))
        && let Some((items, tile)) = os_items(p, &sc)
    {
        let os = dfs_pass(p, sizes, &sc, cfg, items, Some(tile), issue, e_floor);
        finish(os, &mut best, stats)?;
    }
    match (best, first_err) {
        (Some(r), _) => Ok(Some(r)),
        (None, Some(e)) if cfg.bound.is_none() => Err(e),
        _ => Ok(None),
    }
}

/// The output's tile at its outermost on-chip level (after the stationarity merge) is `tile` and every
/// reduction loop runs inside it: the output-stationary nest the pass is for.
fn holds_tile(p: &Prep, tm: &TemporalMapping, tile: &[(usize, u64)]) -> bool {
    let Some(si) = p.streams.iter().position(|s| s.is_output) else { return false };
    let s = &p.streams[si];
    let mut x = tm.alloc[si][p.chains[si].len() - 2];
    while x < tm.loops.len() && !s.relevant(tm.loops[x].dim) {
        x += 1;
    }
    let mut ext = [1u64; MAX_DIMS];
    for l in &tm.loops[..x] {
        ext[l.dim] *= l.factor;
    }
    tile.iter().all(|&(d, t)| ext[d] == t) && tm.loops[x..].iter().all(|l| s.relevant(l.dim))
}

/// A reduction loop sits above the output's outermost on-chip boundary (after the stationarity merge): its
/// partial sums go off chip and are read back.
fn spills_offchip(p: &Prep, tm: &TemporalMapping) -> bool {
    p.streams.iter().enumerate().filter(|(si, s)| s.is_output && p.chains[*si].len() > 1).any(|(si, s)| {
        let n = p.chains[si].len();
        let mut x = tm.alloc[si][n - 2];
        while x < tm.loops.len() && !s.relevant(tm.loops[x].dim) {
            x += 1;
        }
        tm.loops[x..].iter().any(|l| !s.relevant(l.dim))
    })
}

/// The DFS splits each level's capacity evenly (or by fixed rules) between streams before choosing tiles, so a
/// mapping whose tiles leave room to double buffer can be found only without buffering. Turn buffering on for
/// the best mapping wherever the actual tiles fit: all eligible flags at once, else one at a time.
fn upgrade_double_buffer(
    p: &Prep,
    sizes: &[u64],
    sc: &SpatialCtx,
    cfg: &SearchCfg,
    ev: ClassEval,
    tm: TemporalMapping,
    stats: &mut SearchStats,
) -> TemporalMapping {
    let Ok(oc) = OrderCtx::new(p, sc, &tm.loops) else { return tm };
    let eligible: Vec<(usize, usize)> = tm
        .double_buffer
        .iter()
        .enumerate()
        .flat_map(|(si, v)| {
            let a = &tm.alloc[si];
            (0..v.len())
                .filter(move |&j| !v[j] && j + 1 < v.len() && !(j > 0 && a[j] == a[j - 1]))
                .filter(move |&j| p.unit.levels[p.chains[si][j]].double_buffer)
                .map(move |j| (si, j))
        })
        .collect();
    if eligible.is_empty() {
        return tm;
    }
    let (fc, fe) = cfg.floors;
    let mut eval = |t: &TemporalMapping| {
        stats.evaluated += 1;
        evaluate_with(p, sizes, sc, &oc, t, true, false).ok()
    };
    let mut all = tm.clone();
    eligible.iter().for_each(|&(si, j)| all.double_buffer[si][j] = true);
    if let Some(e) = eval(&all)
        && better(cfg.obj, &e, &ev, fc, fe).is_lt()
    {
        return all;
    }
    // Greedy: keep each flag that still fits and is no worse, so flags that only help together accumulate.
    let (mut cur, mut cur_ev) = (tm.clone(), ev.clone());
    for &(si, j) in &eligible {
        let mut t = cur.clone();
        t.double_buffer[si][j] = true;
        if let Some(e) = eval(&t)
            && better(cfg.obj, &e, &cur_ev, fc, fe).is_le()
        {
            (cur, cur_ev) = (t, e);
        }
    }
    if better(cfg.obj, &cur_ev, &ev, fc, fe).is_lt() { cur } else { tm }
}
