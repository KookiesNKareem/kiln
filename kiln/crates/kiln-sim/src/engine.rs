//! Tier A core (03 §4.2-4.4): per-segment busy time on every resource, A0 / A1 / A2 / A_est, critical path,
//! M/D/1 contention on critical-path resources, and execution-model overheads.

use kiln_map::hwview::{HwView, ResClass};
use kiln_map::lower::{Amount, Task, TaskGraph, TaskKind};
use kiln_map::mapping::LaunchKind;
use kiln_phys::ClockPlan;
use kiln_trace::sim::{Binding, BindingClass, OverheadKind, RunnerUp};

use crate::params::SimParams;

#[derive(Clone, Debug, PartialEq)]
pub struct SegOut {
    pub groups: (usize, usize),
    pub iteration: Option<u32>,
    pub start_s: f64,
    pub a0: f64,
    pub a1_floor: f64,
    pub l_floor: f64,
    pub a1_est: f64,
    pub l_est: f64,
    pub contention: f64,
    pub overhead: f64,
    pub overhead_kind: OverheadKind,
    /// Software-stack kernels beyond the modelled groups (08 §F): their traffic time off chip and on chip, and
    /// their launch terms (minimum-kernel padding plus gap or dispatch).
    pub stack_dram: f64,
    pub stack_onchip: f64,
    pub stack_overhead: f64,
    /// `max(A1, L)` at peak DRAM efficiency plus overheads (03 A2).
    pub time_floor: f64,
    /// A2 core at calibrated efficiency plus contention and overheads (03 A_est).
    pub time_est: f64,
    pub binding: Binding,
    pub runner_up: Option<RunnerUp>,
    pub binding_resource: Option<u32>,
}

impl SegOut {
    pub fn core_est(&self) -> f64 {
        self.a1_est.max(self.l_est)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunOut {
    pub segs: Vec<SegOut>,
    /// Busy seconds (estimate efficiencies) and bytes per resource, by iteration class: prologue/epilogue,
    /// the middle window iteration (steady state, 03 §4.9), and the other window iterations.
    pub busy: [Vec<f64>; 3],
    pub bytes: [Vec<f64>; 3],
    /// Of `bytes`, those written into memories (priced at their write energy).
    pub wbytes: [Vec<f64>; 3],
    pub task_start: Vec<f64>,
    pub task_end: Vec<f64>,
    pub mid_iteration: Option<u32>,
}

pub const PE: usize = 0;
pub const MID: usize = 1;
pub const OTHER: usize = 2;

/// Per-run buffers reused by every segment: busy time per resource and bytes per profile, zero outside the
/// entries a segment touched (kept in `touched` / `ptouched`).
struct Scratch {
    busy_f: Vec<f64>,
    busy_e: Vec<f64>,
    mark: Vec<bool>,
    touched: Vec<u32>,
    prof_bytes: Vec<f64>,
    ptouched: Vec<u32>,
    /// Off-chip and shared on-chip memory resources with their share of the level's bandwidth (stack kernels).
    dram_sh: Vec<(usize, f64)>,
    onchip_sh: Vec<(usize, f64)>,
    /// One task's demand time per resource (zero outside `ttouched`).
    task_f: Vec<f64>,
    task_e: Vec<f64>,
    ttouched: Vec<u32>,
    /// Every task's latency at the plan's clocks.
    lat: Vec<f64>,
}

pub struct Engine<'a> {
    pub view: &'a HwView,
    pub g: &'a TaskGraph,
    pub params: &'a SimParams,
    pub clocks: &'a ClockPlan,
}

impl Engine<'_> {
    fn compute_hz(&self, r: usize) -> f64 {
        let res = &self.view.resources[r];
        self.clocks.hz(res.clock).unwrap_or(1.0e9)
    }

    /// Bytes per second of a link or memory resource at the plan's clocks: on-chip memory ports and on-die links
    /// move a fixed width per cycle of their domain; DRAM and PHY rates are clock independent.
    fn capacity(&self, r: usize) -> f64 {
        let res = &self.view.resources[r];
        let synchronous = match res.class {
            ResClass::Mem => true,
            ResClass::Link => !res.link_class.as_deref().is_some_and(|c| crate::result::OFF_DIE.contains(&c)),
            _ => false,
        };
        match res.clock.filter(|_| synchronous) {
            Some(c) => res.capacity * self.clocks.hz(Some(c)).map_or(1.0, |f| f / self.view.phys.nominal_hz(c)),
            None => res.capacity,
        }
    }

    /// Seconds per unit of demand on every resource (per cycle for compute, per byte otherwise), at peak (`.0`)
    /// and at calibrated efficiencies (`.1`: DRAM efficiency, per-template compute efficiency); `cap_scale`
    /// entries apply in their order.
    fn invs(&self) -> (Vec<f64>, Vec<f64>) {
        let nres = self.view.resources.len();
        let mut scale: Vec<(u32, f64)> = self.params.cap_scale.clone();
        scale.sort_by_key(|x| x.0);
        let mut eff: Vec<f64> = vec![1.0; nres];
        if !self.params.unit_eff.is_empty() {
            for u in self.view.units.iter().rev() {
                if let Some(x) = self.params.unit_eff.iter().find(|(t, _)| *t == u.template) {
                    eff[u.compute as usize] = x.1;
                } else {
                    eff[u.compute as usize] = 1.0;
                }
            }
        }
        let mut i = 0;
        let (mut f, mut e) = (Vec::with_capacity(nres), Vec::with_capacity(nres));
        for (r, (res, &eff)) in self.view.resources.iter().zip(&eff).enumerate() {
            let j = i;
            while i < scale.len() && scale[i].0 as usize == r {
                i += 1;
            }
            let cap = |eta: bool| {
                let mut c = self.capacity(r).max(1e-30);
                if eta && res.class == ResClass::Dram {
                    c *= self.params.eta_dram;
                }
                for &(_, k) in &scale[j..i] {
                    c *= k;
                }
                c
            };
            match res.class {
                ResClass::Compute => {
                    f.push(1.0 / self.compute_hz(r));
                    e.push(1.0 / (self.compute_hz(r) * eff));
                }
                ResClass::Sequencer => {
                    f.push(1.0);
                    e.push(1.0);
                }
                _ => {
                    f.push(1.0 / cap(false));
                    e.push(1.0 / cap(true));
                }
            }
        }
        (f, e)
    }

    /// Task latencies at the plan's clocks: clocked components take `f_nom / f` of their nominal time (fill
    /// cycles, on-die hops and on-chip memory access stretch when a domain throttles); DRAM and PHY delays do not.
    fn latencies(&self) -> Vec<f64> {
        let stretch: Vec<f64> = (0..self.clocks.hz.len()).map(|c| self.clocks.hz(Some(c)).map_or(1.0, |f| self.view.phys.nominal_hz(c) / f) - 1.0).collect();
        self.g
            .tasks
            .iter()
            .map(|t| t.lat_s + self.g.lat_clk_of(t).iter().map(|&(c, x)| x * stretch.get(c).copied().unwrap_or(0.0)).sum::<f64>())
            .collect()
    }

    pub fn run(&self, mid_iteration: Option<u32>) -> RunOut {
        let nres = self.view.resources.len();
        let (inv_f, inv_e) = self.invs();
        let prof_max = |inv: &[f64]| -> Vec<f64> {
            self.g.profiles.iter().map(|p| p.entries.iter().map(|&(r, f)| f * inv[r as usize]).fold(0.0, f64::max)).collect()
        };
        let (pm_f, pm_e) = (prof_max(&inv_f), prof_max(&inv_e));
        let pm_arg: Vec<u32> = self
            .g
            .profiles
            .iter()
            .map(|p| p.entries.iter().fold((u32::MAX, 0.0f64), |a, &(r, f)| if f * inv_e[r as usize] > a.1 { (r, f * inv_e[r as usize]) } else { a }).0)
            .collect();
        let mut out = RunOut {
            busy: [vec![0.0; nres], vec![0.0; nres], vec![0.0; nres]],
            bytes: [vec![0.0; nres], vec![0.0; nres], vec![0.0; nres]],
            wbytes: [vec![0.0; nres], vec![0.0; nres], vec![0.0; nres]],
            task_start: vec![0.0; self.g.tasks.len()],
            task_end: vec![0.0; self.g.tasks.len()],
            mid_iteration,
            ..Default::default()
        };
        let peak_macs: f64 = self
            .view
            .units
            .iter()
            .filter(|u| self.view.hw.units[u.unit].spec.kind.is_mac())
            .map(|u| {
                let s = &self.view.hw.units[u.unit].spec;
                s.precisions.iter().map(|m| s.kind.ops_per_cycle(m)).fold(0.0, f64::max) * self.clocks.hz(u.clock).unwrap_or(1.0e9)
            })
            .sum();
        let dram_bw: f64 = (0..nres).filter(|&r| self.view.resources[r].class == ResClass::Dram).map(|r| self.view.resources[r].capacity).sum();
        let mut t = 0.0f64;
        let mut gi = 0usize;
        let ngroups = self.g.groups.len();
        let mut sc = Scratch {
            busy_f: vec![0.0; nres],
            busy_e: vec![0.0; nres],
            mark: vec![false; nres],
            touched: vec![],
            prof_bytes: vec![0.0; self.g.profiles.len()],
            ptouched: vec![],
            dram_sh: vec![],
            onchip_sh: vec![],
            task_f: vec![0.0; nres],
            task_e: vec![0.0; nres],
            ttouched: vec![],
            lat: self.latencies(),
        };
        if !self.g.stack.is_empty() {
            let shares = |rs: Vec<usize>| -> Vec<(usize, f64)> {
                let tot: f64 = rs.iter().map(|&r| self.capacity(r)).sum();
                rs.into_iter().filter(|_| tot > 0.0).map(|r| (r, self.capacity(r) / tot)).collect()
            };
            sc.dram_sh = shares((0..nres).filter(|&r| self.view.resources[r].class == ResClass::Dram).collect());
            sc.onchip_sh = shares(
                self.view.shared_onchip().map(|s| self.view.groups[s].mems.iter().map(|&m| self.view.res_of_mem[m] as usize).collect()).unwrap_or_default(),
            );
        }
        while gi < ngroups {
            let mut gj = gi;
            while gj + 1 < ngroups && !self.g.groups[gj].barrier_after {
                gj += 1;
            }
            let seg = self.segment(gi, gj + 1, t, &inv_f, &inv_e, (&pm_f, &pm_e, &pm_arg), peak_macs, dram_bw, &mut out, &mut sc);
            t += seg.time_est;
            out.segs.push(seg);
            gi = gj + 1;
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn segment(
        &self,
        g0: usize,
        g1: usize,
        start: f64,
        inv_f: &[f64],
        inv_e: &[f64],
        (pm_f, pm_e, pm_arg): (&[f64], &[f64], &[u32]),
        peak_macs: f64,
        dram_bw: f64,
        out: &mut RunOut,
        sc: &mut Scratch,
    ) -> SegOut {
        let g = self.g;
        let (t0, t1) = (g.groups[g0].tasks.0 as usize, g.groups[g1 - 1].tasks.1 as usize);
        let iteration = g.groups[g0].iteration;
        let class = match iteration {
            None => PE,
            i if i == out.mid_iteration => MID,
            _ => OTHER,
        };
        let Scratch { busy_f, busy_e, mark, touched, prof_bytes, ptouched, dram_sh, onchip_sh, task_f, task_e, ttouched, lat } = sc;
        let mut touch = |r: usize| {
            if !mark[r] {
                mark[r] = true;
                touched.push(r as u32);
            }
        };
        let carries_bytes = self.view.resource_carries_bytes();
        let n = t1 - t0;
        let (mut fin_f, mut fin_e) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut st_f, mut st_e) = (vec![0.0f64; n], vec![0.0f64; n]);
        let mut best_pred = vec![u32::MAX; n];
        // Per task: the resource its estimated duration comes from (critical-path attribution).
        let mut dom = vec![u32::MAX; n];
        for ti in t0..t1 {
            let task = &g.tasks[ti];
            let (mut df, mut de) = (0.0f64, 0.0f64);
            let dems = g.demands_of(task);
            for a in dems {
                match *a {
                    Amount::Res(r, x) => {
                        let r = r as usize;
                        touch(r);
                        busy_f[r] += x * inv_f[r];
                        busy_e[r] += x * inv_e[r];
                        if carries_bytes[r] {
                            out.bytes[class][r] += x;
                        }
                    }
                    Amount::Prof(p, x) => {
                        if prof_bytes[p as usize] == 0.0 {
                            ptouched.push(p);
                        }
                        prof_bytes[p as usize] += x;
                    }
                }
            }
            // The task holds each resource for the sum of its demands there (profiles of consecutive hops share
            // their intermediate memory); a lone demand's busiest resource is precomputed.
            match dems {
                [Amount::Res(r, x)] => {
                    (df, de) = (x * inv_f[*r as usize], x * inv_e[*r as usize]);
                    if de > 0.0 {
                        dom[ti - t0] = *r;
                    }
                }
                [Amount::Prof(p, x)] => {
                    (df, de) = (x * pm_f[*p as usize], x * pm_e[*p as usize]);
                    if de > 0.0 {
                        dom[ti - t0] = pm_arg[*p as usize];
                    }
                }
                _ => {
                    let mut add = |r: u32, x: f64| {
                        if task_f[r as usize] == 0.0 && task_e[r as usize] == 0.0 {
                            ttouched.push(r);
                        }
                        task_f[r as usize] += x * inv_f[r as usize];
                        task_e[r as usize] += x * inv_e[r as usize];
                    };
                    for a in dems {
                        match *a {
                            Amount::Res(r, x) => add(r, x),
                            Amount::Prof(p, x) => g.profiles[p as usize].entries.iter().for_each(|&(r, f)| add(r, x * f)),
                        }
                    }
                    for &r in ttouched.iter() {
                        let (xf, xe) = (std::mem::take(&mut task_f[r as usize]), std::mem::take(&mut task_e[r as usize]));
                        df = df.max(xf);
                        if xe > de {
                            de = xe;
                            dom[ti - t0] = r;
                        }
                    }
                    ttouched.clear();
                }
            }
            let (mut sf, mut se, mut ff, mut fe, mut bp) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, u32::MAX);
            for &p in g.preds_of(task) {
                let p = p as usize;
                if p < t0 {
                    continue;
                }
                let pt = &g.tasks[p];
                let (pf, pe) = (fin_f[p - t0], fin_e[p - t0]);
                if streaming(g, pt, task) {
                    sf = sf.max(st_f[p - t0] + lat[p]);
                    se = se.max(st_e[p - t0] + lat[p]);
                    ff = ff.max(pf);
                    fe = fe.max(pe);
                } else {
                    sf = sf.max(pf);
                    se = se.max(pe);
                }
                if bp == u32::MAX || pe > fin_e[bp as usize - t0] {
                    bp = p as u32;
                }
            }
            st_f[ti - t0] = sf;
            st_e[ti - t0] = se;
            fin_f[ti - t0] = (sf + df).max(ff) + lat[ti];
            fin_e[ti - t0] = (se + de).max(fe) + lat[ti];
            best_pred[ti - t0] = bp;
            out.task_start[ti] = se;
        }
        // Profiles, then resources, in index order (as a scan over all of them would visit them).
        ptouched.sort_unstable();
        ptouched.dedup();
        for &p in ptouched.iter() {
            let b = std::mem::take(&mut prof_bytes[p as usize]);
            if b == 0.0 {
                continue;
            }
            for &(r, f) in &g.profiles[p as usize].entries {
                let r = r as usize;
                touch(r);
                busy_f[r] += b * f * inv_f[r];
                busy_e[r] += b * f * inv_e[r];
                out.bytes[class][r] += b * f;
            }
            for &(r, f) in &g.profiles[p as usize].writes {
                out.wbytes[class][r as usize] += b * f;
            }
        }
        ptouched.clear();
        touched.sort_unstable();
        if self.params.t_dram_ramp > 0.0 {
            for &r in touched.iter() {
                let b = &mut busy_e[r as usize];
                if *b > 0.0 && self.view.resources[r as usize].class == ResClass::Dram {
                    *b = dram_ramp(*b, self.params.t_dram_ramp);
                }
            }
        }
        let argmax = |v: &[f64]| v.iter().enumerate().fold((0usize, 0.0f64), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
        // Untouched resources are idle (zero) and never win: the first strict maximum over the touched ones in
        // index order is the scan's.
        let argmax_touched = |v: &[f64]| touched.iter().fold((0usize, 0.0f64), |a, &r| if v[r as usize] > a.1 { (r as usize, v[r as usize]) } else { a });
        let (_, a1f) = argmax_touched(busy_f);
        let (rstar, a1e) = argmax_touched(busy_e);
        let lf = fin_f.iter().copied().fold(0.0, f64::max);
        let (last, le) = argmax(&fin_e);
        let base = a1e.max(le);
        let mut path = vec![];
        let mut cur = if n > 0 { (last + t0) as u32 } else { u32::MAX };
        while cur != u32::MAX && (cur as usize) >= t0 {
            path.push(cur as usize);
            cur = best_pred[cur as usize - t0];
        }
        let mut contention = 0.0f64;
        if base > 0.0 && self.params.contention_scale > 0.0 {
            let mut on_path: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
            for &ti in &path {
                for a in g.demands_of(&g.tasks[ti]) {
                    match *a {
                        Amount::Res(r, x) => *on_path.entry(r as usize).or_default() += x * inv_e[r as usize],
                        Amount::Prof(p, x) => {
                            for &(r, f) in &g.profiles[p as usize].entries {
                                *on_path.entry(r as usize).or_default() += x * f * inv_e[r as usize];
                            }
                        }
                    }
                }
            }
            let mut cp = 0.0f64;
            for (r, d) in on_path {
                let rho = ((busy_e[r] - d) / base).max(0.0);
                cp = cp.max(d * rho / (2.0 * (1.0 - rho.min(self.params.rho_max))));
            }
            contention = ((le + cp * self.params.contention_scale) - base).max(0.0);
        }
        let core = base;
        for &r in touched.iter() {
            out.busy[class][r as usize] += busy_e[r as usize];
        }
        for ti in t0..t1 {
            out.task_start[ti] += start;
            out.task_end[ti] = start + fin_e[ti - t0];
        }
        let p = self.params;
        let launch = g.groups[g0].launch;
        let barrier = g.groups[g1 - 1].barrier_after;
        let (min_k, per_group, sync) = match launch {
            LaunchKind::HostLaunch => (p.t_min_kernel, p.t_gap, 0.0),
            LaunchKind::DeviceQueued => (0.0, p.t_dispatch, 0.0),
            LaunchKind::StaticProgram => (0.0, 0.0, if barrier { p.t_sync } else { 0.0 }),
        };
        // Groups without tasks (layout views) launch nothing.
        let ngr = g.groups[g0..g1].iter().filter(|x| x.tasks.1 > x.tasks.0).count() as f64;
        let min_k = if ngr > 0.0 { min_k } else { 0.0 };
        let pad = |c: f64| (min_k - c).max(0.0);
        // Stack kernels run after the group's own kernels on the same queue: each takes its traffic time or the
        // minimum kernel time, plus a gap.
        let (mut stack_dram, mut stack_onchip, mut stack_overhead, mut stack_floor) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (k0, k1) = (g.stack.partition_point(|k| (k.group as usize) < g0), g.stack.partition_point(|k| (k.group as usize) < g1));
        let min_sk = if launch == LaunchKind::HostLaunch { p.t_min_kernel } else { 0.0 };
        for k in &g.stack[k0..k1] {
            let onchip = k.onchip && !onchip_sh.is_empty();
            let sh: &[(usize, f64)] = if onchip { onchip_sh } else { dram_sh };
            let (mut mf, mut me) = (0.0f64, 0.0f64);
            for &(r, s) in sh {
                mf = mf.max(k.bytes * s * inv_f[r]);
                me = me.max(k.bytes * s * inv_e[r]);
            }
            let raw = me;
            if !onchip && p.t_dram_ramp > 0.0 && me > 0.0 {
                me = dram_ramp(me, p.t_dram_ramp);
            }
            for &(r, s) in sh {
                out.busy[class][r] += k.bytes * s * inv_e[r] * if raw > 0.0 { me / raw } else { 1.0 };
                out.bytes[class][r] += k.bytes * s;
            }
            if onchip {
                stack_onchip += me;
            } else {
                stack_dram += me;
            }
            stack_overhead += (min_sk - me).max(0.0) + per_group;
            stack_floor += mf.max(min_sk) + per_group;
        }
        let overhead_est = pad(core + contention) + ngr * per_group + sync;
        let overhead_floor = pad(a1f.max(lf)) + ngr * per_group + sync;
        let kind = match launch {
            LaunchKind::HostLaunch if pad(core + contention) > ngr * per_group => OverheadKind::MinKernel,
            LaunchKind::HostLaunch => OverheadKind::Gap,
            LaunchKind::DeviceQueued => OverheadKind::Dispatch,
            LaunchKind::StaticProgram => OverheadKind::Sync,
        };
        let (useful, offchip): (u128, u128) = g
            .ops
            .iter()
            .filter(|o| (g0..g1).contains(&(o.group as usize)))
            .fold((0, 0), |a, o| (a.0 + o.useful_macs, a.1 + o.compulsory_offchip));
        let a0 = (useful as f64 / peak_macs.max(1.0)).max(offchip as f64 / dram_bw.max(1.0));
        let class_of = |r: usize| match self.view.resources[r].class {
            ResClass::Compute => BindingClass::Compute,
            ResClass::Dram => BindingClass::Dram,
            ResClass::Link => BindingClass::Link,
            ResClass::Mem => BindingClass::Port,
            ResClass::Sequencer => BindingClass::Overhead,
        };
        let mut terms: Vec<(BindingClass, f64, Option<usize>)> = vec![];
        for (r, b) in touched.iter().map(|&r| (r as usize, busy_e[r as usize])).filter(|(_, b)| *b > 0.0) {
            let c = class_of(r);
            match terms.iter_mut().find(|t| t.0 == c) {
                Some(t) if b > t.1 => *t = (c, b, Some(r)),
                Some(_) => {}
                None => terms.push((c, b, Some(r))),
            }
        }
        for &r in touched.iter() {
            busy_f[r as usize] = 0.0;
            busy_e[r as usize] = 0.0;
            mark[r as usize] = false;
        }
        touched.clear();
        // The critical path's time belongs to the resources its tasks wait on (each task's dominant demand);
        // only latencies are pure dependency.
        let (mut w_lat, mut w_tot) = (0.0f64, 0.0f64);
        let mut on_path: Vec<(BindingClass, f64, usize, f64)> = vec![];
        for &ti in &path {
            let d = (fin_e[ti - t0] - st_e[ti - t0] - lat[ti]).max(0.0);
            w_lat += lat[ti];
            w_tot += lat[ti] + d;
            let r = dom[ti - t0];
            if r != u32::MAX && d > 0.0 {
                let c = class_of(r as usize);
                match on_path.iter_mut().find(|x| x.0 == c) {
                    Some(x) => {
                        x.1 += d;
                        if d > x.3 {
                            (x.2, x.3) = (r as usize, d);
                        }
                    }
                    None => on_path.push((c, d, r as usize, d)),
                }
            }
        }
        if w_tot > 0.0 {
            for (c, d, r, _) in on_path {
                let v = le * d / w_tot;
                match terms.iter_mut().find(|t| t.0 == c) {
                    Some(t) if v > t.1 => t.1 = v,
                    Some(_) => {}
                    None => terms.push((c, v, Some(r))),
                }
            }
        }
        terms.push((BindingClass::Dependency, if w_tot > 0.0 { le * w_lat / w_tot } else { le }, None));
        terms.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let bind = |c: BindingClass, r: Option<usize>| {
            let id = |r: Option<usize>| self.view.resource_ids()[r.unwrap_or(rstar)].clone();
            match c {
                BindingClass::Compute => Binding::Compute { resource: id(r) },
                BindingClass::Dram => Binding::Dram { resource: id(r) },
                BindingClass::Link => Binding::Link { resource: id(r) },
                BindingClass::Port => Binding::MemPort { resource: id(r) },
                BindingClass::Dependency => Binding::Dependency,
                _ => Binding::Overhead { overhead: OverheadKind::Dispatch },
            }
        };
        let (binding, runner_up, binding_resource) = match terms.as_slice() {
            [] => (Binding::Overhead { overhead: kind }, None, None),
            [b, rest @ ..] if b.1 > 0.0 => (
                bind(b.0, b.2),
                rest.first().filter(|x| x.1 > 0.0).map(|x| RunnerUp { binding: bind(x.0, x.2), ratio: x.1 / b.1 }),
                b.2.map(|r| r as u32),
            ),
            _ => (Binding::Overhead { overhead: kind }, None, None),
        };
        SegOut {
            groups: (g0, g1),
            iteration,
            start_s: start,
            a0,
            a1_floor: a1f,
            l_floor: lf,
            a1_est: a1e,
            l_est: le,
            contention,
            overhead: overhead_est,
            overhead_kind: kind,
            stack_dram,
            stack_onchip,
            stack_overhead,
            time_floor: a1f.max(lf) + overhead_floor + stack_floor,
            time_est: core + contention + overhead_est + stack_dram + stack_onchip + stack_overhead,
            binding,
            runner_up,
            binding_resource,
        }
    }
}

/// Busy time of a DRAM resource whose in-flight requests ramp linearly up over `t_r` at the start of a
/// segment and down over `t_r` at its end (memory-level parallelism builds as the kernel's work spreads):
/// `b + t_r` once the stream reaches steady bandwidth (`b >= t_r`), `2 sqrt(b t_r)` for a triangle profile.
pub fn dram_ramp(b: f64, t_r: f64) -> f64 {
    if b >= t_r { b + t_r } else { 2.0 * (b * t_r).sqrt() }
}

/// Intra-op edges between transfers and compute stream tile by tile (double buffering): the successor may
/// start once the first data arrives and finishes no earlier than its predecessor. So do edges between
/// workload nodes fused into one group (03 §3.6, a stack that fuses elementwise nodes into a contraction:
/// the fusion consumes tiles as they are produced). Other cross-op edges (including those between the ops
/// of one multi-op node) and reduce edges are finish-to-start. This keeps `L` a lower bound on a
/// tile-pipelined execution (03 §4.6).
pub fn streaming(g: &TaskGraph, pred: &Task, t: &Task) -> bool {
    let fused_nodes = || pred.group == t.group && g.groups[t.group as usize].fused && g.op_node[pred.op as usize] != g.op_node[t.op as usize];
    (pred.op == t.op || fused_nodes()) && pred.kind != TaskKind::Reduce && t.kind != TaskKind::Reduce
}

/// Resource paths contain `->` and `[`; result ids follow `[a-z0-9_.-]+` (00 conventions).
pub fn sanitize(path: &str) -> String {
    kiln_map::hwview::sanitize(path)
}
