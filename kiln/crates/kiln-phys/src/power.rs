//! Power model (04 §4.7, §8): V/f tables, leakage P_static(V, T), clock tree, DRAM background, board overheads,
//! the Tier A junction estimate (§9) and the phase power that 03 §4.5's clock solve bisects on.

use kiln_ir::common::Diagnostic;
use std::collections::BTreeMap;

use kiln_ir::hw::model::{ClockIx, ContainerKind, HwModel, NodeIx, ResIx};
use kiln_ir::hw::phys::{CapLevel, ClockDomain, CoolingClass, PowerPolicy};
use serde::Serialize;

use crate::characterize::Characterized;
use crate::params::Params;
use crate::tables::{TechNode, Tables};
use crate::wire::LinkCost;

/// Junction temperature beyond which the Tier A fixed point counts as diverged.
const T_RUNAWAY_C: f64 = 1000.0;

/// Monotone frequency -> voltage table of one clock domain (32 points, ascending in f).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VfTable {
    pub points: Vec<(f64, f64)>,
    /// Lowest frequency the controller may pick (01 `base`, else f(V_min)).
    pub floor_hz: f64,
    pub boost_hz: f64,
    /// True when 01 gave explicit V/f points (used verbatim between them).
    pub declared: bool,
}

impl VfTable {
    /// Alpha-power law `f(V) = k_f (V - V_th)^alpha / V` (Sakurai-Newton) through `(f_ref, v_ref)`.
    fn law(v_th: f64, alpha: f64, f_ref: f64, v_ref: f64) -> impl Fn(f64) -> f64 {
        let k = f_ref * v_ref / (v_ref - v_th).max(1e-3).powf(alpha);
        move |v: f64| k * (v - v_th).max(1e-6).powf(alpha) / v
    }

    pub fn build(c: &ClockDomain, n: &TechNode, v_th: f64, alpha: f64) -> VfTable {
        let boost = c.freq.0;
        let mut pts: Vec<(f64, f64)> = c.vf.iter().map(|p| (p.freq.0, p.voltage.0)).collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let declared = !pts.is_empty();
        if pts.is_empty() {
            let v_b = c.voltage.map_or(n.v_max, |v| v.0);
            let f = Self::law(v_th, alpha, boost, v_b);
            for i in 0..32 {
                let v = n.v_min + (v_b - n.v_min) * f64::from(i) / 31.0;
                pts.push((f(v), v));
            }
        } else {
            // Extend below the lowest declared point down to V_min with the law through that point.
            let (f0, v0) = pts[0];
            let f = Self::law(v_th, alpha, f0, v0);
            let mut below = vec![];
            for i in 0..8 {
                let v = n.v_min.min(v0) + (v0 - n.v_min.min(v0)) * f64::from(i) / 8.0;
                if v < v0 - 1e-9 {
                    below.push((f(v), v));
                }
            }
            below.extend(pts);
            pts = below;
            if pts.last().is_some_and(|p| p.0 < boost * (1.0 - 1e-9)) {
                let (fl, vl) = *pts.last().expect("nonempty");
                let f = Self::law(v_th, alpha, fl, vl);
                let mut v = vl;
                while f(v) < boost && v < 1.5 {
                    v += 0.005;
                }
                pts.push((boost, v));
            }
        }
        pts.retain(|p| p.0 <= boost * (1.0 + 1e-9));
        pts.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-6 * b.0.abs());
        let floor = c.base.map_or(pts[0].0, |b| b.0);
        VfTable { points: pts, floor_hz: floor, boost_hz: boost, declared }
    }

    pub fn voltage(&self, f: f64) -> f64 {
        let p = &self.points;
        if f <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            if f <= w[1].0 {
                let t = (f - w[0].0) / (w[1].0 - w[0].0).max(1e-9);
                return w[0].1 + t * (w[1].1 - w[0].1);
            }
        }
        p[p.len() - 1].1
    }

    /// 32 candidate frequencies between floor and boost for the bisection (ascending).
    pub fn grid(&self) -> Vec<f64> {
        let (lo, hi) = (self.floor_hz.min(self.boost_hz), self.boost_hz);
        (0..32).map(|i| lo + (hi - lo) * f64::from(i) / 31.0).collect()
    }
}

/// Technologies of the blocks each clock domain powers (the nodes [`PowerModel::scoped`] charges to a domain,
/// PHYs and DRAM stacks aside) with their area, ascending by id; the first die's when a domain powers none.
fn domain_techs(hw: &HwModel, ch: &Characterized) -> Vec<Vec<(&'static TechNode, f64)>> {
    let t = Tables::get();
    let (dom, main) = node_clocks(hw);
    let phy: Vec<usize> = ch.phys.iter().map(|p| p.node).collect();
    let mut out: Vec<Vec<(&TechNode, f64)>> = vec![vec![]; hw.clocks.len()];
    for (i, (np, d)) in ch.nodes.iter().zip(&dom).enumerate() {
        if (np.area_um2 <= 0.0 && np.leak_w == 0.0 && np.flops == 0.0 && np.ctrl_ge == 0.0)
            || phy.contains(&i)
            || matches!(hw.nodes[i].ix, NodeIx::Mem(m) if ch.mems[m].dram)
        {
            continue;
        }
        let (Some(d), Some(n)) = (d.or(main).filter(|&d| d < out.len()), t.node(&ch.tech[i])) else { continue };
        match out[d].iter_mut().find(|x| x.0.id == n.id) {
            Some(x) => x.1 += np.area_um2,
            None => out[d].push((n, np.area_um2)),
        }
    }
    let die_tech = ch.tech.get(hw.tree.iter().find(|x| x.kind == ContainerKind::Die).map_or(0, |x| x.node)).cloned().unwrap_or_default();
    let first = t.node(&die_tech).or_else(|| t.node("tsmc_n7")).expect("N7");
    for v in &mut out {
        v.sort_by(|a, b| a.0.id.cmp(&b.0.id));
        if v.is_empty() {
            v.push((first, 0.0));
        }
    }
    out
}

/// The technology whose V/f law and nominal voltage a domain takes: the one with most area among its blocks.
fn primary<'a>(techs: &[(&'a TechNode, f64)]) -> &'a TechNode {
    techs.iter().fold(techs[0], |a, &b| if b.1 > a.1 { b } else { a }).0
}

/// 04 §4.7: the V/f curve is defined on the node's `[V_min, V_max]`; a declared voltage outside the range of any
/// technology the domain clocks is `E-PHYS-VF-VOLTAGE` (01 checks only that the points are monotone).
pub fn vf_problems(hw: &HwModel, ch: &Characterized) -> Vec<Diagnostic> {
    let techs = domain_techs(hw, ch);
    hw.clocks
        .iter()
        .zip(&techs)
        .filter_map(|(c, ts)| {
            let (v, n) = c
                .spec
                .vf
                .iter()
                .map(|p| p.voltage.0)
                .chain(c.spec.voltage.map(|v| v.0))
                .find_map(|v| ts.iter().find(|(n, _)| !(v >= n.v_min - 1e-9 && v <= n.v_max + 1e-9)).map(|(n, _)| (v, *n)))?;
            Some(
                Diagnostic::error("E-PHYS-VF-VOLTAGE", format!("{}: {v} V is outside the {} operating range [{}, {}] V", c.path, n.id, n.v_min, n.v_max))
                    .at(&c.path)
                    .hint("declare V/f points within the voltage range of every technology the clock drives, or omit them to use the derived curve"),
            )
        })
        .collect()
}

/// Leakage of one technology's blocks in a domain at V_nom, 85 C, with that technology's sensitivities.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Leak {
    pub tech: String,
    pub v_nom: f64,
    pub k_v: f64,
    pub k_t: f64,
    pub w: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DomainPower {
    pub vf: VfTable,
    /// Nominal voltage of the domain's primary technology (energies of blocks of other technologies scale with
    /// their own, see [`PowerModel::dyn_scale_tech`]).
    pub v_nom: f64,
    /// Nominal voltage of every technology the domain clocks.
    pub v_noms: Vec<(String, f64)>,
    /// Leakage of the domain's blocks at V_nom, 85 C, in total and by technology.
    pub leak_w: f64,
    pub leak: Vec<Leak>,
    /// Clock-tree capacitance (kappa_clk included), F.
    pub c_clk_f: f64,
    /// Clock load of pipelined links' flops ([`Pipe`]), F.
    pub c_pipe_f: f64,
    /// Switched control-logic capacitance per cycle while running (alpha_ctrl included), F.
    pub c_ctrl_f: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PowerModel {
    pub domains: Vec<DomainPower>,
    /// PHY bias and other clock-independent leakage, W.
    pub indep_leak_w: f64,
    pub dram_background_w: f64,
    pub board_w: f64,
    pub eta_vr: f64,
    pub clk_ungated: f64,
    /// Enforced (published) cap, W.
    pub cap_w: Option<f64>,
    /// Assumed caps: (nominal, lo, hi) W; never enforced.
    pub assumed_cap_w: Option<(f64, f64, f64)>,
    pub cap_level: CapLevel,
    pub die_mm2: f64,
    pub t_inlet_c: f64,
    pub r_ja_k_mm2_w: f64,
    pub tj_max_c: f64,
    pub q_avg_max: f64,
}

/// One enforced power cap (01 §12): the power of its members at `level` stays at or under `cap_w`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapDomain {
    pub path: String,
    pub cap_w: f64,
    pub level: CapLevel,
    /// Clocks its DVFS controller may lower (01 `clocks`, else every clock of the members, the MAC clock first);
    /// none under a fixed or duty-cycle policy.
    pub clocks: Vec<ClockIx>,
    /// Arena nodes inside the members.
    pub nodes: Vec<bool>,
    /// The members' own power terms (leakage, clock tree, control, DRAM background, board, die area).
    pub power: PowerModel,
}

/// The pipeline flops of one physical on-die link (a shared resource's widest channel): the node driving it, its
/// clock and the clock energy of its flops per cycle at its technology's V_nom (width x stages x idle flop energy).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Pipe {
    pub node: usize,
    pub clock: Option<ClockIx>,
    pub e_cycle_j: f64,
}

/// A region with its own Tier A thermal and density check (04 §9): an enabled package (its dies' junction
/// estimate over its own cooling) or an enabled die (its own power over its own area).
#[derive(Clone, Debug)]
pub struct ThermalZone {
    /// Index into the floorplan's (and report's) dies for a die zone.
    pub die: Option<usize>,
    /// Arena nodes inside the zone.
    pub nodes: Vec<bool>,
    pub power: PowerModel,
}

impl ThermalZone {
    /// One zone per enabled package holding an enabled die, then one per enabled die.
    pub fn build_all(hw: &HwModel, ch: &Characterized, params: &Params, dies: &[(usize, f64)], clocked_um2: &[f64], pipes: &[Pipe]) -> Vec<ThermalZone> {
        let subtree = |root: usize| -> Vec<bool> { (0..hw.nodes.len()).map(|i| std::iter::successors(Some(i), |&x| hw.nodes[x].parent).any(|x| x == root)).collect() };
        let zone = |die: Option<usize>, nodes: Vec<bool>| {
            let power = PowerModel::scoped(hw, ch, params, dies, clocked_um2, pipes, Some((&nodes, Some(0.0))));
            ThermalZone { die, nodes, power }
        };
        let live: Vec<(usize, usize)> = dies.iter().enumerate().filter(|(_, d)| hw.nodes[d.0].enabled).map(|(i, d)| (i, d.0)).collect();
        let mut zones: Vec<ThermalZone> = hw
            .tree
            .iter()
            .filter(|c| c.kind == ContainerKind::Package && hw.nodes[c.node].enabled)
            .map(|c| subtree(c.node))
            .filter(|n| live.iter().any(|&(_, d)| n[d]))
            .map(|n| zone(None, n))
            .collect();
        zones.extend(live.iter().map(|&(i, d)| zone(Some(i), subtree(d))));
        zones
    }
}

/// The pipelined links of `hw` between enabled endpoints, one per physical link resource.
pub fn pipes(hw: &HwModel, links: &[LinkCost]) -> Vec<Pipe> {
    let mut by_res: BTreeMap<ResIx, Pipe> = BTreeMap::new();
    for (c, l) in hw.channels.iter().zip(links).filter(|(_, l)| l.e_j_per_cycle_idle > 0.0) {
        let (s, d) = (hw.node_of(c.src), hw.node_of(c.dst));
        if !hw.nodes[s].enabled || !hw.nodes[d].enabled {
            continue;
        }
        let f = c.clock.and_then(|k| hw.clocks.get(k)).map_or(1e9, |k| k.spec.freq.0);
        let width = c.width_bits.map_or(c.bandwidth.map_or(0.0, |b| b.0) * 8.0 / f, f64::from).max(1.0);
        let p = Pipe { node: s, clock: c.clock, e_cycle_j: width * l.e_j_per_cycle_idle };
        let x = by_res.entry(c.resource).or_insert(p);
        if p.e_cycle_j > x.e_cycle_j {
            *x = p;
        }
    }
    by_res.into_values().collect()
}

/// Clock domain of every node (its own, else the nearest ancestor's) and the MAC units' clock.
fn node_clocks(hw: &HwModel) -> (Vec<Option<ClockIx>>, Option<ClockIx>) {
    let mut dom: Vec<Option<ClockIx>> = vec![None; hw.nodes.len()];
    for i in 0..hw.nodes.len() {
        let own = match hw.nodes[i].ix {
            NodeIx::Unit(u) => hw.units[u].clock,
            NodeIx::Mem(m) => hw.memories[m].clock,
            NodeIx::Block(b) => hw.blocks[b].clock,
            NodeIx::Container(c) => hw.tree[c].clock,
            NodeIx::Net(n) => hw.networks[n].clock,
            _ => None,
        };
        dom[i] = own.or_else(|| hw.nodes[i].parent.and_then(|p| dom[p]));
    }
    let main = hw.units.iter().find(|u| u.spec.kind.is_mac()).and_then(|u| u.clock).or_else(|| hw.tree.iter().find(|c| c.kind == ContainerKind::Die).and_then(|c| c.clock));
    (dom, main)
}

/// Tier A thermal limits (04 §9) of the packages holding the scoped dies.
struct Thermal {
    die_mm2: f64,
    r_ja_k_mm2_w: f64,
    tj_max_c: f64,
    q_avg_max: f64,
}

impl Thermal {
    /// Each enabled package holding a scoped die (every die when none is in scope) contributes its own cooling,
    /// `theta_ja` and `tj_max`; the worst package sets each limit. With power spread over the dies by area, a
    /// package's junction rise is `P * theta_ja * A_pkg / A`, so `theta_ja` (K/W) enters as `theta_ja * A_pkg`.
    fn of(hw: &HwModel, params: &Params, dies: &[(usize, f64)], inside: &impl Fn(usize) -> bool) -> Thermal {
        let live: Vec<(usize, f64)> = dies.iter().copied().filter(|d| hw.nodes[d.0].enabled).collect();
        let scoped: Vec<(usize, f64)> = live.iter().copied().filter(|d| inside(d.0)).collect();
        let scoped = if scoped.iter().map(|d| d.1).sum::<f64>() > 0.0 { scoped } else { live };
        let die_mm2: f64 = scoped.iter().map(|d| d.1).sum();
        let package_of = |n: usize| {
            std::iter::successors(Some(n), |&x| hw.nodes[x].parent).find_map(|x| match hw.nodes[x].ix {
                NodeIx::Container(c) if hw.tree[c].kind == ContainerKind::Package => Some(c),
                _ => None,
            })
        };
        let mut pkgs: Vec<(Option<usize>, f64)> = vec![];
        for &(n, a) in &scoped {
            let p = package_of(n);
            match pkgs.iter_mut().find(|x| x.0 == p) {
                Some(x) => x.1 += a,
                None => pkgs.push((p, a)),
            }
        }
        let (r_air, r_liquid) = (params.get("r_ja_k_mm2_w", None), params.get("r_ja_liquid_k_mm2_w", None));
        let (q_air, q_liquid) = (params.get("q_avg_max_air", None), params.get("q_avg_max_liquid", None));
        let tj_default = params.get("tj_max_c", None);
        let worst = pkgs
            .iter()
            .map(|&(p, a)| {
                let th = p.and_then(|c| hw.tree[c].package.as_ref()).and_then(|p| p.power.as_ref()).and_then(|p| p.thermal.as_ref());
                let liquid = th.is_some_and(|t| matches!(t.cooling, CoolingClass::LiquidColdPlate | CoolingClass::Immersion));
                let r = th.and_then(|t| t.theta_ja).map_or(if liquid { r_liquid } else { r_air }, |k| k * a);
                (r, th.map_or(tj_default, |t| t.tj_max_c), if liquid { q_liquid } else { q_air })
            })
            .reduce(|a, b| (a.0.max(b.0), a.1.min(b.1), a.2.min(b.2)));
        let (r_ja_k_mm2_w, tj_max_c, q_avg_max) = worst.unwrap_or((r_air, tj_default, q_air));
        Thermal { die_mm2, r_ja_k_mm2_w, tj_max_c, q_avg_max }
    }
}

impl CapDomain {
    /// The enforced caps of `hw` (an assumed cap never throttles, 04 §8.1). `dies` holds every die's node and
    /// envelope area, mm^2.
    pub fn build_all(hw: &HwModel, ch: &Characterized, params: &Params, dies: &[(usize, f64)], clocked_um2: &[f64], pipes: &[Pipe]) -> Vec<CapDomain> {
        let (dom, main) = node_clocks(hw);
        hw.power_domains
            .iter()
            .filter(|p| p.assumed.is_none())
            .map(|p| {
                let roots: Vec<usize> = p.members.iter().map(|&c| hw.tree[c].node).collect();
                let nodes: Vec<bool> = (0..hw.nodes.len()).map(|i| std::iter::successors(Some(i), |&x| hw.nodes[x].parent).any(|x| roots.contains(&x))).collect();
                let clocks = if p.policy != PowerPolicy::Dvfs {
                    vec![]
                } else if !p.clocks.is_empty() {
                    p.clocks.clone()
                } else {
                    let mut cs: Vec<ClockIx> = (0..hw.nodes.len()).filter(|&i| nodes[i] && hw.nodes[i].enabled).filter_map(|i| dom[i]).collect();
                    cs.sort_unstable_by_key(|&c| (Some(c) != main, c));
                    cs.dedup();
                    cs
                };
                let power = PowerModel::scoped(hw, ch, params, dies, clocked_um2, pipes, Some((&nodes, p.idle.map(|w| w.0))));
                CapDomain { path: p.path.clone(), cap_w: p.cap.0, level: p.level.unwrap_or(CapLevel::Board), clocks, nodes, power }
            })
            .collect()
    }
}

/// Energies of one phase window as the engine accounts them (core-domain energies already at the plan's V).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PhaseEnergy {
    pub makespan_s: f64,
    /// Core-domain dynamic energy (compute, on-chip memories, on-die links), J.
    pub core_dyn_j: f64,
    /// Die-level clock-independent energy (DRAM/IO PHYs, off-die links), J.
    pub indep_j: f64,
    /// DRAM core + IO energy in the stacks (board level), J.
    pub dram_j: f64,
    /// MAC-pipe activity in [0, 1] (clock gating of idle units).
    pub activity: f64,
    /// Fraction of the window the compute dies run a phase (control logic switching); 0 when idle.
    pub busy: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PowerBreakdown {
    pub dyn_core_w: f64,
    pub indep_w: f64,
    pub static_w: f64,
    pub clock_w: f64,
    /// Control-logic switching while running (alpha_ctrl).
    pub ctrl_w: f64,
    pub dram_w: f64,
    pub board_fixed_w: f64,
    pub vr_loss_w: f64,
    pub chip_w: f64,
    pub package_w: f64,
    pub board_w: f64,
    pub t_j_c: f64,
    /// No stable junction temperature: the leakage-temperature loop gain reaches 1 (04 §8 E-PHYS-THERMAL-RUNAWAY).
    pub runaway: bool,
}

impl PowerBreakdown {
    /// The power the envelope cap applies to.
    pub fn at(&self, level: CapLevel) -> f64 {
        match level {
            CapLevel::Die => self.chip_w,
            CapLevel::Package => self.package_w,
            CapLevel::Board => self.board_w,
        }
    }

}

impl PowerModel {
    /// The whole design's power terms; `dies` holds every die's node and envelope area, mm^2.
    pub fn build(hw: &HwModel, ch: &Characterized, params: &Params, dies: &[(usize, f64)], clocked_um2: &[f64], pipes: &[Pipe]) -> PowerModel {
        Self::scoped(hw, ch, params, dies, clocked_um2, pipes, None)
    }

    /// The power terms of the whole design, or of the nodes in `scope.0` with board power `scope.1` (a cap's
    /// declared idle power, else the board overhead of its packages).
    #[allow(clippy::too_many_arguments)]
    pub fn scoped(hw: &HwModel, ch: &Characterized, params: &Params, dies: &[(usize, f64)], clocked_um2: &[f64], pipes: &[Pipe], scope: Option<(&[bool], Option<f64>)>) -> PowerModel {
        let t = Tables::get();
        let nc = hw.clocks.len();
        let (dom, main) = node_clocks(hw);
        let inside = |i: usize| scope.is_none_or(|s| s.0[i]);
        let techs = domain_techs(hw, ch);
        let mut domains: Vec<DomainPower> = hw
            .clocks
            .iter()
            .zip(&techs)
            .map(|(c, ts)| {
                let n = primary(ts);
                DomainPower {
                    vf: VfTable::build(&c.spec, n, params.get("v_th", Some(&n.id)), params.get("alpha", Some(&n.id))),
                    v_nom: n.vdd_nom,
                    v_noms: ts.iter().map(|(x, _)| (x.id.clone(), x.vdd_nom)).collect(),
                    leak_w: 0.0,
                    leak: vec![],
                    c_clk_f: 0.0,
                    c_pipe_f: 0.0,
                    c_ctrl_f: 0.0,
                }
            })
            .collect();
        let phy_nodes: Vec<usize> = ch.phys.iter().map(|p| p.node).collect();
        let mut indep = 0.0;
        let kclk = params.get("kappa_clk", None);
        let actl = params.get("alpha_ctrl", None);
        for &i in ch.order.iter().filter(|&&i| inside(i)) {
            let np = &ch.nodes[i];
            if np.leak_w == 0.0 && np.flops == 0.0 && clocked_um2[i] == 0.0 && np.ctrl_ge == 0.0 {
                continue;
            }
            let n = t.node(&ch.tech[i]).or_else(|| t.node("tsmc_n7")).expect("N7");
            let d = dom[i].or(main).filter(|&d| d < nc);
            if phy_nodes.contains(&i) || d.is_none() {
                indep += np.leak_w;
                continue;
            }
            let d = d.expect("domain");
            let dp = &mut domains[d];
            dp.leak_w += np.leak_w;
            let lt = techs[d].iter().find(|x| x.0.id == n.id).map_or_else(|| primary(&techs[d]), |x| x.0);
            match dp.leak.iter_mut().find(|l| l.tech == lt.id) {
                Some(l) => l.w += np.leak_w,
                None => dp.leak.push(Leak { tech: lt.id.clone(), v_nom: lt.vdd_nom, k_v: lt.k_v, k_t: lt.k_t, w: np.leak_w }),
            }
            if hw.nodes[i].enabled {
                dp.c_ctrl_f += actl * np.ctrl_ge * n.c_ge_ff * 1e-15;
                dp.c_clk_f += kclk * (n.c_clk_flop_ff * 1e-15 * np.flops + n.c_clk_area_pf_mm2 * 1e-12 * clocked_um2[i] * 1e-6);
            }
        }
        // Summed in value order: arena and resource order (P4) cannot move the total.
        let mut pipe_c: Vec<(usize, f64)> = pipes
            .iter()
            .filter(|p| inside(p.node))
            .filter_map(|p| {
                let d = p.clock.or(dom[p.node]).or(main).filter(|&d| d < nc)?;
                let n = t.node(&ch.tech[p.node]).or_else(|| t.node("tsmc_n7")).expect("N7");
                Some((d, p.e_cycle_j / (n.vdd_nom * n.vdd_nom)))
            })
            .collect();
        pipe_c.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        for (d, c) in pipe_c {
            domains[d].c_pipe_f += c;
        }
        let dram_bg: f64 = ch.mems.iter().zip(&hw.memories).filter(|(_, m)| inside(m.node)).map(|(e, _)| e.background_w).sum();
        let caps: Vec<_> = hw.power_domains.iter().collect();
        // An assumed (unpublished) cap never throttles (04 §8.1); its range is reported.
        let enforced: Vec<_> = caps.iter().filter(|p| p.assumed.is_none()).collect();
        let cap_w = (!enforced.is_empty()).then(|| enforced.iter().map(|p| p.cap.0).sum());
        let assumed_cap_w = caps
            .iter()
            .filter_map(|p| p.assumed.as_ref().map(|r| (p.cap.0, r.lo.0, r.hi.0)))
            .reduce(|a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let cap_level = caps.first().and_then(|p| p.level).unwrap_or(CapLevel::Board);
        let idle = match scope {
            Some((_, idle)) => idle,
            None => caps.iter().filter_map(|p| p.idle.map(|w| w.0)).reduce(|a, b| a + b),
        };
        let board = idle.unwrap_or_else(|| {
            let pk = hw.tree.iter().filter(|c| c.kind == ContainerKind::Package && hw.nodes[c.node].enabled && inside(c.node)).count().max(1) as f64;
            params.get("p_board_w", None) * pk
        });
        let th = Thermal::of(hw, params, dies, &inside);
        PowerModel {
            domains,
            indep_leak_w: indep,
            dram_background_w: dram_bg,
            board_w: board,
            eta_vr: params.get("eta_vr", None),
            clk_ungated: params.get("clk_ungated", None),
            cap_w,
            assumed_cap_w,
            cap_level,
            die_mm2: th.die_mm2,
            t_inlet_c: params.get("t_inlet_c", None),
            r_ja_k_mm2_w: th.r_ja_k_mm2_w,
            tj_max_c: th.tj_max_c,
            q_avg_max: th.q_avg_max,
        }
    }

    pub fn voltage(&self, d: ClockIx, hz: f64) -> f64 {
        self.domains[d].vf.voltage(hz)
    }

    /// `(V(f) / V_nom)^2` of domain `d` at `hz` (03 §4.5 dynamic-energy rescale).
    pub fn dyn_scale(&self, d: ClockIx, hz: f64) -> f64 {
        let dp = &self.domains[d];
        (dp.vf.voltage(hz) / dp.v_nom).powi(2)
    }

    /// [`Self::dyn_scale`] for a block of technology `tech` (characterized at its own V_nom).
    pub fn dyn_scale_tech(&self, d: ClockIx, hz: f64, tech: &str) -> f64 {
        let dp = &self.domains[d];
        if dp.v_noms.len() <= 1 {
            return self.dyn_scale(d, hz);
        }
        let id = Tables::get().node(tech).map_or(tech, |n| n.id.as_str());
        let v_nom = dp.v_noms.iter().find(|x| x.0 == id).map_or(dp.v_nom, |x| x.1);
        (dp.vf.voltage(hz) / v_nom).powi(2)
    }

    pub fn p_static(&self, hz: &[f64], t_j: f64) -> f64 {
        let mut p = self.indep_leak_w;
        for (d, dp) in self.domains.iter().enumerate() {
            let v = dp.vf.voltage(hz.get(d).copied().unwrap_or(dp.vf.boost_hz));
            for l in &dp.leak {
                p += l.w * (v / l.v_nom) * (l.k_v * (v - l.v_nom)).exp() * (l.k_t * (t_j - 85.0)).exp();
            }
        }
        p
    }

    pub fn p_clock(&self, hz: &[f64], activity: f64) -> f64 {
        let g = self.clk_ungated + (1.0 - self.clk_ungated) * activity.clamp(0.0, 1.0);
        self.domains
            .iter()
            .enumerate()
            .map(|(d, dp)| {
                let f = hz.get(d).copied().unwrap_or(dp.vf.boost_hz);
                let v = dp.vf.voltage(f);
                (dp.c_clk_f + dp.c_pipe_f) * v * v * f * g
            })
            .sum()
    }

    pub fn p_ctrl(&self, hz: &[f64], busy: f64) -> f64 {
        self.domains
            .iter()
            .enumerate()
            .map(|(d, dp)| {
                let f = hz.get(d).copied().unwrap_or(dp.vf.boost_hz);
                let v = dp.vf.voltage(f);
                dp.c_ctrl_f * v * v * f * busy.clamp(0.0, 1.0)
            })
            .sum()
    }

    /// d[`Self::p_static`]/dT at `t_j`, W/K.
    fn p_static_dt(&self, hz: &[f64], t_j: f64) -> f64 {
        let mut p = 0.0;
        for (d, dp) in self.domains.iter().enumerate() {
            let v = dp.vf.voltage(hz.get(d).copied().unwrap_or(dp.vf.boost_hz));
            for l in &dp.leak {
                p += l.w * (v / l.v_nom) * (l.k_v * (v - l.v_nom)).exp() * l.k_t * (l.k_t * (t_j - 85.0)).exp();
            }
        }
        p
    }

    /// Phase power at the lowest junction-temperature equilibrium above the inlet (04 §9 Tier A estimate);
    /// `runaway` when none exists (the leakage-temperature loop gain reaches 1 before the junction balances).
    pub fn power(&self, e: &PhaseEnergy, hz: &[f64]) -> PowerBreakdown {
        let tw = e.makespan_s.max(1e-30);
        let dyn_core = e.core_dyn_j / tw;
        let indep = e.indep_j / tw;
        let dram = e.dram_j / tw + self.dram_background_w;
        let clock = self.p_clock(hz, e.activity);
        let ctrl = self.p_ctrl(hz, e.busy);
        let r = self.r_ja_k_mm2_w / self.die_mm2.max(1.0);
        let other = dyn_core + indep + clock + ctrl + dram;
        // h(T) = T_inlet + R P(T) - T is convex (leakage grows exponentially in T) with h(T_inlet) >= 0, so Newton
        // from the inlet climbs monotonically onto the lowest root; a nonnegative slope while h > 0 means none.
        let mut t_j = self.t_inlet_c;
        let mut found = false;
        for _ in 0..200 {
            let h = self.t_inlet_c + r * (other + self.p_static(hz, t_j)) - t_j;
            let slope = r * self.p_static_dt(hz, t_j) - 1.0;
            if !h.is_finite() || slope >= 0.0 || t_j >= T_RUNAWAY_C {
                break;
            }
            let step = -h / slope;
            t_j += step;
            if step <= 1e-6 {
                found = true;
                break;
            }
        }
        let st = self.p_static(hz, t_j);
        let chip = dyn_core + indep + st + clock + ctrl;
        let vr = (1.0 / self.eta_vr - 1.0) * (dyn_core + st + clock + ctrl);
        let pkg = chip + dram;
        let board = pkg + vr + self.board_w;
        PowerBreakdown {
            dyn_core_w: dyn_core,
            indep_w: indep,
            static_w: st,
            clock_w: clock,
            ctrl_w: ctrl,
            dram_w: dram,
            board_fixed_w: self.board_w,
            vr_loss_w: vr,
            chip_w: chip,
            package_w: pkg,
            board_w: board,
            t_j_c: t_j,
            runaway: !found || self.runaway(hz, t_j).is_some(),
        }
    }

    /// `E-PHYS-THERMAL-RUNAWAY` when the leakage-temperature loop gain reaches 1 (04 §8).
    pub fn runaway(&self, hz: &[f64], t_j: f64) -> Option<Diagnostic> {
        let dp_dt = (self.p_static(hz, t_j + 0.5) - self.p_static(hz, t_j - 0.5)).max(0.0);
        let gain = dp_dt * self.r_ja_k_mm2_w / self.die_mm2.max(1.0);
        (gain >= 1.0).then(|| {
            Diagnostic::error("E-PHYS-THERMAL-RUNAWAY", format!("leakage-temperature loop gain {gain:.2} >= 1 at {t_j:.0} C"))
                .hint("cut leakage (smaller die, fewer or smaller memories) or improve cooling")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_ir::hw::phys::VfPoint;
    use kiln_ir::hw::quantity::{Hz, Volts};

    /// 04 §8: leakage 40 W at 85 C doubling every ~35 K, 50 W other package power, 90 K mm^2/W over 100 mm^2 has no
    /// stable junction temperature; at 9 K mm^2/W it converges.
    #[test]
    fn thermal_runaway_is_detected() {
        let n = Tables::get().node("tsmc_n7").unwrap();
        let c = ClockDomain { id: kiln_ir::common::Id::new("c").unwrap(), freq: Hz(1e9), base: None, voltage: Some(Volts(n.vdd_nom)), vf: vec![VfPoint { freq: Hz(1e9), voltage: Volts(n.vdd_nom) }], crossing_latency: None };
        let leak = Leak { tech: n.id.clone(), v_nom: n.vdd_nom, k_v: 3.0, k_t: 0.02, w: 40.0 };
        let d = DomainPower { vf: VfTable::build(&c, n, 0.3, 1.3), v_nom: n.vdd_nom, v_noms: vec![(n.id.clone(), n.vdd_nom)], leak_w: 40.0, leak: vec![leak], c_clk_f: 0.0, c_pipe_f: 0.0, c_ctrl_f: 0.0 };
        let pm = |r: f64| PowerModel {
            domains: vec![d.clone()],
            indep_leak_w: 0.0,
            dram_background_w: 0.0,
            board_w: 0.0,
            eta_vr: 1.0,
            clk_ungated: 0.0,
            cap_w: None,
            assumed_cap_w: None,
            cap_level: CapLevel::Package,
            die_mm2: 100.0,
            t_inlet_c: 30.0,
            r_ja_k_mm2_w: r,
            tj_max_c: 105.0,
            q_avg_max: 1.0,
        };
        let e = PhaseEnergy { makespan_s: 1.0, core_dyn_j: 50.0, ..Default::default() };
        let hot = pm(90.0).power(&e, &[1e9]);
        assert!(hot.runaway, "{hot:?}");
        let cool = pm(9.0).power(&e, &[1e9]);
        assert!(!cool.runaway && (cool.t_j_c - (30.0 + cool.package_w * 0.09)).abs() < 0.01, "{cool:?}");
    }

    /// 04 §9: the junction settles at the lowest equilibrium above the inlet. `T = 30 + 55.1 exp(0.02 (T - 85))` has
    /// a stable root near 76.3 C (loop gain 0.93) although the iteration from 85 C would climb away from it.
    #[test]
    fn the_lowest_stable_junction_temperature_is_found() {
        let n = Tables::get().node("tsmc_n7").unwrap();
        let c = ClockDomain { id: kiln_ir::common::Id::new("c").unwrap(), freq: Hz(1e9), base: None, voltage: Some(Volts(n.vdd_nom)), vf: vec![VfPoint { freq: Hz(1e9), voltage: Volts(n.vdd_nom) }], crossing_latency: None };
        let leak = |w: f64| Leak { tech: n.id.clone(), v_nom: n.vdd_nom, k_v: 3.0, k_t: 0.02, w };
        let pm = |w: f64| PowerModel {
            domains: vec![DomainPower { vf: VfTable::build(&c, n, 0.3, 1.3), v_nom: n.vdd_nom, v_noms: vec![(n.id.clone(), n.vdd_nom)], leak_w: w, leak: vec![leak(w)], c_clk_f: 0.0, c_pipe_f: 0.0, c_ctrl_f: 0.0 }],
            indep_leak_w: 0.0,
            dram_background_w: 0.0,
            board_w: 0.0,
            eta_vr: 1.0,
            clk_ungated: 0.0,
            cap_w: None,
            assumed_cap_w: None,
            cap_level: CapLevel::Package,
            die_mm2: 100.0,
            t_inlet_c: 30.0,
            r_ja_k_mm2_w: 100.0,
            tj_max_c: 105.0,
            q_avg_max: 1.0,
        };
        let e = PhaseEnergy { makespan_s: 1.0, ..Default::default() };
        let b = pm(55.1).power(&e, &[1e9]);
        assert!(!b.runaway && (b.t_j_c - 76.305).abs() < 0.01, "{b:?}");
        assert!((b.t_j_c - (30.0 + b.package_w)).abs() < 1e-3, "{b:?}");
        // No equilibrium at all: 70 W at 85 C with the same loop.
        assert!(pm(70.0).power(&e, &[1e9]).runaway);
    }

    #[test]
    fn vf_tables_are_monotone() {
        let n = Tables::get().node("tsmc_n7").unwrap();
        let c = ClockDomain {
            id: kiln_ir::common::Id::new("c").unwrap(),
            freq: Hz(1.41e9),
            base: Some(Hz(1.095e9)),
            voltage: None,
            vf: vec![
                VfPoint { freq: Hz(1.095e9), voltage: Volts(0.73) },
                VfPoint { freq: Hz(1.29e9), voltage: Volts(0.81) },
                VfPoint { freq: Hz(1.41e9), voltage: Volts(0.87) },
            ],
            crossing_latency: None,
        };
        let t = VfTable::build(&c, n, 0.3, 1.3);
        assert!(t.points.windows(2).all(|w| w[1].0 > w[0].0 && w[1].1 >= w[0].1), "{:?}", t.points);
        assert!((t.voltage(1.29e9) - 0.81).abs() < 1e-9 && (t.voltage(1.35e9) - 0.84).abs() < 1e-9);
        assert_eq!(t.floor_hz, 1.095e9);
        let fixed = ClockDomain { vf: vec![], base: None, ..c };
        let t2 = VfTable::build(&fixed, n, 0.3, 1.3);
        assert!(t2.points.windows(2).all(|w| w[1].0 > w[0].0 && w[1].1 > w[0].1));
        assert!((t2.points.last().unwrap().0 - 1.41e9).abs() < 1.0 && (t2.voltage(1.41e9) - n.v_max).abs() < 1e-9);
    }
}
