//! 03 §8 invariants for Tier A results. Failures are errors (`E-FLOOR-Ixx`), never warnings.

use kiln_map::hwview::{HwView, ResClass};
use kiln_map::lower::TaskGraph;
use kiln_map::program::Program;
use kiln_phys::ClockPlan;
use kiln_trace::SUM_REL_TOL;
use kiln_trace::sim::{CheckStatus, InvariantCheck, InvariantId, InvariantReport, Scope, SimResult};

use crate::engine::RunOut;

const TOL: f64 = 1e-9;

fn check(id: InvariantId, ok: bool, value: f64, limit: f64, unit: &str, msg: String, path: Option<String>) -> InvariantCheck {
    InvariantCheck {
        id,
        status: if ok { CheckStatus::Pass } else { CheckStatus::Fail },
        margin: Some(value - limit),
        value: Some(value),
        limit: Some(limit),
        unit: Some(unit.into()),
        path,
        message: if ok { String::new() } else { msg },
    }
}

fn skipped(id: InvariantId, why: &str) -> InvariantCheck {
    InvariantCheck { id, status: CheckStatus::Skipped, margin: None, value: None, limit: None, unit: None, path: None, message: why.into() }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= SUM_REL_TOL * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

pub struct Inputs<'a> {
    pub view: &'a HwView,
    pub prog: &'a Program,
    pub graph: &'a TaskGraph,
    pub run: &'a RunOut,
    pub clocks: &'a ClockPlan,
    pub resident_overflow: bool,
}

pub fn check_all(r: &SimResult, x: &Inputs) -> InvariantReport {
    let mut c = vec![];
    let (g, run) = (x.graph, x.run);
    let seg_of_group: Vec<usize> = {
        let mut m = vec![0usize; g.groups.len()];
        for (si, s) in run.segs.iter().enumerate() {
            m[s.groups.0..s.groups.1].iter_mut().for_each(|v| *v = si);
        }
        m
    };
    let worst = g
        .ops
        .iter()
        .filter(|o| o.peak_macs_per_s > 0.0 && o.useful_macs > 0)
        .map(|o| {
            let floor = o.useful_macs as f64 / o.peak_macs_per_s;
            let s = &run.segs[seg_of_group[o.group as usize]];
            (s.time_est - floor * (1.0 - TOL), floor, s.time_est, &x.prog.ops[o.op].id)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    c.push(match worst {
        Some((m, floor, t, op)) => check(InvariantId::I1, m >= 0.0, t, floor, "s", format!("group time {t} s below compute floor {floor} s"), Some(op.clone())),
        None => skipped(InvariantId::I1, "no MAC work"),
    });
    let dram_bytes: f64 = (0..x.view.resources.len())
        .filter(|&i| x.view.resources[i].class == ResClass::Dram)
        .map(|i| run.bytes.iter().map(|b| b[i]).sum::<f64>())
        .sum();
    let compulsory: f64 = g.ops.iter().map(|o| o.compulsory_offchip as f64).sum::<f64>();
    c.push(check(
        InvariantId::I2,
        dram_bytes >= compulsory * (1.0 - TOL),
        dram_bytes,
        compulsory,
        "B",
        format!("off-chip traffic {dram_bytes} B below compulsory {compulsory} B"),
        None,
    ));
    let over = r.resources.iter().max_by(|a, b| a.utilization.total_cmp(&b.utilization));
    c.push(match over {
        Some(o) => check(InvariantId::I3, o.utilization <= 1.0 + TOL, 1.0, o.utilization, "1", format!("utilization {} > 1", o.utilization), Some(o.resource.as_str().into())),
        None => skipped(InvariantId::I3, "no busy resource"),
    });
    let mut bad: Option<String> = None;
    for p in &g.profiles {
        for (gix, name) in [(p.src, "source"), (p.dst, "destination")] {
            let mems: Vec<u32> = x.view.groups[gix].mems.iter().map(|&m| x.view.res_of_mem[m]).collect();
            let s: f64 = p.entries.iter().filter(|(r, _)| mems.contains(r)).map(|e| e.1).sum();
            if (s - 1.0).abs() > 1e-9 && p.src != p.dst {
                bad.get_or_insert(format!("profile {}->{} {name} shares sum to {s}", x.view.groups[p.src].name, x.view.groups[p.dst].name));
            }
        }
        if p.entries.iter().any(|e| e.1 > 1.0 + 1e-9 || e.1 < 0.0) {
            bad.get_or_insert("link share outside [0, 1]".into());
        }
    }
    c.push(check(InvariantId::I4, bad.is_none(), 0.0, 0.0, "1", bad.clone().unwrap_or_default(), None));
    let mut i5: Option<(String, u128, u128)> = None;
    for o in &g.ops {
        let want = x.prog.ops[o.op].useful_macs();
        if o.useful_macs != want || o.issued_macs < o.useful_macs {
            i5.get_or_insert((x.prog.ops[o.op].id.clone(), o.useful_macs, want));
        }
    }
    c.push(match i5 {
        None => check(InvariantId::I5, true, 0.0, 0.0, "MAC", String::new(), None),
        Some((op, got, want)) => check(InvariantId::I5, false, got as f64, want as f64, "MAC", format!("slices cover {got} MACs, op has {want}"), Some(op)),
    });
    let e = &r.energy;
    let pw_ok = r.makespan_s <= 0.0 || close(r.power.avg_w, e.total_j / r.makespan_s);
    c.push(check(InvariantId::I6, close(e.total_j, e.component_sum()) && pw_ok, e.total_j, e.component_sum(), "J", "energy total != sum of components or avg power != E/T".into(), None));
    if x.resident_overflow && r.scope == Scope::Step {
        c.push(check(InvariantId::I7, false, 0.0, 0.0, "B", "resident model state exceeds its home capacity at scope step".into(), None));
    } else if x.resident_overflow {
        c.push(skipped(InvariantId::I7, "whole step does not fit; scope layer evaluated (02 §12.5)"));
    } else {
        c.push(check(InvariantId::I7, true, 0.0, 0.0, "B", String::new(), None));
    }
    let attributed = r.bottleneck.attributed_s();
    c.push(check(InvariantId::I8, close(attributed, r.makespan_s), attributed, r.makespan_s, "s", format!("attribution {attributed} s != makespan {} s", r.makespan_s), None));
    c.push(skipped(InvariantId::I9, "Tier B not run (M4)"));
    let hw = &x.view.hw;
    let clk_bad = hw.clocks.iter().enumerate().find(|(i, ci)| x.clocks.hz[*i] > ci.spec.freq.0 * (1.0 + TOL) || x.clocks.hz[*i] <= 0.0);
    c.push(match clk_bad {
        Some((i, ci)) => check(InvariantId::I10, false, x.clocks.hz[i], ci.spec.freq.0, "Hz", "clock above f_max".into(), Some(ci.path.clone())),
        None => check(InvariantId::I10, true, 0.0, 0.0, "Hz", String::new(), None),
    });
    c.push(skipped(InvariantId::I11, "checked by the determinism property tests (1 vs N threads, repeated runs)"));
    let prec_bad = x.prog.tensors.iter().find(|t| {
        let n: u128 = t.shape.iter().map(|&d| u128::from(d)).product();
        matches!(t.dtype.scaling, kiln_ir::wl::Scaling::None) && t.footprint() != (n * u128::from(t.dtype.elem_bits())).div_ceil(8)
    });
    c.push(check(InvariantId::I12, prec_bad.is_none(), 0.0, 0.0, "B", "tensor bytes != ceil(elements * bits / 8)".into(), prec_bad.map(|t| t.id.clone())));
    c.push(skipped(InvariantId::I13, "Tier B only"));
    c.push(skipped(InvariantId::I14, "single chip: no collectives"));
    c.push(check(
        InvariantId::I15,
        r.makespan_s >= r.t_a0_s * (1.0 - TOL),
        r.makespan_s,
        r.t_a0_s,
        "s",
        format!("makespan {} s below roofline A0 {} s", r.makespan_s, r.t_a0_s),
        None,
    ));
    InvariantReport { checks: c }
}
