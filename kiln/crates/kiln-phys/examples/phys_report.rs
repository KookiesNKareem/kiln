//! Prints the physical report of reference designs: `cargo run -p kiln-phys --release --example phys_report [names]`.

use kiln_ir::hw::{Profile, check_file};
use kiln_phys::{PlaceTier, Phys};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names = if args.is_empty() {
        ["a100_sxm4_40gb", "h100_sxm5_80gb", "v100_sxm2_32gb", "tpu_v4", "tpu_v5e", "tpu_v6e", "ember"].map(String::from).to_vec()
    } else {
        args
    };
    for n in names {
        let p = format!("{}/../../designs/reference/{n}.json5", env!("CARGO_MANIFEST_DIR"));
        let r = check_file(&p, Profile::Full);
        let Some(hw) = r.model else {
            println!("{n}: {:?}", r.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>());
            continue;
        };
        let t = std::time::Instant::now();
        let ph = Phys::new(&hw);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let tb = std::time::Instant::now();
        let pb = Phys::with(&hw, kiln_phys::Params::for_family(hw.family.as_deref()), PlaceTier::B);
        let msb = tb.elapsed().as_secs_f64() * 1e3;
        let rep = ph.report().unwrap();
        println!("== {n} ({} nodes, {} macros) build {ms:.1} ms (tier B {msb:.0} ms, phi A {:.4} B {:.4})", hw.nodes.len(), rep.macros, ph.m3().unwrap().fp.stats.phi, pb.m3().unwrap().fp.stats.phi);
        println!("   timing us: characterize {:.0} place {:.0} links {:.0}", rep.timing_us.0, rep.timing_us.1, rep.timing_us.2);
        for d in &rep.dies {
            println!(
                "   die {} [{}] env {:.1} mm2 [{:.1}, {:.1}] outline {:.1} legal {:.1} xtors {:.1} B sram {:.1} MiB shore-limited {} hbm shore {:.1}/{:.1} mm",
                d.path, d.node, d.area_mm2, d.area_low_mm2, d.area_high_mm2, d.outline_mm2, d.legalized_mm2, d.transistors_b, d.sram_mib, d.shoreline_limited, d.hbm_shoreline_used_mm, d.hbm_shoreline_available_mm
            );
            println!("     parts {:?} whitespace {:.1}", d.parts_mm2.iter().map(|(k, v)| format!("{k} {v:.1}")).collect::<Vec<_>>(), d.whitespace_mm2);
        }
        let m = ph.m3().unwrap();
        println!("   package {:.0} mm2 ({}) static {:.1} W (85C nominal) dram bg {:.1} W board {:.1} W", rep.package_mm2, rep.package_table, ph.static_power_w(), m.power.dram_background_w, m.power.board_w);
        for (i, c) in hw.clocks.iter().enumerate() {
            let d = &m.power.domains[i];
            println!("   clock {} vf {:?} leak {:.1} W c_clk {:.2} nF", c.path, d.vf.points.iter().step_by(8).map(|p| format!("{:.0}MHz@{:.3}V", p.0 / 1e6, p.1)).collect::<Vec<_>>(), d.leak_w, d.c_clk_f * 1e9);
        }
        if std::env::var("ROUTERS").is_ok() { routers(&hw, &ph); }
        if std::env::var("STACKS").is_ok() { stacks(&hw, &ph); }
        for p in &rep.problems {
            println!("   ! {} {}", p.code, p.message);
        }
    }
}

#[allow(dead_code)]
pub fn routers(hw: &kiln_ir::hw::HwModel, ph: &Phys) {
    let m = ph.m3().unwrap();
    for (i, r) in hw.routers.iter().enumerate() {
        let net = &hw.networks[r.net];
        println!("   router {} topo {} radix {:?} outdeg {} w {:?} area {:.2} mm2", hw.nodes[r.node].path, net.spec.topology.name(), net.spec.router.radix, hw.out_edges[r.node].len(), net.spec.link.width_bits, m.ch.routers[i].area_um2 / 1e6);
    }
}

#[allow(dead_code)]
pub fn stacks(hw: &kiln_ir::hw::HwModel, ph: &Phys) {
    let m = ph.m3().unwrap();
    for d in &m.fp.dies {
        for (i, mm) in d.macros.iter().enumerate() {
            let r = d.rects[i];
            println!("   macro {} n={} area {:.1} phy {:?} rect [{:.1},{:.1}]x[{:.1},{:.1}] mm", mm.name, mm.nodes.len(), mm.area_um2 / 1e6, mm.phy.map(|p| (p.0, p.1 / 1000.0)), r.x0 / 1e3, r.x1 / 1e3, r.y0 / 1e3, r.y1 / 1e3);
        }
    }
    for p in &m.fp.packages {
        for (mi, r) in &p.stacks {
            println!("   stack {} rect [{:.1},{:.1}]x[{:.1},{:.1}]", hw.nodes[hw.memories[*mi].node].path, r.x0 / 1e3, r.x1 / 1e3, r.y0 / 1e3, r.y1 / 1e3);
        }
    }
}
