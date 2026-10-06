//! Per-entity own area (children excluded) of the first compute die, by part, summed over instances:
//! `cargo run -p kiln-phys --release --example area_breakdown [names]` (PRIORS=1 for registry priors).

use std::collections::BTreeMap;

use kiln_ir::hw::{Profile, check_file};
use kiln_phys::characterize::{PARTS, characterize};
use kiln_phys::{Params, calib};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names = if args.is_empty() { ["a100_sxm4_40gb", "h100_sxm5_80gb", "v100_sxm2_32gb"].map(String::from).to_vec() } else { args };
    for n in names {
        let p = format!("{}/../../designs/reference/{n}.json5", env!("CARGO_MANIFEST_DIR"));
        let hw = check_file(&p, Profile::Full).model.unwrap();
        let params = if std::env::var("PRIORS").is_ok() { Params::priors() } else { Params::for_family(hw.family.as_deref()) };
        let ch = characterize(&hw, &params);
        let a = calib::area_of(&hw, &params);
        let mut by: BTreeMap<String, (usize, [f64; 7], f64)> = BTreeMap::new();
        for (i, np) in ch.nodes.iter().enumerate() {
            if np.area_um2 == 0.0 {
                continue;
            }
            let e = by.entry(hw.nodes[i].entity_id.clone()).or_default();
            e.0 += 1;
            for k in 0..7 {
                e.1[k] += np.parts[k] / 1e6;
            }
            e.2 += np.area_um2 / 1e6;
        }
        let tot: f64 = by.values().map(|x| x.2).sum();
        println!("== {n}: env {:.1} mm2, blocks {tot:.1} mm2, xtors {:.1} B", a.die_mm2, a.transistors_b);
        for (k, (c, parts, area)) in &by {
            let ps: Vec<String> = PARTS.iter().enumerate().filter(|(j, _)| parts[*j] > 0.0).map(|(j, p)| format!("{p:?} {:.2}", parts[j])).collect();
            println!("  {k:<14} x{c:<5} {area:>8.2} mm2 (per inst {:.4})  {}", area / *c as f64, ps.join(", "));
        }
    }
}
