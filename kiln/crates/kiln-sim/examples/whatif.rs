//! Whole-step predictions for an arbitrary design file: `cargo run --release -p kiln-sim --example whatif -- <design>`.
use kiln_ir::hw::Profile;
use kiln_sim::{Prepared, SimOptions};
fn main() {
    let path = std::env::args().nth(1).expect("design path");
    let p = Prepared::from_file(std::path::Path::new(&path), Profile::Full).unwrap_or_else(|e| panic!("{e:?}"));
    for m in kiln_wl::zoo::suite("standard").unwrap() {
        let (r, _) = kiln_sim::simulate_member(&p, &m, &SimOptions::default()).unwrap();
        let dom = r.central.bottleneck.dominant().map(|(c, x)| format!("{c:?} {:.0}%", 100.0 * x / r.central.makespan_s)).unwrap_or_default();
        println!("{:<11} {:>9.3} ms [{:.3}, {:.3}]  {dom}", m.scenario.to_string(), r.time.central * 1e3, r.time.low * 1e3, r.time.high * 1e3);
    }
}
