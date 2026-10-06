//! The committed kiln-phys calibration sets are what `phys_calibrate` fits from this tree (04 §12.3): a change to a
//! design, table, target or the engine traffic the power group reads must come with a refit
//! (`WRITE=1 cargo run -p kiln-sim --release --example phys_calibrate`).

#[allow(dead_code)]
#[path = "../examples/phys_calibrate.rs"]
mod phys_calibrate;

#[test]
fn phys_calibration_reproduces() {
    let (g, a) = phys_calibrate::run();
    let t = kiln_phys::tables::Tables::get();
    for fresh in [&g, &a] {
        let committed = t.calib.iter().find(|c| c.id == fresh.id).unwrap_or_else(|| panic!("{} is not committed", fresh.id));
        assert_eq!(committed.targets_hash, fresh.targets_hash, "{}: fitted from other inputs", fresh.id);
        assert_eq!(committed.params.len(), fresh.params.len(), "{}", fresh.id);
        for (c, f) in committed.params.iter().zip(&fresh.params) {
            assert_eq!((&c.name, &c.key), (&f.name, &f.key), "{}", fresh.id);
            assert!((c.value / f.value - 1.0).abs() < 1e-9, "{} {}{:?}: committed {} vs refit {}", fresh.id, c.name, c.key, c.value, f.value);
            assert_eq!(c.at_bound, f.at_bound, "{} {}", fresh.id, c.name);
        }
        assert!(fresh.params.iter().all(|p| !p.at_bound), "{}: a parameter at its bound is a calibration failure (04 §12.3): {:?}", fresh.id, fresh.params.iter().filter(|p| p.at_bound).map(|p| &p.name).collect::<Vec<_>>());
    }
}
