//! Archives written by pyarrow (tests/golden/archive_pyarrow/gen.py) read through the 05 §3.9 reader.

use std::path::Path;

use kiln_trace::archive::{Archive, DesignStatus};

#[test]
fn reads_pyarrow_appended_archive() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/archive_pyarrow");
    let a = Archive::read_dir(&dir).unwrap();
    assert!(!a.partial);
    assert_eq!(a.meta.axes.len(), 2);
    assert_eq!(a.designs.len(), 3);
    let d1 = a.design("d1").unwrap();
    assert_eq!(d1.parent_ids, vec!["d0"]);
    assert_eq!(d1.cell, vec![1, 1]);
    assert_eq!(d1.fitness_components["prefill_b1"], 1.3 * 1.1);
    assert_eq!(a.designs[2].status, DesignStatus::Invalid);
    assert_eq!(a.designs[0].extrapolated, None);
    assert_eq!(a.elites().len(), 2);
    assert_eq!(a.generations.len(), 2);
    assert_eq!(a.generations[1].invalid_count, 1);
    assert_eq!(a.generations[1].best_low, None);
}
