//! Golden `.kiln` traces (05 §9.2): every reader version opens them and they pass `kiln trace validate`.
//! Regenerate with `kiln eval designs/reference/<d>.json5 --workload llama3_8b:<phase> --trace <level> -o <file>`.

use std::path::Path;

use kiln_trace::check::check_trace;
use kiln_trace::container::{TRACE_SCHEMA_VERSION, read_kiln, verify_members, write_kiln};
use kiln_trace::perfetto::export_perfetto;

#[test]
fn goldens_open_validate_and_round_trip() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/trace");
    let (mut n, mut current) = (0, 0);
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_none_or(|x| x != "kiln") {
            continue;
        }
        n += 1;
        let bytes = std::fs::read(&p).unwrap();
        let t = read_kiln(&bytes).unwrap_or_else(|d| panic!("{}: {d:?}", p.display()));
        assert!(
            verify_members(&bytes, &t.manifest).is_empty(),
            "{}",
            p.display()
        );
        let diags = check_trace(&t);
        assert!(diags.is_empty(), "{}: {diags:?}", p.display());
        assert!(!t.resources.is_empty() && !t.phases.is_empty() && !t.floorplan.is_empty());
        // Older minor versions are read through the migration path; only current ones rewrite identically.
        if t.manifest.schema_version == TRACE_SCHEMA_VERSION {
            assert_eq!(
                write_kiln(&t),
                bytes,
                "{}: rewrite is byte-identical",
                p.display()
            );
            assert!(
                !t.wires.is_empty() && t.floorplan.iter().any(|f| f.source != 0),
                "{}: placed floorplan and wires",
                p.display()
            );
            current += 1;
        }
        assert!(!export_perfetto(&t).is_empty());
    }
    assert!(n >= 6 && current >= 3);
}
