//! Rendered-view snapshots over golden traces (05 §9.5): SVG text via insta (diffable), PNG determinism.
//! Update with `INSTA_UPDATE=always cargo test -p kiln-viz-render` (or `cargo insta review`).

use std::path::{Path, PathBuf};

use kiln_trace::archive::Archive;
use kiln_trace::calib_report::{CalibRow, from_calibrate_json};
use kiln_trace::container::read_kiln;
use kiln_trace::trace::Trace;
use kiln_viz_render::views::{FloorColor, ViewKind, ViewSpec};
use kiln_viz_render::{Inputs, Selection, render, to_png, to_svg};

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../kiln-trace/tests/golden")
        .join(name)
}

fn trace(name: &str) -> Trace {
    read_kiln(&std::fs::read(golden(&format!("trace/{name}"))).unwrap()).unwrap()
}

fn calib() -> Vec<CalibRow> {
    let v = serde_json::json!([{"set_hash": "cal1-fixture", "devices": [
        {"device": "a100_40gb", "role": "fit", "ops": [
            {"name": "prefill_b1/gemm_2048_6144_4096", "phase": "prefill_b1", "class": "gemm_compute", "count": 32,
             "meas_s": 4.207e-4, "cal_s": 4.195e-4, "uncal_s": 3.746e-4, "bytes": 9.23e7, "flops": 1.03e11},
            {"name": "decode_b1/gemv_4096_6144", "phase": "decode_b1", "class": "gemv_memory", "count": 32,
             "meas_s": 3.3e-5, "cal_s": 3.1e-5, "bytes": 5.03e7, "flops": 5.03e7}]},
        {"device": "tpu_v6e", "role": "held_out", "ops": [
            {"name": "decode_b1/gemv_4096_6144", "class": "gemv_memory", "meas_s": 3.0e-5, "cal_s": 3.9e-5}]}]}]);
    from_calibrate_json(&v)
}

fn spec(view: ViewKind) -> ViewSpec {
    ViewSpec {
        view,
        width: 900.0,
        height: 560.0,
        ..ViewSpec::default()
    }
}

#[test]
fn views_svg_snapshots() {
    let t = trace("tpu_v5e_decode_b1_ops_v1.0.kiln");
    let multi = trace("tpu_v5e_2x2_decode_b1_ops_v1.0.kiln");
    let arch = Archive::read_dir(&golden("archive_pyarrow")).unwrap();
    let cal = calib();
    let one = Inputs {
        runs: vec![&t],
        archive: None,
        calib: Some(&cal),
    };
    let sel = Selection::default();
    for v in [
        ViewKind::Floorplan,
        ViewKind::Roofline,
        ViewKind::Bottleneck,
        ViewKind::Timeline,
        ViewKind::Noc,
    ] {
        insta::assert_snapshot!(
            format!("v5e_decode_{}", v.name()),
            to_svg(&render(&one, &spec(v), &sel))
        );
    }
    let kind = ViewSpec {
        color: FloorColor::Kind,
        ..spec(ViewKind::Floorplan)
    };
    insta::assert_snapshot!(
        "v5e_2x2_floorplan_kind",
        to_svg(&render(
            &Inputs {
                runs: vec![&multi],
                ..Default::default()
            },
            &kind,
            &sel
        ))
    );
    let cmp = Inputs {
        runs: vec![&t, &multi],
        ..Default::default()
    };
    insta::assert_snapshot!(
        "compare_v5e_vs_2x2",
        to_svg(&render(&cmp, &spec(ViewKind::Compare), &sel))
    );
    let a = Inputs {
        archive: Some(&arch),
        ..Default::default()
    };
    insta::assert_snapshot!(
        "archive_fixture",
        to_svg(&render(&a, &spec(ViewKind::Archive), &sel))
    );
    insta::assert_snapshot!(
        "calibration_fixture",
        to_svg(&render(&one, &spec(ViewKind::Calibration), &sel))
    );
}

#[test]
fn png_is_deterministic_and_hits_resolve() {
    let t = trace("tpu_v5e_decode_b1_ops_v1.0.kiln");
    let inp = Inputs {
        runs: vec![&t],
        ..Default::default()
    };
    let s = render(&inp, &spec(ViewKind::Floorplan), &Selection::default());
    let (a, b) = (
        to_png(&s, 1.0),
        to_png(
            &render(&inp, &spec(ViewKind::Floorplan), &Selection::default()),
            1.0,
        ),
    );
    assert!(a.starts_with(b"\x89PNG") && a == b);
    // Every resource hit id in the floorplan resolves to a resource with a floorplan rect.
    let mut hits = 0;
    for p in &s.prims {
        if let kiln_viz_render::scene::Prim::Rect {
            hit: kiln_viz_render::Hit::Resource(r),
            ..
        } = p
        {
            assert!(t.floorplan.iter().any(|f| f.resource == *r));
            hits += 1;
        }
    }
    assert!(hits > 10);
}
