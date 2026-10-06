//! Rendered-view snapshots over golden traces (05 §9.5): SVG text via insta (diffable), PNG determinism.
//! Update with `INSTA_UPDATE=always cargo test -p kiln-viz-render` (or `cargo insta review`).

use std::path::{Path, PathBuf};

use kiln_trace::archive::Archive;
use kiln_trace::calib_report::{CalibRow, from_calibrate_json};
use kiln_trace::container::read_kiln;
use kiln_trace::trace::Trace;
use kiln_viz_render::views::{FloorColor, ViewKind, ViewSpec, WireColor};
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

/// A design-only trace (no simulation): kiln-ir expansion and validation, kiln-phys placement.
fn structure(name: &str) -> Trace {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../designs/reference/{name}.json5"));
    let d = kiln_ir::hw::load_file(&p).unwrap();
    let (hw, _) = d.expand(&Default::default()).unwrap();
    let checks = kiln_trace::phys::profile_checks(&d, &hw);
    kiln_trace::build::structure(&hw, Some(d.canonical.clone()), Some(name.into()), &checks)
}

#[test]
fn placed_views_svg_snapshots() {
    let sel = Selection::default();
    let run = trace("tpu_v5e_decode_b1_ops_v1.1.kiln");
    let multi = trace("tpu_v5e_2x2_decode_b1_ops_v1.1.kiln");
    let v6e = structure("tpu_v6e");
    let ember = structure("ember");
    let one = |t: &Trace, sp: &ViewSpec| {
        to_svg(&render(
            &Inputs {
                runs: vec![t],
                ..Default::default()
            },
            sp,
            &sel,
        ))
    };
    let fp = spec(ViewKind::Floorplan);
    insta::assert_snapshot!("v5e_decode_placed_floorplan", one(&run, &fp));
    insta::assert_snapshot!(
        "v5e_decode_placed_floorplan_traffic",
        one(&run, &ViewSpec { color: FloorColor::Energy, wire_color: WireColor::Traffic, ..fp.clone() })
    );
    insta::assert_snapshot!(
        "v5e_2x2_placed_floorplan_kind",
        one(&multi, &ViewSpec { color: FloorColor::Kind, ..fp.clone() })
    );
    insta::assert_snapshot!("v5e_decode_design", one(&run, &spec(ViewKind::Design)));
    insta::assert_snapshot!("v5e_decode_wires", one(&run, &spec(ViewKind::Wires)));
    insta::assert_snapshot!("v6e_design_only_floorplan", one(&v6e, &fp));
    insta::assert_snapshot!(
        "v6e_design_only_floorplan_density",
        one(&v6e, &ViewSpec { color: FloorColor::Density, wire_color: WireColor::Length, ..fp.clone() })
    );
    insta::assert_snapshot!("v6e_design_only_design", one(&v6e, &spec(ViewKind::Design)));
    insta::assert_snapshot!("v6e_design_only_wires", one(&v6e, &spec(ViewKind::Wires)));
    insta::assert_snapshot!(
        "ember_design_only_floorplan_layers",
        one(&ember, &ViewSpec { labels: false, wires: false, ..fp.clone() })
    );
    insta::assert_snapshot!(
        "ember_design_only_floorplan_layer1",
        one(&ember, &ViewSpec { layer: Some(1), root: Some("board.pkg".into()), color: FloorColor::Area, ..fp.clone() })
    );
}

#[test]
fn placed_png_is_deterministic_and_hits_resolve() {
    for t in [trace("tpu_v5e_decode_b1_ops_v1.1.kiln"), structure("ember")] {
        let inp = Inputs {
            runs: vec![&t],
            ..Default::default()
        };
        let s = render(&inp, &spec(ViewKind::Floorplan), &Selection::default());
        let again = render(&inp, &spec(ViewKind::Floorplan), &Selection::default());
        assert!(to_png(&s, 1.0) == to_png(&again, 1.0) && to_svg(&s) == to_svg(&again));
        assert!(!to_svg(&s).contains("UNPLACED"));
        // Block hits resolve to placed rows, wire hits to links with a wires row.
        let (mut blocks, mut wires) = (0, 0);
        for p in &s.prims {
            match p {
                kiln_viz_render::scene::Prim::Rect {
                    hit: kiln_viz_render::Hit::Resource(r),
                    ..
                } => {
                    assert!(t.floorplan.iter().any(|f| f.resource == *r));
                    blocks += 1;
                }
                kiln_viz_render::scene::Prim::Line {
                    hit: kiln_viz_render::Hit::Resource(r),
                    ..
                } => {
                    assert!(t.wires.iter().any(|w| w.link == *r));
                    wires += 1;
                }
                _ => {}
            }
        }
        assert!(blocks > 10 && wires > 5, "{blocks} blocks, {wires} wires");
    }
}

fn texts(s: &kiln_viz_render::Scene) -> Vec<String> {
    s.prims
        .iter()
        .filter_map(|p| match p {
            kiln_viz_render::scene::Prim::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn one_run<'a>(t: &'a Trace) -> Inputs<'a> {
    Inputs {
        runs: vec![t],
        ..Default::default()
    }
}

#[test]
fn traffic_counts_each_link_once_however_many_channels_it_has() {
    let t = trace("tpu_v5e_decode_b1_ops_v1.1.kiln");
    let sp = ViewSpec {
        wire_color: WireColor::Traffic,
        ..spec(ViewKind::Floorplan)
    };
    let lines = |t: &Trace| -> Vec<String> {
        let s = render(&one_run(t), &sp, &Selection::default());
        let mut out: Vec<String> = s
            .prims
            .iter()
            .filter(|p| matches!(p, kiln_viz_render::scene::Prim::Line { .. }))
            .map(|p| format!("{p:?}"))
            .collect();
        out.extend(texts(&s).into_iter().filter(|x| !x.contains("wires drawn")));
        out
    };
    // Every link split into two channel rows of half the bandwidth: same links, same traffic.
    let mut split = t.clone();
    split.wires = t
        .wires
        .iter()
        .flat_map(|w| {
            let mut h = w.clone();
            h.bw_bps /= 2.0;
            [h.clone(), h]
        })
        .collect();
    assert_eq!(lines(&split), lines(&t));
}

#[test]
fn design_sheet_memory_section_is_per_chip() {
    let t = structure("tpu_v5e_2x2");
    let d = t.scalar("design_summary").unwrap();
    let total = d["onchip_capacity_b"].as_f64().unwrap();
    let per_chip = (0.25 + 1.125 + 128.0) * 1048576.0;
    assert_eq!(total, 4.0 * per_chip);
    let txt = texts(&render(&one_run(&t), &spec(ViewKind::Design), &Selection::default()));
    let at = txt.iter().position(|x| x == "on-chip per chip").expect("per-chip row");
    assert_eq!(txt[at + 1], kiln_viz_render::chart::fmt_bytes(per_chip));
    let at = txt.iter().position(|x| x == "on-chip, 4 chips").expect("all-chip row");
    assert_eq!(txt[at + 1], kiln_viz_render::chart::fmt_bytes(total));
}

#[test]
fn design_sheet_groups_only_identical_dies() {
    let mut t = structure("tpu_v6e");
    let i = t.run_scalars.iter().position(|(k, _)| k == "design_summary").unwrap();
    let mut d: serde_json::Value = serde_json::from_str(&t.run_scalars[i].1).unwrap();
    let die = |path: &str, parts: serde_json::Value, sram: f64| {
        serde_json::json!({"path": path, "node": "tsmc_n5", "layer": 0, "area_mm2": 100.0, "area_low_mm2": 95.0,
            "area_high_mm2": 105.0, "outline_mm2": 100.0, "parts_mm2": parts, "whitespace_mm2": 10.0,
            "transistors_b": 10.0, "sram_mib": sram})
    };
    d["area"]["dies"] = serde_json::json!([
        die("board.a", serde_json::json!({"datapath": 80.0, "sram": 10.0}), 10.0),
        die("board.b", serde_json::json!({"datapath": 10.0, "sram": 80.0}), 100.0),
        die("board.c", serde_json::json!({"datapath": 80.0, "sram": 10.0}), 10.0),
    ]);
    t.run_scalars[i].1 = d.to_string();
    let txt = texts(&render(&one_run(&t), &spec(ViewKind::Design), &Selection::default()));
    assert!(txt.iter().any(|x| x.starts_with("a x2 ")), "{txt:?}");
    assert!(txt.iter().any(|x| x.starts_with("b (")), "{txt:?}");
    assert!(txt.iter().any(|x| x == "10 B / 100 MiB"), "{txt:?}");
}

#[test]
fn harvested_parts_follow_the_selected_root() {
    let t = structure("a100_sxm4_40gb");
    let die = t
        .floorplan
        .iter()
        .map(|f| &t.resources[f.resource as usize])
        .find(|r| t.resource_kind(r) == "die")
        .unwrap()
        .path
        .clone();
    let hatched = |root: Option<String>| {
        let sp = ViewSpec { root, ..spec(ViewKind::Floorplan) };
        texts(&render(&one_run(&t), &sp, &Selection::default())).iter().filter(|x| *x == "harvested").count()
    };
    assert_eq!(hatched(None), 1);
    assert_eq!(hatched(Some(die)), 0);
}

#[test]
fn design_sheet_truncates_nothing_at_the_default_size() {
    for name in ["a100_sxm4_40gb", "h100_sxm5_80gb", "tpu_v6e", "tpu_v5e_2x2", "ember"] {
        let t = structure(name);
        let sp = ViewSpec { view: ViewKind::Design, ..ViewSpec::default() };
        let cut: Vec<String> =
            texts(&render(&one_run(&t), &sp, &Selection::default())).into_iter().filter(|x| x.contains('…')).collect();
        assert!(cut.is_empty(), "{name}: {cut:?}");
    }
}
