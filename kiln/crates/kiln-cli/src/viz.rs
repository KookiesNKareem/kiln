//! `kiln viz`, `kiln viz render`, `kiln trace export` and `.kiln` writing for `kiln eval -o run.kiln` (05 §8).

use std::path::{Path, PathBuf};

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::{Design, ExpandOptions, HwModel};
use kiln_py::inputs::{self, DesignInput};
use kiln_trace::archive::Archive;
use kiln_trace::build::{BuildInput, build};
use kiln_trace::calib_report::{self, CalibRow};
use kiln_trace::container::{self, write_kiln};
use kiln_trace::result::EvalResult;
use kiln_trace::sim::SimResult;
use kiln_trace::trace::Trace;
use kiln_trace::{RESULT_SCHEMA, SIM_SCHEMA, TraceLevel};
use kiln_viz_render::views::{FloorColor, ViewKind, ViewSpec};
use kiln_viz_render::{Inputs, Selection};
use serde_json::Value;

use crate::cli::{ExportArgs, Global, RenderArgs, VizArgs, VizData};
use crate::commands::{read, write_out};
use crate::{Failure, exit};

const CODE: &str = "E-VIZ";

fn usage(msg: impl Into<String>, hint: &str) -> Failure {
    Failure::new(exit::USAGE, Diagnostic::error(CODE, msg).hint(hint))
}

fn input_err(d: Diagnostic) -> Failure {
    Failure::new(exit::INPUT, d)
}

/// Expanded model and canonical JSON of a design argument (file or reference name).
pub fn load_hw(design: &str) -> Result<(HwModel, Design), Failure> {
    let d = inputs::load_design(
        &DesignInput::Str(design.to_string()),
        &inputs::default_designs_dir(),
    )
    .map_err(|diags| Failure {
        code: exit::INPUT,
        diags,
    })?;
    let (hw, _) = d
        .expand(&ExpandOptions::default())
        .map_err(|diags| Failure {
            code: exit::INPUT,
            diags,
        })?;
    Ok((hw, d))
}

pub fn design_name(arg: &str) -> String {
    Path::new(arg).file_stem().map_or_else(
        || arg.to_string(),
        |s| s.to_string_lossy().trim_end_matches(".json").to_string(),
    )
}

/// Trace of an evaluation; the design is re-expanded for the hierarchy, peaks and the floorplan.
pub fn trace_for_eval(
    r: &EvalResult,
    design: &str,
    workload: &str,
    level: TraceLevel,
) -> Result<Trace, Failure> {
    let (hw, d) = load_hw(design)?;
    Ok(build(&BuildInput {
        result: r,
        hw: Some(&hw),
        design_json: Some(d.canonical.clone()),
        design_name: Some(design_name(design)),
        workload_name: Some(workload.to_string()),
        level,
        floorplan: None,
    }))
}

pub fn write_trace(path: &Path, t: &Trace) -> Result<(), Failure> {
    std::fs::write(path, write_kiln(t)).map_err(|e| {
        Failure::new(
            exit::INPUT,
            Diagnostic::error("E-CLI-IO", format!("cannot write {}: {e}", path.display())),
        )
    })
}

/// A run from a `.kiln`, a `kiln.result/1` or `kiln.sim/1` JSON, or a design (structure only).
pub fn load_run(path: &Path) -> Result<Trace, Failure> {
    if !path.exists() {
        return structure_only(&path.display().to_string());
    }
    let bytes = read(path)?;
    if bytes.starts_with(&container::MAGIC) {
        return container::read_kiln(&bytes)
            .map_err(|d| input_err(d.at(path.display().to_string())));
    }
    if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
        match v.get("schema").and_then(Value::as_str) {
            Some(RESULT_SCHEMA) => {
                let r: EvalResult = serde_json::from_value(v).map_err(|e| {
                    input_err(Diagnostic::error(CODE, format!("{}: {e}", path.display())))
                })?;
                if r.sim.is_empty() {
                    return Err(input_err(Diagnostic::error(CODE, format!("{} carries no simulation results", path.display())).hint("re-run kiln eval with --trace ops, or write the trace directly with -o run.kiln")));
                }
                let mut t = build(&BuildInput {
                    result: &r,
                    hw: None,
                    design_json: None,
                    design_name: Some(design_name(&path.display().to_string())),
                    workload_name: None,
                    level: TraceLevel::Ops,
                    floorplan: None,
                });
                t.manifest.notes.push("built from a result JSON without the design: hierarchy recovered from resource paths, no roofline ceilings".into());
                return Ok(t);
            }
            Some(SIM_SCHEMA) => {
                let s: SimResult = serde_json::from_value(v).map_err(|e| {
                    input_err(Diagnostic::error(CODE, format!("{}: {e}", path.display())))
                })?;
                return Ok(kiln_trace::build::from_sim(&s, None, TraceLevel::Ops));
            }
            _ => {}
        }
    }
    structure_only(&path.display().to_string())
}

/// `kiln viz <design>`: expansion and the hierarchy floorplan only, no simulation (05 §8).
pub fn structure_only(design: &str) -> Result<Trace, Failure> {
    let (hw, d) = load_hw(design)?;
    let mut prov = kiln_trace::Provenance::unknown(kiln_trace::Tier::A);
    prov.design_hash = hw.design_hash.clone();
    let r = EvalResult {
        schema: RESULT_SCHEMA.into(),
        status: kiln_trace::result::Status::Ok,
        score: 0.0,
        score_interval: None,
        score_components: None,
        score_realistic: None,
        interval: Default::default(),
        stage_reached: kiln_trace::result::Stage::S0,
        tier: None,
        phases: vec![],
        ops: vec![],
        physical: None,
        features: Default::default(),
        violations: vec![],
        errors: vec![],
        warnings: vec![],
        audit: Default::default(),
        trace: None,
        provenance: prov,
        timing: Default::default(),
        calibration: None,
        invariants: None,
        sim: vec![],
    };
    let mut t = build(&BuildInput {
        result: &r,
        hw: Some(&hw),
        design_json: Some(d.canonical.clone()),
        design_name: Some(design_name(design)),
        workload_name: None,
        level: TraceLevel::Summary,
        floorplan: None,
    });
    t.manifest
        .notes
        .push("structure only: no simulation (run kiln eval -o run.kiln for metrics)".into());
    Ok(t)
}

pub struct Loaded {
    pub runs: Vec<Trace>,
    pub archive: Option<Archive>,
    pub calib: Option<Vec<CalibRow>>,
}

impl Loaded {
    pub fn inputs(&self) -> Inputs<'_> {
        Inputs {
            runs: self.runs.iter().collect(),
            archive: self.archive.as_ref(),
            calib: self.calib.as_deref(),
        }
    }
}

pub fn load(input: Option<&Path>, d: &VizData) -> Result<Loaded, Failure> {
    let mut runs = vec![];
    if let Some(p) = input {
        runs.push(load_run(p)?);
    }
    for p in &d.compare {
        runs.push(load_run(p)?);
    }
    let archive = d
        .archive
        .as_deref()
        .map(Archive::read_dir)
        .transpose()
        .map_err(input_err)?;
    let calib = d
        .calibration
        .as_deref()
        .map(calib_report::load)
        .transpose()
        .map_err(input_err)?;
    Ok(Loaded {
        runs,
        archive,
        calib,
    })
}

fn spec_from(a: &RenderArgs, view: ViewKind) -> Result<ViewSpec, Failure> {
    let mut spec: ViewSpec = match &a.state {
        Some(s) => serde_json::from_str(s).map_err(|e| {
            usage(
                format!("--state is not a view state: {e}"),
                "pass the JSON view spec copied from the viewer",
            )
        })?,
        None => ViewSpec::default(),
    };
    spec.view = view;
    let (w, h) = a
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<f32>().ok()?, h.parse::<f32>().ok()?)))
        .ok_or_else(|| {
            usage(
                format!("--size {:?} is not WxH", a.size),
                "e.g. --size 1600x1000",
            )
        })?;
    if !(64.0..=16384.0).contains(&w) || !(64.0..=16384.0).contains(&h) {
        return Err(usage(
            "--size out of range 64..16384",
            "e.g. --size 1024x640",
        ));
    }
    spec.width = w;
    spec.height = h;
    spec.theme = if a.theme == "dark" {
        kiln_viz_render::theme::ThemeKind::Dark
    } else {
        kiln_viz_render::theme::ThemeKind::Light
    };
    if a.phase.is_some() {
        spec.phase.clone_from(&a.phase);
    }
    if let Some(c) = &a.color {
        spec.color = FloorColor::parse(c).ok_or_else(|| {
            usage(
                format!("unknown --color {c:?}"),
                "utilization, idle, energy, bytes, kind",
            )
        })?;
    }
    if a.root.is_some() {
        spec.root.clone_from(&a.root);
    }
    spec.aggregate |= a.aggregate;
    if a.device.is_some() {
        spec.device.clone_from(&a.device);
    }
    if let Some(ax) = &a.axes {
        let (x, y) = ax
            .split_once(',')
            .and_then(|(x, y)| Some((x.trim().parse().ok()?, y.trim().parse().ok()?)))
            .ok_or_else(|| usage("--axes takes X,Y descriptor indices", "e.g. --axes 0,1"))?;
        spec.x_axis = x;
        spec.y_axis = y;
    }
    Ok(spec)
}

/// `kiln viz render`: one file per `--view` (05 §7.2).
pub fn render_cmd(g: &Global, a: &RenderArgs) -> Result<u8, Failure> {
    let loaded = load(a.input.as_deref(), &a.data)?;
    let out = g.out.clone().ok_or_else(|| {
        usage(
            "kiln viz render needs --out <fig.png|fig.svg>",
            "e.g. kiln viz render run.kiln --view floorplan -o fig.png",
        )
    })?;
    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_ascii_lowercase();
    if ext != "png" && ext != "svg" {
        return Err(usage(
            format!("unsupported output {}", out.display()),
            "use .png or .svg",
        ));
    }
    let views: Vec<ViewKind> = a.views.iter().map(|v| ViewKind::parse(v).ok_or_else(|| usage(format!("unknown view {v:?}"), "floorplan, noc, timeline, roofline, bottleneck, compare, evolution, calibration"))).collect::<Result<_, _>>()?;
    let inputs = loaded.inputs();
    for v in &views {
        let spec = spec_from(a, *v)?;
        let scene = kiln_viz_render::render(&inputs, &spec, &Selection::default());
        let path = if views.len() == 1 {
            out.clone()
        } else {
            sibling(&out, v.name())
        };
        if ext == "svg" {
            write_out(&path, &kiln_viz_render::to_svg(&scene))?;
        } else {
            std::fs::write(&path, kiln_viz_render::to_png(&scene, a.scale)).map_err(|e| {
                Failure::new(
                    exit::INPUT,
                    Diagnostic::error("E-CLI-IO", format!("cannot write {}: {e}", path.display())),
                )
            })?;
        }
        if g.format == crate::cli::Format::Text {
            println!("{}", path.display());
        }
    }
    Ok(exit::OK)
}

fn sibling(out: &Path, view: &str) -> PathBuf {
    let stem = out
        .file_stem()
        .map_or_else(|| "fig".into(), |s| s.to_string_lossy().into_owned());
    let ext = out
        .extension()
        .map_or_else(|| "png".into(), |s| s.to_string_lossy().into_owned());
    out.with_file_name(format!("{stem}-{view}.{ext}"))
}

const CHROME_JSON_LIMIT: usize = 2_000_000;

/// `kiln trace export`.
pub fn export_cmd(g: &Global, a: &ExportArgs) -> Result<u8, Failure> {
    let t = load_run(&a.file)?;
    let default_out = |ext: &str| a.file.with_extension(ext);
    if a.perfetto {
        let out = g.out.clone().unwrap_or_else(|| default_out("pftrace"));
        std::fs::write(&out, kiln_trace::perfetto::export_perfetto(&t)).map_err(|e| {
            Failure::new(
                exit::INPUT,
                Diagnostic::error("E-CLI-IO", format!("cannot write {}: {e}", out.display())),
            )
        })?;
        println!("{}", out.display());
    } else if a.chrome_json {
        if t.spans.len() > CHROME_JSON_LIMIT && !a.force {
            return Err(usage(
                format!(
                    "{} spans exceed the Chrome JSON limit of {CHROME_JSON_LIMIT}",
                    t.spans.len()
                ),
                "use --perfetto, or pass --force",
            ));
        }
        let out = g.out.clone().unwrap_or_else(|| default_out("json"));
        write_out(
            &out,
            &kiln_trace::perfetto::export_chrome_json(&t).to_string(),
        )?;
        println!("{}", out.display());
    } else if let Some(table) = &a.csv {
        let csv = csv_table(&t, table)?;
        match &g.out {
            Some(p) => write_out(p, &csv)?,
            None => print!("{csv}"),
        }
    } else {
        return Err(usage(
            "kiln trace export needs --perfetto, --chrome-json or --csv <table>",
            "--parquet is not implemented yet",
        ));
    }
    Ok(exit::OK)
}

fn csv_table(t: &Trace, name: &str) -> Result<String, Failure> {
    let (_, batch) = t
        .batches()
        .into_iter()
        .find(|(n, _)| *n == name)
        .ok_or_else(|| {
            usage(
                format!("unknown table {name:?}"),
                "resources, ops, phases, limiters, aggregates_resource, floorplan, spans, ...",
            )
        })?;
    let schema = batch.schema();
    let mut out = schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect::<Vec<_>>()
        .join(",")
        + "\n";
    for row in 0..batch.num_rows() {
        let cells: Vec<String> = batch
            .columns()
            .iter()
            .map(|c| {
                let v = kiln_trace::arrowx::display(c.as_ref(), row);
                if v.contains(',') || v.contains('"') {
                    format!("\"{}\"", v.replace('"', "\"\""))
                } else {
                    v
                }
            })
            .collect();
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Ok(out)
}

/// `kiln viz`: the native app (feature `gui`), or Perfetto.
pub fn viz_cmd(g: &Global, a: &VizArgs) -> Result<u8, Failure> {
    if a.perfetto {
        let input = a.input.as_deref().ok_or_else(|| {
            usage(
                "kiln viz --perfetto needs a run",
                "kiln viz run.kiln --perfetto",
            )
        })?;
        return open_perfetto(&load_run(input)?);
    }
    let loaded = load(a.input.as_deref(), &a.data)?;
    if loaded.runs.is_empty() && loaded.archive.is_none() && loaded.calib.is_none() {
        return Err(usage(
            "nothing to open",
            "kiln viz run.kiln | --compare a.kiln b.kiln | --archive <dir> | --calibration <report>",
        ));
    }
    let mut spec: ViewSpec = match &a.state {
        Some(s) => serde_json::from_str(s).map_err(|e| {
            usage(
                format!("--state is not a view state: {e}"),
                "pass the JSON view spec copied from the viewer",
            )
        })?,
        None => ViewSpec::default(),
    };
    spec.view = match a.view.as_deref() {
        Some(v) => ViewKind::parse(v).ok_or_else(|| {
            usage(
                format!("unknown view {v:?}"),
                "floorplan, noc, timeline, roofline, bottleneck, compare, evolution, calibration",
            )
        })?,
        None if loaded.runs.len() > 1 => ViewKind::Compare,
        None if loaded.runs.is_empty() && loaded.archive.is_some() => ViewKind::Archive,
        None if loaded.runs.is_empty() => ViewKind::Calibration,
        None => spec.view,
    };
    let _ = g;
    gui(loaded, spec, a.data.archive.clone().filter(|_| a.watch))
}

#[cfg(feature = "gui")]
fn gui(loaded: Loaded, spec: ViewSpec, watch: Option<PathBuf>) -> Result<u8, Failure> {
    if cfg!(target_os = "linux")
        && std::env::var_os("DISPLAY").is_none()
        && std::env::var_os("WAYLAND_DISPLAY").is_none()
    {
        return Err(Failure::new(
            exit::USAGE,
            Diagnostic::error(CODE, "no display")
                .hint("use `kiln viz render` for images (or forward a display)"),
        ));
    }
    kiln_viz::run(
        kiln_viz::Data {
            runs: loaded.runs,
            archive: loaded.archive,
            calib: loaded.calib,
            watch,
        },
        spec,
    )
    .map_err(|e| {
        Failure::new(
            exit::INTERNAL,
            Diagnostic::error(CODE, format!("viewer failed: {e}")),
        )
    })?;
    Ok(exit::OK)
}

#[cfg(not(feature = "gui"))]
fn gui(_: Loaded, _: ViewSpec, _: Option<PathBuf>) -> Result<u8, Failure> {
    Err(Failure::new(
        exit::NOT_IMPLEMENTED,
        Diagnostic::error(CODE, "this kiln build has no native viewer")
            .hint("build with `cargo build -p kiln-cli --features gui`, or use `kiln viz render` / `kiln viz --perfetto`"),
    ))
}

/// Serves the exported trace on 127.0.0.1:9001 with CORS and opens ui.perfetto.dev on it (05 §3.11), as
/// Perfetto's `open_trace_in_ui` does; the trace stays local.
fn open_perfetto(t: &Trace) -> Result<u8, Failure> {
    use std::io::{Read as _, Write as _};
    let bytes = kiln_trace::perfetto::export_perfetto(t);
    let name = "kiln.pftrace";
    let listener = std::net::TcpListener::bind("127.0.0.1:9001").map_err(|e| Failure::new(exit::INPUT, Diagnostic::error(CODE, format!("cannot listen on 127.0.0.1:9001: {e}")).hint("free the port, or use `kiln trace export --perfetto` and open the file in ui.perfetto.dev")))?;
    let url = format!("https://ui.perfetto.dev/#!/?url=http://127.0.0.1:9001/{name}");
    println!("serving {name} on http://127.0.0.1:9001/ ; opening {url}");
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(opener).arg(&url).status();
    for stream in listener.incoming().take(8) {
        let Ok(mut s) = stream else { continue };
        let mut req = [0u8; 2048];
        let n = s.read(&mut req).unwrap_or(0);
        let head = String::from_utf8_lossy(&req[..n]);
        let cors = "Access-Control-Allow-Origin: https://ui.perfetto.dev\r\nAccess-Control-Allow-Methods: GET, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n";
        if head.starts_with("OPTIONS") {
            let _ = write!(
                s,
                "HTTP/1.1 204 No Content\r\n{cors}Content-Length: 0\r\n\r\n"
            );
            continue;
        }
        if head.starts_with(&format!("GET /{name}")) {
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\n{cors}Content-Type: application/octet-stream\r\nContent-Length: {}\r\n\r\n",
                bytes.len()
            );
            let _ = s.write_all(&bytes);
            println!("trace served; press Ctrl-C to exit");
            return Ok(exit::OK);
        }
        let _ = write!(
            s,
            "HTTP/1.1 404 Not Found\r\n{cors}Content-Length: 0\r\n\r\n"
        );
    }
    Ok(exit::OK)
}
