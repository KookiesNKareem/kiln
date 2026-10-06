//! eframe shell (05 §4.1): view tabs, phase/color controls, the canvas (a `kiln-viz-render` scene painted with
//! egui shapes, hit-tested for hover and click), inspector, ops table, status bar and keyboard shortcuts.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2,
};
use kiln_trace::archive::Archive;
use kiln_trace::trace::{NONE_U32, Trace};
use kiln_viz_render::scene::{self, HAlign, Hit, Prim, Scene, VAlign};
use kiln_viz_render::theme::ThemeKind;
use kiln_viz_render::views::{FloorColor, ViewKind, ViewSpec, WireColor};
use kiln_viz_render::{Inputs, Selection};

use crate::Data;
use crate::state::{Action, AppState};

#[cfg(not(target_arch = "wasm32"))]
pub fn run(data: Data, spec: ViewSpec) -> Result<(), String> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 950.0])
            .with_title("kiln viz"),
        ..Default::default()
    };
    eframe::run_native(
        "kiln viz",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(data, spec)))),
    )
    .map_err(|e| e.to_string())
}

struct Cached {
    spec: ViewSpec,
    sel: Selection,
    size: (u32, u32),
    zoom: f32,
    scene: Scene,
}

struct App {
    data: Data,
    state: AppState,
    cache: Option<Cached>,
    /// Scene magnification (re-rendered, so the floorplan drills in as blocks grow) and pan, in points.
    zoom: f32,
    pan: Vec2,
    status: String,
    watch_rx: Option<Receiver<()>>,
    #[cfg(not(target_arch = "wasm32"))]
    _watcher: Option<notify::RecommendedWatcher>,
    show_ops: bool,
    /// `KILN_VIZ_SCREENSHOT=<path.png>`: save the window after a few frames and exit (docs, smoke tests).
    screenshot: Option<PathBuf>,
    frames: u32,
}

impl App {
    pub(crate) fn new(data: Data, mut spec: ViewSpec) -> Self {
        // A design without a run has no utilization: start on block kinds.
        if spec.color.needs_run() && data.runs.first().is_some_and(|t| t.aggregates_resource.is_empty()) {
            spec.color = FloorColor::Kind;
        }
        #[cfg(target_arch = "wasm32")]
        let watch_rx = None;
        #[cfg(not(target_arch = "wasm32"))]
        let (watch_rx, watcher) = match &data.watch {
            Some(dir) => match watch(dir) {
                Ok((rx, w)) => (Some(rx), Some(w)),
                Err(e) => {
                    eprintln!("warning: cannot watch {}: {e}", dir.display());
                    (None, None)
                }
            },
            None => (None, None),
        };
        App {
            data,
            state: AppState::new(spec),
            cache: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            status: String::new(),
            watch_rx,
            #[cfg(not(target_arch = "wasm32"))]
            _watcher: watcher,
            show_ops: false,
            screenshot: std::env::var_os("KILN_VIZ_SCREENSHOT").map(PathBuf::from),
            frames: 0,
        }
    }

    fn screenshot_step(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        self.frames += 1;
        if self.frames == 8 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = shot {
            let [w, h] = img.size;
            let data: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            let png = tiny_skia::IntSize::from_wh(w as u32, h as u32)
                .and_then(|s| tiny_skia::Pixmap::from_vec(data, s))
                .and_then(|p| p.encode_png().ok());
            match png.map(|b| std::fs::write(&path, b)) {
                Some(Ok(())) => eprintln!("saved {}", path.display()),
                _ => eprintln!("cannot save screenshot to {}", path.display()),
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint();
    }

    fn inputs(&self) -> Inputs<'_> {
        Inputs {
            runs: self.data.runs.iter().collect(),
            archive: self.data.archive.as_ref(),
            calib: self.data.calib.as_deref(),
        }
    }

    fn phases(&self) -> Vec<String> {
        self.data.runs.first().map_or_else(Vec::new, |t| {
            t.phases.iter().map(|p| p.id.clone()).collect()
        })
    }

    fn scene_for(&mut self, w: f32, h: f32) -> &Scene {
        let size = (
            (w * self.zoom).round() as u32,
            (h * self.zoom).round() as u32,
        );
        let sel = self.state.linked(&self.inputs());
        let fresh = self.cache.as_ref().is_some_and(|c| {
            c.spec == self.state.spec && c.sel == sel && c.size == size && c.zoom == self.zoom
        });
        if !fresh {
            let mut spec = self.state.spec.clone();
            spec.width = size.0.max(64) as f32;
            spec.height = size.1.max(64) as f32;
            let scene = kiln_viz_render::render(&self.inputs(), &spec, &sel);
            self.cache = Some(Cached {
                spec: self.state.spec.clone(),
                sel,
                size,
                zoom: self.zoom,
                scene,
            });
        }
        &self.cache.as_ref().expect("scene cached").scene
    }

    fn reload_archive(&mut self) {
        let Some(dir) = self.data.watch.clone() else {
            return;
        };
        match Archive::read_dir(&dir) {
            Ok(a) => {
                self.status = format!("archive reloaded: {} designs", a.latest().len());
                self.data.archive = Some(a);
                self.cache = None;
            }
            Err(e) => self.status = format!("archive reload failed: {}", e.message),
        }
    }

    fn keys(&mut self, ctx: &egui::Context) {
        let typed: Vec<char> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Text(t) => t.chars().next(),
                    _ => None,
                })
                .collect()
        });
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let phases = self.phases();
        for c in typed {
            if c == 'r' || c == 'R' {
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
                self.state.spec.window = None;
            } else if c == 'o' || c == 'O' {
                self.show_ops = !self.show_ops;
            } else if c == 'l' || c == 'L' {
                let layers = self.layers();
                self.state.cycle_layer(&layers);
            } else if let Some(a) = AppState::key(c) {
                self.state.apply(a, &phases);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.state.apply(Action::ClearSelection, &phases);
        }
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) {
            self.save_png();
        }
    }

    /// Stacked-die layers of the first run's floorplan (empty for a single layer).
    fn layers(&self) -> Vec<u8> {
        let Some(t) = self.data.runs.first() else { return vec![] };
        let mut l: Vec<u8> = t
            .floorplan
            .iter()
            .filter(|f| t.resource_kind(&t.resources[f.resource as usize]) == "die")
            .map(|f| f.layer)
            .collect();
        l.sort_unstable();
        l.dedup();
        if l.len() > 1 { l } else { vec![] }
    }

    fn save_png(&mut self) {
        let Some(c) = &self.cache else { return };
        let path = PathBuf::from(format!("kiln-{}.png", self.state.spec.view.name()));
        self.status = match std::fs::write(&path, kiln_viz_render::to_png(&c.scene, 1.0)) {
            Ok(()) => format!("saved {}", path.display()),
            Err(e) => format!("cannot save {}: {e}", path.display()),
        };
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("kiln viz");
            if let Some(t) = self.data.runs.first() {
                ui.label(format!(
                    "{}  {}  tier {:?}  {}",
                    t.manifest.design_name.as_deref().unwrap_or("run"),
                    t.manifest.workload_name.as_deref().unwrap_or(""),
                    t.manifest.tier,
                    t.manifest
                        .provenance
                        .design_hash
                        .chars()
                        .take(14)
                        .collect::<String>()
                ));
            }
            ui.separator();
            for (k, v) in ViewKind::ALL.iter().enumerate() {
                if ui
                    .selectable_label(
                        self.state.spec.view == *v,
                        format!("{} {}", (k + 1) % 10, v.title()),
                    )
                    .clicked()
                {
                    self.state.spec.view = *v;
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            let phases = self.phases();
            egui::ComboBox::from_id_salt("phase")
                .selected_text(
                    self.state
                        .spec
                        .phase
                        .clone()
                        .unwrap_or_else(|| "all phases".into()),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.state.spec.phase, None, "all phases");
                    for p in phases {
                        ui.selectable_value(&mut self.state.spec.phase, Some(p.clone()), p);
                    }
                });
            if self.state.spec.view == ViewKind::Floorplan
                || self.state.spec.view == ViewKind::Compare
            {
                egui::ComboBox::from_id_salt("color")
                    .selected_text(format!("color: {}", self.state.spec.color.name()))
                    .show_ui(ui, |ui| {
                        for c in FloorColor::ALL {
                            ui.selectable_value(&mut self.state.spec.color, c, c.name());
                        }
                    });
                if self.state.spec.view == ViewKind::Floorplan {
                    ui.checkbox(&mut self.state.spec.wires, "wires (W)");
                    if self.state.spec.wires {
                        egui::ComboBox::from_id_salt("wire_color")
                            .selected_text(format!("wires: {}", self.state.spec.wire_color.name()))
                            .show_ui(ui, |ui| {
                                for c in WireColor::ALL {
                                    ui.selectable_value(&mut self.state.spec.wire_color, c, c.name());
                                }
                            });
                    }
                    let layers = self.layers();
                    if !layers.is_empty() {
                        egui::ComboBox::from_id_salt("layer")
                            .selected_text(match self.state.spec.layer {
                                None => "layers: side by side (L)".to_string(),
                                Some(l) => format!("layer {l} (L)"),
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.state.spec.layer, None, "side by side");
                                for l in layers {
                                    ui.selectable_value(&mut self.state.spec.layer, Some(l), format!("layer {l}"));
                                }
                            });
                    }
                }
                if !self.state.crumbs.is_empty() {
                    if ui.button("up (U)").clicked() {
                        self.state.apply(Action::Up, &[]);
                    }
                    ui.label(format!(
                        "root: {}",
                        self.state.spec.root.as_deref().unwrap_or("system")
                    ));
                }
            }
            if self.state.spec.view == ViewKind::Roofline {
                ui.checkbox(&mut self.state.spec.aggregate, "per op family (A)");
            }
            if self.state.spec.view == ViewKind::Archive
                && let Some(a) = &self.data.archive
            {
                let names: Vec<String> = a.meta.axes.iter().map(|x| x.name.clone()).collect();
                for (label, axis) in [
                    ("x", &mut self.state.spec.x_axis),
                    ("y", &mut self.state.spec.y_axis),
                ] {
                    egui::ComboBox::from_id_salt(label)
                        .selected_text(format!(
                            "{label}: {}",
                            names.get(*axis).cloned().unwrap_or_default()
                        ))
                        .show_ui(ui, |ui| {
                            for (i, n) in names.iter().enumerate() {
                                ui.selectable_value(axis, i, n);
                            }
                        });
                }
            }
            if self.state.spec.view == ViewKind::Calibration
                && let Some(c) = &self.data.calib
            {
                let mut devs: Vec<String> = c.iter().map(|r| r.device.clone()).collect();
                devs.sort();
                devs.dedup();
                egui::ComboBox::from_id_salt("device")
                    .selected_text(
                        self.state
                            .spec
                            .device
                            .clone()
                            .unwrap_or_else(|| "all devices".into()),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.state.spec.device, None, "all devices");
                        for d in devs {
                            ui.selectable_value(&mut self.state.spec.device, Some(d.clone()), d);
                        }
                    });
            }
            ui.separator();
            let dark = self.state.spec.theme == ThemeKind::Dark;
            if ui.selectable_label(dark, "dark").clicked() {
                self.state.spec.theme = if dark {
                    ThemeKind::Light
                } else {
                    ThemeKind::Dark
                };
                ui.ctx().set_theme(if dark {
                    egui::Theme::Light
                } else {
                    egui::Theme::Dark
                });
            }
            if ui.button("copy view state").clicked() {
                ui.ctx().copy_text(self.state.state_string());
                self.status = "view state copied (kiln viz <run> --state '<json>')".into();
            }
            if ui.button("save PNG (Cmd-S)").clicked() {
                self.save_png();
            }
            ui.toggle_value(&mut self.show_ops, "ops table (O)");
        });
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        let hits: Vec<Hit> = self.state.sel.items.iter().copied().collect();
        if hits.is_empty() {
            ui.label("click an item; shift-click adds; click a selected floorplan block again to drill in");
            if let Some(t) = self.data.runs.first() {
                ui.separator();
                run_info(ui, t);
            }
            return;
        }
        let mut clicked: Option<Hit> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for h in hits {
                describe(ui, h, &self.data, &self.state.spec, &mut clicked);
                ui.separator();
            }
        });
        if let Some(h) = clicked {
            self.state.sel.items.clear();
            self.state.sel.items.insert(h);
        }
    }

    fn ops_table(&mut self, ui: &mut egui::Ui) {
        let Some(t) = self.data.runs.first() else {
            ui.label("no run loaded");
            return;
        };
        let times = kiln_trace::analysis::op_times(t);
        let bind = t.binding_of_ops();
        let mut clicked = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("ops").striped(true).show(ui, |ui| {
                for h in ["op", "kind", "phase", "time", "FLOPs", "energy", "binding"] {
                    ui.strong(h);
                }
                ui.end_row();
                for (i, o) in t.ops.iter().enumerate() {
                    let sel = self.state.sel.items.contains(&Hit::Op(i as u32));
                    if ui.selectable_label(sel, &o.path).clicked() {
                        clicked = Some(Hit::Op(i as u32));
                    }
                    ui.label(t.op_kind(o));
                    ui.label(t.phase_name(o.phase));
                    ui.label(kiln_viz_render::chart::fmt_time(times[i]));
                    ui.label(kiln_viz_render::chart::fmt_num(o.flops));
                    ui.label(kiln_viz_render::chart::fmt_energy(o.energy_j));
                    ui.label(bind[i].map_or("-", |l| t.binding_name(l.binding)));
                    ui.end_row();
                }
            });
        });
        if let Some(h) = clicked {
            self.state.sel.items.clear();
            self.state.sel.items.insert(h);
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let (resp, painter) = ui.allocate_painter(avail, Sense::click_and_drag());
        let origin = resp.rect.min + self.pan;
        let timeline = self.state.spec.view == ViewKind::Timeline;
        // Zoom and pan: the timeline zooms its time window; other views magnify the re-rendered scene.
        if resp.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            let factor = zoom * (scroll.y / 400.0).exp();
            if (factor - 1.0).abs() > 1e-4 {
                if timeline {
                    self.zoom_window(factor, resp.hover_pos(), resp.rect);
                } else if let Some(p) = resp.hover_pos() {
                    let nz = (self.zoom * factor).clamp(1.0, 64.0);
                    let k = nz / self.zoom;
                    self.pan = (self.pan - (p - resp.rect.min)) * k + (p - resp.rect.min);
                    self.zoom = nz;
                }
            }
        }
        if resp.dragged() {
            if timeline {
                self.pan_window(resp.drag_delta().x, resp.rect.width());
            } else {
                self.pan += resp.drag_delta();
            }
        }
        if self.zoom <= 1.0 {
            self.pan = Vec2::ZERO;
        }
        let (w, h) = (resp.rect.width(), resp.rect.height());
        let shift = ui.input(|i| i.modifiers.shift);
        let hover = resp.hover_pos().map(|p| (p - origin) / 1.0);
        let scene = self.scene_for(w, h).clone();
        let hit = hover.map_or(Hit::None, |p| scene.hit_test(p.x, p.y));
        if hit != self.state.sel.hover {
            self.state.sel.hover = hit;
        }
        paint(&painter.with_clip_rect(resp.rect), &scene, origin);
        if resp.clicked() {
            let inputs = Inputs {
                runs: self.data.runs.iter().collect(),
                archive: self.data.archive.as_ref(),
                calib: self.data.calib.as_deref(),
            };
            self.state.click(hit, shift, &inputs);
        }
        if hit != Hit::None {
            resp.on_hover_text(hover_text(hit, &self.data));
        }
    }

    fn window_ticks(&self) -> Option<(f64, f64, f64)> {
        let t = self.data.runs.first()?;
        let (a, b) = kiln_viz_render::views::timeline::window(t, &self.state.spec);
        Some((a as f64 * t.tick_s(), b as f64 * t.tick_s(), t.tick_s()))
    }

    fn zoom_window(&mut self, factor: f32, at: Option<Pos2>, rect: Rect) {
        let Some((a, b, _)) = self.window_ticks() else {
            return;
        };
        // The plot starts after the 230 px track labels (timeline.rs).
        let x0 = rect.min.x + 16.0 + 230.0;
        let w = (rect.width() - 32.0 - 230.0).max(1.0);
        let f = at.map_or(0.5, |p| f64::from(((p.x - x0) / w).clamp(0.0, 1.0)));
        let c = a + f * (b - a);
        let span = ((b - a) / f64::from(factor)).max(1e-12);
        self.state.spec.window = Some([c - f * span, c + (1.0 - f) * span]);
    }

    fn pan_window(&mut self, dx: f32, width: f32) {
        let Some((a, b, _)) = self.window_ticks() else {
            return;
        };
        let d = -f64::from(dx) / f64::from((width - 262.0).max(1.0)) * (b - a);
        self.state.spec.window = Some([a + d, b + d]);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(rx) = &self.watch_rx {
            let mut changed = false;
            while rx.try_recv().is_ok() {
                changed = true;
            }
            if changed {
                self.reload_archive();
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
        self.keys(&ctx);
        self.screenshot_step(&ctx);
        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                let (errs, warns) = self.data.runs.first().map_or((0, 0), |t| {
                    (
                        t.diagnostics.iter().filter(|d| d.severity == 0).count(),
                        t.diagnostics.iter().filter(|d| d.severity != 0).count(),
                    )
                });
                ui.label(format!("issues: {errs} errors, {warns} warnings"));
                ui.separator();
                ui.label(format!(
                    "zoom {:.1}x | R reset | 1-9, 0 views | C color | W wires | L layer | U up | P phase | Esc clear",
                    self.zoom
                ));
                ui.separator();
                ui.label(&self.status);
            });
        });
        if self.show_ops {
            egui::Panel::bottom("ops")
                .resizable(true)
                .default_size(240.0)
                .show(ui, |ui| self.ops_table(ui));
        }
        egui::Panel::right("inspector")
            .resizable(true)
            .default_size(330.0)
            .show(ui, |ui| self.inspector(ui));
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn watch(dir: &std::path::Path) -> notify::Result<(Receiver<()>, notify::RecommendedWatcher)> {
    use notify::Watcher;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut w = notify::recommended_watcher(move |ev: notify::Result<notify::Event>| {
        if ev.is_ok() {
            let _ = tx.send(());
        }
    })?;
    w.watch(dir, notify::RecursiveMode::NonRecursive)?;
    Ok((rx, w))
}

fn c32(c: scene::Color) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a)
}

fn stroke(s: &scene::Stroke) -> Stroke {
    Stroke::new(s.width, c32(s.color))
}

/// Scene -> egui shapes. Scene units are points; `origin` is the scene's top-left on screen.
pub fn paint(painter: &egui::Painter, s: &Scene, origin: Pos2) {
    let at = |p: [f32; 2]| origin + Vec2::new(p[0], p[1]);
    let rect =
        |r: &scene::Rect| Rect::from_min_size(origin + Vec2::new(r.x, r.y), Vec2::new(r.w, r.h));
    painter.rect_filled(
        Rect::from_min_size(origin, Vec2::new(s.width, s.height)),
        CornerRadius::ZERO,
        c32(s.background),
    );
    let mut clips: Vec<Rect> = vec![painter.clip_rect()];
    let mut p = painter.clone();
    for prim in &s.prims {
        match prim {
            Prim::ClipPush(r) => {
                let c = rect(r).intersect(*clips.last().expect("root clip"));
                clips.push(c);
                p = painter.with_clip_rect(c);
            }
            Prim::ClipPop => {
                if clips.len() > 1 {
                    clips.pop();
                }
                p = painter.with_clip_rect(*clips.last().expect("root clip"));
            }
            Prim::Rect {
                r,
                fill,
                stroke: st,
                radius,
                ..
            } => {
                let rr = rect(r);
                if !p.clip_rect().intersects(rr) {
                    continue;
                }
                let cr = CornerRadius::same(radius.round().clamp(0.0, 255.0) as u8);
                if let Some(f) = fill {
                    p.rect_filled(rr, cr, c32(*f));
                }
                if let Some(st) = st {
                    if st.dash > 0.0 {
                        let pts = vec![
                            rr.left_top(),
                            rr.right_top(),
                            rr.right_bottom(),
                            rr.left_bottom(),
                            rr.left_top(),
                        ];
                        p.extend(Shape::dashed_line(&pts, stroke(st), st.dash, st.dash));
                    } else {
                        p.rect_stroke(rr, cr, stroke(st), StrokeKind::Inside);
                    }
                }
            }
            Prim::Line {
                pts, stroke: st, ..
            } => {
                let pts: Vec<Pos2> = pts.iter().map(|q| at(*q)).collect();
                if st.dash > 0.0 {
                    p.extend(Shape::dashed_line(&pts, stroke(st), st.dash, st.dash));
                } else {
                    p.add(Shape::line(pts, stroke(st)));
                }
            }
            Prim::Polygon {
                pts,
                fill,
                stroke: st,
                ..
            } => {
                let pts: Vec<Pos2> = pts.iter().map(|q| at(*q)).collect();
                p.add(Shape::convex_polygon(
                    pts,
                    fill.map_or(Color32::TRANSPARENT, c32),
                    st.as_ref().map_or(Stroke::NONE, stroke),
                ));
            }
            Prim::Circle {
                c,
                r,
                fill,
                stroke: st,
                ..
            } => {
                if let Some(f) = fill {
                    p.circle_filled(at(*c), *r, c32(*f));
                }
                if let Some(st) = st {
                    p.circle_stroke(at(*c), *r, stroke(st));
                }
            }
            Prim::Text {
                pos,
                text,
                size,
                color,
                h,
                v,
                mono,
                vertical,
            } => {
                let font = if *mono {
                    FontId::monospace(*size)
                } else {
                    FontId::proportional(*size)
                };
                let align = Align2([
                    match h {
                        HAlign::Left => egui::Align::Min,
                        HAlign::Center => egui::Align::Center,
                        HAlign::Right => egui::Align::Max,
                    },
                    match v {
                        VAlign::Top => egui::Align::Min,
                        VAlign::Middle => egui::Align::Center,
                        VAlign::Baseline | VAlign::Bottom => egui::Align::Max,
                    },
                ]);
                if *vertical {
                    let galley = p.layout_no_wrap(text.clone(), font, c32(*color));
                    let (gw, gh) = (galley.size().x, galley.size().y);
                    let ox = match h {
                        HAlign::Left => 0.0,
                        HAlign::Center => -gw / 2.0,
                        HAlign::Right => -gw,
                    };
                    let oy = match v {
                        VAlign::Top => 0.0,
                        VAlign::Middle => -gh / 2.0,
                        VAlign::Baseline | VAlign::Bottom => -gh,
                    };
                    // Local (x along the text, y down) rotated by -90 degrees on screen.
                    let anchor = at(*pos) + Vec2::new(oy, -ox);
                    p.add(
                        egui::epaint::TextShape::new(anchor, galley, c32(*color))
                            .with_angle(-std::f32::consts::FRAC_PI_2),
                    );
                } else {
                    p.text(at(*pos), align, text, font, c32(*color));
                }
            }
        }
    }
}

fn hover_text(h: Hit, d: &Data) -> String {
    let t = d.runs.first();
    match (h, t) {
        (Hit::Resource(r), Some(t)) => t.resources.get(r as usize).map_or_else(String::new, |x| {
            let mut s = format!("{} ({})", x.path, t.resource_kind(x));
            if let Some(f) = t.floorplan.iter().find(|f| f.resource == r) {
                if let Some(a) = f.area_um2 {
                    s.push_str(&format!("  {}", kiln_viz_render::views::floorplan::fmt_area(a)));
                }
                if let Some(w) = f.leak_w {
                    s.push_str(&format!(", {:.3} W static", w));
                }
            }
            if let Some(w) = t.wires.iter().find(|w| w.link == r) {
                s.push_str(&format!(
                    "  {:.2} mm, {}, {:.2} ns, {:.3} pJ/bit",
                    w.length_um / 1000.0,
                    kiln_viz_render::chart::fmt_bw(w.bw_bps),
                    w.latency_s * 1e9,
                    w.e_j_per_bit * 1e12
                ));
            }
            s
        }),
        (Hit::Op(o), Some(t)) => t.ops.get(o as usize).map_or_else(String::new, |x| {
            format!("{} ({}, {})", x.path, t.op_kind(x), t.phase_name(x.phase))
        }),
        (Hit::Phase(p), Some(t)) => t.phase_name(p).to_string(),
        (Hit::Binding(b), Some(t)) => {
            kiln_viz_render::theme::binding_label(t.binding_name(b)).to_string()
        }
        (Hit::Calib(i), _) => d
            .calib
            .as_ref()
            .and_then(|c| c.get(i as usize))
            .map_or_else(String::new, |r| format!("{} {}", r.device, r.name)),
        (Hit::Cell(x, y), _) => format!("cell ({x}, {y})"),
        _ => String::new(),
    }
}

fn run_info(ui: &mut egui::Ui, t: &Trace) {
    let m = &t.manifest;
    ui.strong("Run");
    ui.label(format!(
        "level {:?}, tier {:?}, schema {}",
        m.level, m.tier, m.schema_version
    ));
    ui.label(format!(
        "kiln {} ({})",
        m.provenance.kiln_version, m.provenance.git_hash
    ));
    ui.label(format!("design {}", m.provenance.design_hash));
    ui.label(format!("workload {}", m.provenance.workload_hash));
    ui.label(format!("calibration {}", m.provenance.calibration_hash));
    for p in &t.phases {
        ui.separator();
        ui.strong(&p.id);
        ui.label(format!(
            "makespan {} [{}, {}]",
            kiln_viz_render::chart::fmt_time(p.makespan_s),
            kiln_viz_render::chart::fmt_time(p.makespan_low_s.unwrap_or(p.makespan_s)),
            kiln_viz_render::chart::fmt_time(p.makespan_high_s.unwrap_or(p.makespan_s))
        ));
        ui.label(format!(
            "energy {}, avg power {}",
            kiln_viz_render::chart::fmt_energy(p.energy_j),
            kiln_viz_render::chart::fmt_power(p.avg_power_w)
        ));
        ui.label(egui::RichText::new(&p.summary).small());
    }
    for n in &m.notes {
        ui.label(egui::RichText::new(n).italics().small());
    }
}

fn describe(ui: &mut egui::Ui, h: Hit, d: &Data, spec: &ViewSpec, clicked: &mut Option<Hit>) {
    use kiln_viz_render::chart::{fmt_bw, fmt_bytes, fmt_energy, fmt_num, fmt_time};
    let t = d.runs.first();
    match (h, t) {
        (Hit::Resource(r), Some(t)) => {
            let Some(x) = t.resources.get(r as usize) else {
                return;
            };
            ui.strong(&x.path);
            ui.label(format!("kind {}", t.resource_kind(x)));
            if let Some(l) = x.mem_level {
                ui.label(format!("memory level {l}"));
            }
            if let Some(c) = x.capacity_b {
                ui.label(format!("capacity {}", fmt_bytes(c)));
            }
            if let Some(b) = x.peak_bw_bps {
                ui.label(format!("peak bandwidth {}", fmt_bw(b)));
            }
            let st = kiln_trace::analysis::resource_stats(
                t,
                kiln_trace::analysis::phase_code(t, spec.phase.as_deref()),
            );
            let s = st[r as usize];
            if let Some(u) = s.util {
                ui.label(format!(
                    "busy {:.1}% ({}), {} moved, {}",
                    100.0 * u,
                    fmt_time(s.busy_s),
                    fmt_bytes(s.bytes),
                    fmt_energy(s.energy_j)
                ));
            }
            ui.horizontal(|ui| {
                if ui.small_button("copy IR path").clicked() {
                    ui.ctx().copy_text(x.path.clone());
                }
            });
            let ops: Vec<u32> = t
                .limiters
                .iter()
                .filter(|l| l.rank == 0 && l.resource == Some(r))
                .map(|l| l.op)
                .collect();
            if !ops.is_empty() {
                ui.label(format!("{} ops bound here:", ops.len()));
                for o in ops.iter().take(40) {
                    if ui.link(&t.ops[*o as usize].path).clicked() {
                        *clicked = Some(Hit::Op(*o));
                    }
                }
            }
        }
        (Hit::Op(o), Some(t)) => {
            let Some(x) = t.ops.get(o as usize) else {
                return;
            };
            let times = kiln_trace::analysis::op_times(t);
            ui.strong(&x.path);
            ui.label(format!(
                "{} | {} | layer {}",
                t.op_kind(x),
                t.phase_name(x.phase),
                x.layer.map_or("-".into(), |l| l.to_string())
            ));
            ui.label(format!(
                "attributed time {} (envelope {})",
                fmt_time(times[o as usize]),
                fmt_time(x.time_s(t.tick_s()))
            ));
            if let (Some(lo), Some(hi)) = (x.time_low_s, x.time_high_s) {
                ui.label(format!(
                    "envelope at corners [{}, {}]",
                    fmt_time(lo),
                    fmt_time(hi)
                ));
            }
            ui.label(format!(
                "{} FLOP, {} MACs issued / {} useful",
                fmt_num(x.flops),
                fmt_num(x.macs_issued as f64),
                fmt_num(x.macs_useful as f64)
            ));
            for (l, b) in x.bytes_by_level.iter().enumerate() {
                if *b > 0.0 {
                    ui.label(format!("  level {l}: {}", fmt_bytes(*b)));
                }
            }
            ui.label(format!("energy {}", fmt_energy(x.energy_j)));
            for l in t.limiters.iter().filter(|l| l.op == o) {
                let what = l.resource.map_or(String::new(), |r| {
                    format!(" on {}", t.resources[r as usize].path)
                });
                let line = format!(
                    "{} {}{what}",
                    if l.rank == 0 { "bound by" } else { "runner-up" },
                    kiln_viz_render::theme::binding_label(t.binding_name(l.binding))
                );
                if let Some(r) = l.resource {
                    if ui.link(line).clicked() {
                        *clicked = Some(Hit::Resource(r));
                    }
                } else {
                    ui.label(line);
                }
            }
            if x.group != NONE_U32 {
                ui.label(format!("execution group {}", x.group));
            }
        }
        (Hit::Phase(p), Some(t)) => {
            if let Some(ph) = t.phases.iter().find(|x| x.phase == p) {
                ui.strong(&ph.id);
                ui.label(&ph.summary);
            }
        }
        (Hit::Binding(b), Some(t)) => {
            ui.strong(kiln_viz_render::theme::binding_label(t.binding_name(b)));
        }
        (Hit::Cell(..), _) => {
            if let (Some(a), Some(id)) = (&d.archive, &spec.lineage_of)
                && let Some(r) = a.design(id)
            {
                ui.strong(&r.design_id);
                ui.label(format!(
                    "generation {}, status {:?}",
                    r.generation, r.status
                ));
                ui.label(format!(
                    "fitness {:.4} [{}, {}]",
                    r.fitness,
                    r.fitness_low.map_or("-".into(), |x| format!("{x:.4}")),
                    r.fitness_high.map_or("-".into(), |x| format!("{x:.4}"))
                ));
                ui.label(format!("operator {}: {}", r.operator, r.mutation_summary));
                ui.label(format!("parents {}", r.parent_ids.join(", ")));
                for (k, v) in &r.fitness_components {
                    ui.label(format!("  {k}: {v:.4}"));
                }
                if let Some(p) = a.run_path(r) {
                    ui.label(format!(
                        "run: {}{}",
                        p.display(),
                        if p.exists() { "" } else { " (missing)" }
                    ));
                }
            }
        }
        (Hit::Calib(i), _) => {
            if let Some(r) = d.calib.as_ref().and_then(|c| c.get(i as usize)) {
                ui.strong(&r.name);
                ui.label(format!("{} | {} | {}", r.device, r.op_kind, r.split));
                ui.label(format!(
                    "measured {}, predicted {} ({:+.1}%)",
                    fmt_time(r.measured_s),
                    fmt_time(r.predicted_s_tier_a),
                    100.0 * (r.predicted_s_tier_a / r.measured_s - 1.0)
                ));
                if let Some(u) = r.uncalibrated_s {
                    ui.label(format!("uncalibrated {}", fmt_time(u)));
                }
            }
        }
        _ => {
            ui.label(format!("{h:?}"));
        }
    }
}

/// Web entry point of the same viewer (05 §2.2): a `.kiln` file's bytes rendered into `canvas`.
#[cfg(target_arch = "wasm32")]
pub async fn start_web(
    canvas: eframe::web_sys::HtmlCanvasElement,
    kiln: Vec<u8>,
    spec: ViewSpec,
) -> Result<(), eframe::wasm_bindgen::JsValue> {
    let trace = kiln_trace::container::read_kiln(&kiln)
        .map_err(|d| eframe::wasm_bindgen::JsValue::from_str(&d.message))?;
    let data = Data {
        runs: vec![trace],
        archive: None,
        calib: None,
        watch: None,
    };
    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(move |_cc| Ok(Box::new(App::new(data, spec)))),
        )
        .await
}
