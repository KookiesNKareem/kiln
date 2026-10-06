//! GUI-free viewer state (05 §4.2-4.3): active view, per-view specs, linked selection, hover, the view-state
//! string and keyboard actions.

use kiln_viz_render::scene::{Hit, Scene};
use kiln_viz_render::views::{FloorColor, ViewKind, ViewSpec};
use kiln_viz_render::{Inputs, Selection};

#[derive(Clone, Debug, PartialEq)]
pub struct AppState {
    pub spec: ViewSpec,
    pub sel: Selection,
    /// Breadcrumb of floorplan drill-ins (resource paths).
    pub crumbs: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    View(ViewKind),
    CycleColor,
    ClearSelection,
    Up,
    ToggleAggregate,
    NextPhase,
}

impl AppState {
    pub fn new(spec: ViewSpec) -> Self {
        AppState {
            spec,
            sel: Selection::default(),
            crumbs: vec![],
        }
    }

    /// `1`..`8` switch views, `C` cycles the floorplan color mode, `Esc` clears, `U` goes up (05 §4.3).
    pub fn key(c: char) -> Option<Action> {
        Some(match c {
            '1'..='8' => Action::View(ViewKind::ALL[(c as u8 - b'1') as usize]),
            'c' | 'C' => Action::CycleColor,
            'u' | 'U' => Action::Up,
            'a' | 'A' => Action::ToggleAggregate,
            'p' | 'P' => Action::NextPhase,
            '\u{1b}' => Action::ClearSelection,
            _ => return None,
        })
    }

    pub fn apply(&mut self, a: Action, phases: &[String]) {
        match a {
            Action::View(v) => self.spec.view = v,
            Action::CycleColor => {
                let i = FloorColor::ALL
                    .iter()
                    .position(|c| *c == self.spec.color)
                    .unwrap_or(0);
                self.spec.color = FloorColor::ALL[(i + 1) % FloorColor::ALL.len()];
            }
            Action::ClearSelection => self.sel.items.clear(),
            Action::Up => {
                self.crumbs.pop();
                self.spec.root = self.crumbs.last().cloned();
            }
            Action::ToggleAggregate => self.spec.aggregate = !self.spec.aggregate,
            Action::NextPhase => {
                let i = self
                    .spec
                    .phase
                    .as_ref()
                    .and_then(|p| phases.iter().position(|x| x == p));
                self.spec.phase = match i {
                    None => phases.first().cloned(),
                    Some(i) if i + 1 < phases.len() => Some(phases[i + 1].clone()),
                    Some(_) => None,
                };
            }
        }
    }

    /// Click: select (shift adds); a second click on a selected container drills into it.
    pub fn click(&mut self, hit: Hit, shift: bool, inputs: &Inputs) {
        if hit == Hit::None {
            if !shift {
                self.sel.items.clear();
            }
            return;
        }
        if !shift {
            if self.spec.view == ViewKind::Floorplan
                && self.sel.items.len() == 1
                && self.sel.items.contains(&hit)
                && let (Hit::Resource(r), Some(t)) = (hit, inputs.runs.first())
            {
                let path = t.resources[r as usize].path.clone();
                self.crumbs.push(path.clone());
                self.spec.root = Some(path);
                self.sel.items.clear();
                return;
            }
            self.sel.items.clear();
        }
        self.sel.items.insert(hit);
        if let (Hit::Cell(..), Some(a)) = (hit, inputs.archive)
            && let Some(id) = cell_design(a, &self.spec, hit)
        {
            self.spec.lineage_of = Some(id);
        }
    }

    /// Hits linked to `hit` across views: an op selects its binding resource and vice versa (05 §4.2).
    pub fn linked(&self, inputs: &Inputs) -> Selection {
        let mut out = self.sel.clone();
        let Some(t) = inputs.runs.first() else {
            return out;
        };
        let bind = t.binding_of_ops();
        for h in self
            .sel
            .items
            .iter()
            .chain(std::iter::once(&self.sel.hover))
        {
            match *h {
                Hit::Op(o) => {
                    if let Some(r) = bind
                        .get(o as usize)
                        .copied()
                        .flatten()
                        .and_then(|l| l.resource)
                    {
                        out.items.insert(Hit::Resource(r));
                    }
                }
                Hit::Resource(r) => {
                    for (i, l) in bind.iter().enumerate() {
                        if l.is_some_and(|l| l.resource == Some(r)) {
                            out.items.insert(Hit::Op(i as u32));
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The view-state string (05 §4.2): JSON of the view spec, restorable with `--state`.
    pub fn state_string(&self) -> String {
        serde_json::to_string(&self.spec).expect("view spec serializes")
    }

    pub fn restore(s: &str) -> Result<ViewSpec, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    pub fn scene(&self, inputs: &Inputs) -> Scene {
        kiln_viz_render::render(inputs, &self.spec, &self.linked(inputs))
    }
}

fn cell_design(a: &kiln_trace::archive::Archive, spec: &ViewSpec, hit: Hit) -> Option<String> {
    let Hit::Cell(x, y) = hit else { return None };
    let latest = a.latest();
    let g = kiln_viz_render::views::archive::grid(a, &latest, spec.x_axis, spec.y_axis);
    g.best.get(&(x, y)).map(|&k| latest[k].design_id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_state_round_trip() {
        let mut s = AppState::new(ViewSpec::default());
        s.apply(AppState::key('4').unwrap(), &[]);
        assert_eq!(s.spec.view, ViewKind::Roofline);
        s.apply(AppState::key('c').unwrap(), &[]);
        assert_eq!(s.spec.color, FloorColor::Idle);
        let phases = vec!["decode".to_string(), "prefill".to_string()];
        s.apply(Action::NextPhase, &phases);
        s.apply(Action::NextPhase, &phases);
        assert_eq!(s.spec.phase.as_deref(), Some("prefill"));
        s.apply(Action::NextPhase, &phases);
        assert_eq!(s.spec.phase, None);
        assert_eq!(AppState::restore(&s.state_string()).unwrap(), s.spec);
    }

    #[test]
    fn click_selects_links_and_drills_on_a_golden_trace() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kiln-trace/tests/golden/trace/tpu_v5e_decode_b1_ops_v1.0.kiln");
        let t = kiln_trace::container::read_kiln(&std::fs::read(p).unwrap()).unwrap();
        let inputs = Inputs {
            runs: vec![&t],
            ..Default::default()
        };
        let mut s = AppState::new(ViewSpec {
            width: 900.0,
            height: 560.0,
            ..ViewSpec::default()
        });
        let scene = s.scene(&inputs);
        // A container block (has children) of the floorplan, found through hit-testing its rect center.
        let children = t.children();
        let (hit, _) = scene
            .prims
            .iter()
            .find_map(|p| match p {
                kiln_viz_render::scene::Prim::Rect {
                    r,
                    hit: Hit::Resource(i),
                    ..
                } if !children[*i as usize].is_empty() && r.w > 40.0 => {
                    let h = scene.hit_test(r.x + 2.0, r.y + 2.0);
                    (h == Hit::Resource(*i)).then_some((h, *r))
                }
                _ => None,
            })
            .expect("a clickable container");
        s.click(hit, false, &inputs);
        assert!(s.sel.items.contains(&hit));
        s.click(hit, false, &inputs);
        let Hit::Resource(r) = hit else {
            unreachable!()
        };
        assert_eq!(
            s.spec.root.as_deref(),
            Some(t.resources[r as usize].path.as_str())
        );
        s.apply(Action::Up, &[]);
        assert_eq!(s.spec.root, None);
        // An op selected anywhere highlights its binding resource (linked selection).
        let (op, res) = t
            .limiters
            .iter()
            .filter(|l| l.rank == 0)
            .find_map(|l| Some((l.op, l.resource?)))
            .expect("an op bound to a resource");
        s.click(Hit::Op(op), false, &inputs);
        assert!(s.linked(&inputs).items.contains(&Hit::Resource(res)));
    }
}
