//! Canvas views (05 §6): pure functions `data x ViewSpec x Selection -> Scene`.

pub mod archive;
pub mod bottleneck;
pub mod calibration;
pub mod compare;
pub mod design;
pub mod floorplan;
pub mod placed;
pub mod roofline;
pub mod timeline;
pub mod wires;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::scene::Hit;
use crate::theme::{ColorMap, Theme, ThemeKind};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ViewKind {
    #[default]
    Floorplan,
    Noc,
    Timeline,
    Roofline,
    Bottleneck,
    Compare,
    Archive,
    Calibration,
    Design,
    Wires,
}

impl ViewKind {
    pub const ALL: [ViewKind; 10] = [
        ViewKind::Floorplan,
        ViewKind::Noc,
        ViewKind::Timeline,
        ViewKind::Roofline,
        ViewKind::Bottleneck,
        ViewKind::Compare,
        ViewKind::Archive,
        ViewKind::Calibration,
        ViewKind::Design,
        ViewKind::Wires,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ViewKind::Floorplan => "floorplan",
            ViewKind::Noc => "noc",
            ViewKind::Timeline => "timeline",
            ViewKind::Roofline => "roofline",
            ViewKind::Bottleneck => "bottleneck",
            ViewKind::Compare => "compare",
            ViewKind::Archive => "archive",
            ViewKind::Calibration => "calibration",
            ViewKind::Design => "design",
            ViewKind::Wires => "wires",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            ViewKind::Floorplan => "Floorplan",
            ViewKind::Noc => "NoC",
            ViewKind::Timeline => "Timeline",
            ViewKind::Roofline => "Roofline",
            ViewKind::Bottleneck => "Bottleneck",
            ViewKind::Compare => "Compare",
            ViewKind::Archive => "Evolution",
            ViewKind::Calibration => "Calibration",
            ViewKind::Design => "Design",
            ViewKind::Wires => "Wires",
        }
    }

    pub fn parse(s: &str) -> Option<ViewKind> {
        ViewKind::ALL
            .into_iter()
            .find(|v| v.name() == s || (s == "evolution" && *v == ViewKind::Archive))
    }
}

/// Floorplan color modes (05 §6.1): run metrics (utilization, idle, energy, bytes), block kind, and the
/// physical model's silicon area, static power and static power density (placed floorplans).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloorColor {
    #[default]
    Utilization,
    Idle,
    Energy,
    Bytes,
    Kind,
    Area,
    Power,
    Density,
}

impl FloorColor {
    pub const ALL: [FloorColor; 8] = [
        FloorColor::Utilization,
        FloorColor::Idle,
        FloorColor::Energy,
        FloorColor::Bytes,
        FloorColor::Kind,
        FloorColor::Area,
        FloorColor::Power,
        FloorColor::Density,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FloorColor::Utilization => "utilization",
            FloorColor::Idle => "idle",
            FloorColor::Energy => "energy",
            FloorColor::Bytes => "bytes",
            FloorColor::Kind => "kind",
            FloorColor::Area => "area",
            FloorColor::Power => "power",
            FloorColor::Density => "density",
        }
    }

    pub fn parse(s: &str) -> Option<FloorColor> {
        FloorColor::ALL.into_iter().find(|c| c.name() == s)
    }

    /// Needs a run (per-resource aggregates).
    pub fn needs_run(self) -> bool {
        matches!(
            self,
            FloorColor::Utilization | FloorColor::Idle | FloorColor::Energy | FloorColor::Bytes
        )
    }
}

/// Wire color modes: run traffic (utilization, bytes) or the link's physical attributes. `Auto` is
/// utilization for a run and bandwidth for a design.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireColor {
    #[default]
    Auto,
    Utilization,
    Traffic,
    Bandwidth,
    Length,
    Energy,
    Latency,
}

impl WireColor {
    pub const ALL: [WireColor; 7] = [
        WireColor::Auto,
        WireColor::Utilization,
        WireColor::Traffic,
        WireColor::Bandwidth,
        WireColor::Length,
        WireColor::Energy,
        WireColor::Latency,
    ];

    pub fn name(self) -> &'static str {
        match self {
            WireColor::Auto => "auto",
            WireColor::Utilization => "utilization",
            WireColor::Traffic => "traffic",
            WireColor::Bandwidth => "bandwidth",
            WireColor::Length => "length",
            WireColor::Energy => "energy",
            WireColor::Latency => "latency",
        }
    }

    pub fn parse(s: &str) -> Option<WireColor> {
        WireColor::ALL.into_iter().find(|c| c.name() == s)
    }
}

/// Everything a view needs besides its data; serializable as the view-state string (05 §4.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewSpec {
    pub view: ViewKind,
    pub width: f32,
    pub height: f32,
    pub theme: ThemeKind,
    pub cmap: ColorMap,
    /// Phase id; `None` = all phases.
    pub phase: Option<String>,
    pub color: FloorColor,
    /// Floorplan subtree to draw (resource path); `None` = whole system.
    pub root: Option<String>,
    /// Container on-screen size (px) above which children are drawn (05 §6.1: 24 px).
    pub drill_px: f32,
    pub labels: bool,
    /// Timeline window in seconds on the trace time axis.
    pub window: Option<[f64; 2]>,
    /// Roofline: one point per op family (aggregated across layers) instead of per op.
    pub aggregate: bool,
    /// Rows in top-N lists (bottleneck limiters, compare waterfall, calibration outliers).
    pub top: usize,
    /// Archive grid axes (descriptor indices) and evolution color.
    pub x_axis: usize,
    pub y_axis: usize,
    /// Show the archive lineage of this design id.
    pub lineage_of: Option<String>,
    /// Calibration: only this device.
    pub device: Option<String>,
    /// Floorplan: stacked-die layer to show (others as outlines); `None` = every layer, side by side.
    pub layer: Option<u8>,
    /// Floorplan: draw links between blocks.
    pub wires: bool,
    pub wire_color: WireColor,
}

impl Default for ViewSpec {
    fn default() -> Self {
        ViewSpec {
            view: ViewKind::Floorplan,
            width: 1600.0,
            height: 1000.0,
            theme: ThemeKind::Light,
            cmap: ColorMap::Cividis,
            phase: None,
            color: FloorColor::Utilization,
            root: None,
            drill_px: 24.0,
            labels: true,
            window: None,
            aggregate: false,
            top: 10,
            x_axis: 0,
            y_axis: 1,
            lineage_of: None,
            device: None,
            layer: None,
            wires: true,
            wire_color: WireColor::Auto,
        }
    }
}

impl ViewSpec {
    pub fn theme(&self) -> Theme {
        Theme::get(self.theme)
    }
}

/// Linked selection and hover (05 §4.2), shared by every view.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub items: BTreeSet<Hit>,
    pub hover: Hit,
}

impl Selection {
    pub fn has(&self, h: Hit) -> bool {
        h != Hit::None && (self.items.contains(&h) || self.hover == h)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.hover == Hit::None
    }
}
