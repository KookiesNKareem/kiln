//! Colors (05 §4.4): cividis (default sequential, colorblind-safe) and viridis, a blue-white-red diverging
//! map centred at 0, and a fixed 12-color categorical palette shared by every view and the app.

use serde::{Deserialize, Serialize};

use crate::scene::Color;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemeKind {
    #[default]
    Light,
    Dark,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    pub kind: ThemeKind,
    pub bg: Color,
    pub panel: Color,
    pub fg: Color,
    pub muted: Color,
    pub grid: Color,
    pub outline: Color,
    pub accent: Color,
    pub bad: Color,
    pub good: Color,
    pub hatch: Color,
    pub idle: Color,
}

impl Theme {
    pub fn get(kind: ThemeKind) -> Theme {
        match kind {
            ThemeKind::Light => Theme {
                kind,
                bg: Color::rgb(255, 255, 255),
                panel: Color::rgb(246, 246, 248),
                fg: Color::rgb(28, 30, 36),
                muted: Color::rgb(110, 114, 124),
                grid: Color::rgb(226, 228, 233),
                outline: Color::rgb(160, 164, 172),
                accent: Color::rgb(0, 114, 178),
                bad: Color::rgb(213, 50, 40),
                good: Color::rgb(0, 140, 90),
                hatch: Color::rgb(190, 193, 200),
                idle: Color::rgb(238, 239, 242),
            },
            ThemeKind::Dark => Theme {
                kind,
                bg: Color::rgb(24, 26, 31),
                panel: Color::rgb(34, 37, 44),
                fg: Color::rgb(226, 228, 233),
                muted: Color::rgb(150, 155, 165),
                grid: Color::rgb(52, 56, 64),
                outline: Color::rgb(96, 101, 112),
                accent: Color::rgb(86, 180, 233),
                bad: Color::rgb(240, 90, 80),
                good: Color::rgb(80, 200, 140),
                hatch: Color::rgb(70, 74, 84),
                idle: Color::rgb(40, 43, 50),
            },
        }
    }

    /// Text color readable on `fill`.
    pub fn on(&self, fill: Color) -> Color {
        if fill.luma() > 0.55 {
            Color::rgb(20, 20, 24)
        } else {
            Color::rgb(245, 245, 248)
        }
    }
}

const CIVIDIS: [[u8; 3]; 17] = [
    [0, 34, 78],
    [0, 46, 106],
    [26, 56, 111],
    [50, 67, 109],
    [67, 78, 108],
    [83, 90, 109],
    [97, 101, 111],
    [111, 112, 115],
    [125, 124, 120],
    [140, 136, 120],
    [155, 148, 118],
    [171, 160, 114],
    [188, 174, 108],
    [205, 187, 99],
    [222, 201, 88],
    [240, 216, 70],
    [254, 232, 56],
];

const VIRIDIS: [[u8; 3]; 17] = [
    [68, 1, 84],
    [72, 24, 106],
    [71, 45, 123],
    [66, 64, 134],
    [59, 82, 139],
    [51, 99, 141],
    [44, 114, 142],
    [38, 130, 142],
    [33, 145, 140],
    [31, 160, 136],
    [40, 174, 128],
    [63, 188, 115],
    [94, 201, 98],
    [132, 212, 75],
    [173, 220, 48],
    [216, 226, 25],
    [253, 231, 37],
];

fn lut(t: f64, l: &[[u8; 3]; 17]) -> Color {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let x = t * 16.0;
    let i = (x.floor() as usize).min(15);
    let f = (x - i as f64) as f32;
    let a = Color::rgb(l[i][0], l[i][1], l[i][2]);
    let b = Color::rgb(l[i + 1][0], l[i + 1][1], l[i + 1][2]);
    a.lerp(b, f)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorMap {
    #[default]
    Cividis,
    Viridis,
}

impl ColorMap {
    pub fn at(self, t: f64) -> Color {
        match self {
            ColorMap::Cividis => lut(t, &CIVIDIS),
            ColorMap::Viridis => lut(t, &VIRIDIS),
        }
    }
}

/// Blue-white-red for `t` in [-1, 1].
pub fn diverging(t: f64) -> Color {
    let t = if t.is_finite() {
        t.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let white = Color::rgb(247, 247, 247);
    if t < 0.0 {
        white.lerp(Color::rgb(33, 102, 172), (-t) as f32)
    } else {
        white.lerp(Color::rgb(178, 24, 43), t as f32)
    }
}

pub const CATEGORICAL: [Color; 12] = [
    Color::rgb(0, 114, 178),
    Color::rgb(230, 159, 0),
    Color::rgb(0, 158, 115),
    Color::rgb(213, 94, 0),
    Color::rgb(204, 121, 167),
    Color::rgb(86, 180, 233),
    Color::rgb(240, 228, 66),
    Color::rgb(136, 34, 85),
    Color::rgb(68, 170, 153),
    Color::rgb(51, 34, 136),
    Color::rgb(17, 119, 51),
    Color::rgb(153, 153, 153),
];

pub fn cat(i: usize) -> Color {
    CATEGORICAL[i % CATEGORICAL.len()]
}

/// Fixed color per binding class name, so limiter colors agree across views.
pub fn binding_color(name: &str) -> Color {
    match name {
        "compute" => cat(0),
        "dram" => cat(1),
        "link" => cat(2),
        "port" => cat(5),
        "dependency" => cat(11),
        "overhead" => cat(4),
        "nmp" => cat(8),
        "contention" => cat(3),
        "pipeline_bubble" => cat(7),
        _ => cat(9),
    }
}

/// Human label of a binding class.
pub fn binding_label(name: &str) -> &str {
    match name {
        "compute" => "compute",
        "dram" => "off-chip memory bw",
        "link" => "link bw",
        "port" => "on-chip memory port",
        "dependency" => "dependency chain",
        "overhead" => "launch/sync overhead",
        "nmp" => "near-memory compute",
        "contention" => "contention",
        "pipeline_bubble" => "pipeline bubble",
        x => x,
    }
}
