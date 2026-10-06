//! Retained 2D primitive list (05 §2.3): what every canvas view produces, rasterized headless (tiny-skia),
//! written as SVG, or converted to egui shapes by `kiln-viz`. Coordinates are logical pixels, y down.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b, a: 255 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn with_alpha(self, a: u8) -> Self {
        Color { a, ..self }
    }

    pub fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    pub fn lerp(self, o: Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        let m = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t).round() as u8;
        Color {
            r: m(self.r, o.r),
            g: m(self.g, o.g),
            b: m(self.b, o.b),
            a: m(self.a, o.a),
        }
    }

    /// Relative luminance (sRGB approximation), for picking label color on a fill.
    pub fn luma(self) -> f32 {
        (0.2126 * f32::from(self.r) + 0.7152 * f32::from(self.g) + 0.0722 * f32::from(self.b))
            / 255.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Rect { x, y, w, h }
    }

    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.right() && y >= self.y && y <= self.bottom()
    }

    pub fn inset(&self, l: f32, t: f32, r: f32, b: f32) -> Rect {
        Rect {
            x: self.x + l,
            y: self.y + t,
            w: (self.w - l - r).max(0.0),
            h: (self.h - t - b).max(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub width: f32,
    pub color: Color,
    /// Dash length (gap = dash); 0 = solid.
    pub dash: f32,
}

impl Stroke {
    pub const fn solid(width: f32, color: Color) -> Self {
        Stroke {
            width,
            color,
            dash: 0.0,
        }
    }

    pub const fn dashed(width: f32, color: Color, dash: f32) -> Self {
        Stroke { width, color, dash }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HAlign {
    Left,
    Center,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VAlign {
    Top,
    Middle,
    Baseline,
    Bottom,
}

/// What a primitive stands for, for hit-testing and linked selection (05 §4.2).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum Hit {
    #[default]
    None,
    Resource(u32),
    Op(u32),
    /// Op family (aggregated across layers) by index into the view's family list.
    Family(u32),
    Binding(u8),
    Phase(u8),
    /// Archive design row.
    Design(u32),
    Cell(u32, u32),
    /// Calibration report row.
    Calib(u32),
    /// Compare: run index and op index.
    RunOp(u8, u32),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Prim {
    Rect {
        r: Rect,
        fill: Option<Color>,
        stroke: Option<Stroke>,
        radius: f32,
        hit: Hit,
    },
    Line {
        pts: Vec<[f32; 2]>,
        stroke: Stroke,
        hit: Hit,
    },
    Polygon {
        pts: Vec<[f32; 2]>,
        fill: Option<Color>,
        stroke: Option<Stroke>,
        hit: Hit,
    },
    Circle {
        c: [f32; 2],
        r: f32,
        fill: Option<Color>,
        stroke: Option<Stroke>,
        hit: Hit,
    },
    Text {
        pos: [f32; 2],
        text: String,
        size: f32,
        color: Color,
        h: HAlign,
        v: VAlign,
        mono: bool,
        /// Rotated -90 degrees (vertical axis titles).
        vertical: bool,
    },
    ClipPush(Rect),
    ClipPop,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub width: f32,
    pub height: f32,
    pub background: Color,
    pub prims: Vec<Prim>,
}

impl Scene {
    pub fn new(width: f32, height: f32, background: Color) -> Self {
        Scene {
            width,
            height,
            background,
            prims: Vec::new(),
        }
    }

    pub fn rect(&mut self, r: Rect, fill: Option<Color>, stroke: Option<Stroke>, hit: Hit) {
        self.prims.push(Prim::Rect {
            r,
            fill,
            stroke,
            radius: 0.0,
            hit,
        });
    }

    pub fn fill(&mut self, r: Rect, c: Color) {
        self.rect(r, Some(c), None, Hit::None);
    }

    pub fn line(&mut self, pts: Vec<[f32; 2]>, stroke: Stroke) {
        self.prims.push(Prim::Line {
            pts,
            stroke,
            hit: Hit::None,
        });
    }

    pub fn seg(&mut self, a: [f32; 2], b: [f32; 2], stroke: Stroke) {
        self.line(vec![a, b], stroke);
    }

    pub fn circle(
        &mut self,
        c: [f32; 2],
        r: f32,
        fill: Option<Color>,
        stroke: Option<Stroke>,
        hit: Hit,
    ) {
        self.prims.push(Prim::Circle {
            c,
            r,
            fill,
            stroke,
            hit,
        });
    }

    pub fn polygon(
        &mut self,
        pts: Vec<[f32; 2]>,
        fill: Option<Color>,
        stroke: Option<Stroke>,
        hit: Hit,
    ) {
        self.prims.push(Prim::Polygon {
            pts,
            fill,
            stroke,
            hit,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn text_full(
        &mut self,
        pos: [f32; 2],
        text: impl Into<String>,
        size: f32,
        color: Color,
        h: HAlign,
        v: VAlign,
        mono: bool,
        vertical: bool,
    ) {
        self.prims.push(Prim::Text {
            pos,
            text: text.into(),
            size,
            color,
            h,
            v,
            mono,
            vertical,
        });
    }

    pub fn text(
        &mut self,
        pos: [f32; 2],
        text: impl Into<String>,
        size: f32,
        color: Color,
        h: HAlign,
        v: VAlign,
    ) {
        self.text_full(pos, text, size, color, h, v, false, false);
    }

    pub fn clip(&mut self, r: Rect) {
        self.prims.push(Prim::ClipPush(r));
    }

    pub fn unclip(&mut self) {
        self.prims.push(Prim::ClipPop);
    }

    /// Diagonal hatch over `r` (idle / no-data marking, never color alone, 05 §4.4).
    pub fn hatch(&mut self, r: Rect, spacing: f32, stroke: Stroke) {
        if r.w <= 0.0 || r.h <= 0.0 || spacing <= 0.0 {
            return;
        }
        // Lines x - y = c for c from -h to w, clipped to the rect.
        let mut c = -r.h + (r.h % spacing);
        while c < r.w {
            let (x0, y0) = if c >= 0.0 { (c, 0.0) } else { (0.0, -c) };
            let len = (r.w - x0).min(r.h - y0);
            if len > 0.0 {
                self.seg(
                    [r.x + x0, r.y + y0],
                    [r.x + x0 + len, r.y + y0 + len],
                    stroke,
                );
            }
            c += spacing;
        }
    }

    /// Topmost primitive with a hit id containing the point.
    pub fn hit_test(&self, x: f32, y: f32) -> Hit {
        let mut clips: Vec<Rect> = vec![];
        let mut found = Hit::None;
        for p in &self.prims {
            match p {
                Prim::ClipPush(r) => clips.push(*r),
                Prim::ClipPop => {
                    clips.pop();
                }
                _ if clips.last().is_some_and(|c| !c.contains(x, y)) => {}
                Prim::Rect { r, hit, .. } if *hit != Hit::None && r.contains(x, y) => found = *hit,
                Prim::Circle { c, r, hit, .. }
                    if *hit != Hit::None && (c[0] - x).hypot(c[1] - y) <= r.max(3.0) =>
                {
                    found = *hit
                }
                Prim::Polygon { pts, hit, .. } if *hit != Hit::None && point_in_poly(pts, x, y) => {
                    found = *hit
                }
                Prim::Line { pts, hit, stroke }
                    if *hit != Hit::None && near_polyline(pts, x, y, stroke.width.max(4.0)) =>
                {
                    found = *hit
                }
                _ => {}
            }
        }
        found
    }
}

fn point_in_poly(pts: &[[f32; 2]], x: f32, y: f32) -> bool {
    let mut inside = false;
    let n = pts.len();
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + n - 1) % n]);
        if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
    }
    inside
}

fn near_polyline(pts: &[[f32; 2]], x: f32, y: f32, tol: f32) -> bool {
    pts.windows(2).any(|w| {
        let ([ax, ay], [bx, by]) = (w[0], w[1]);
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 {
            (((x - ax) * dx + (y - ay) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (ax + t * dx - x).hypot(ay + t * dy - y) <= tol
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_test_respects_order_and_clip() {
        let mut s = Scene::new(100.0, 100.0, Color::rgb(255, 255, 255));
        s.rect(
            Rect::new(0.0, 0.0, 50.0, 50.0),
            None,
            None,
            Hit::Resource(1),
        );
        s.rect(Rect::new(10.0, 10.0, 10.0, 10.0), None, None, Hit::Op(2));
        s.clip(Rect::new(60.0, 60.0, 10.0, 10.0));
        s.rect(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            None,
            None,
            Hit::Design(3),
        );
        s.unclip();
        assert_eq!(s.hit_test(15.0, 15.0), Hit::Op(2));
        assert_eq!(s.hit_test(40.0, 40.0), Hit::Resource(1));
        assert_eq!(s.hit_test(65.0, 65.0), Hit::Design(3));
        assert_eq!(s.hit_test(90.0, 90.0), Hit::None);
    }
}
