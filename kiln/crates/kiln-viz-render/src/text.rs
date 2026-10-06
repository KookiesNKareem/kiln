//! Bundled fonts (05 §7.1: no system font lookup): egui's Ubuntu-Light (sans) and Hack (mono), so headless
//! images and the interactive app use the same faces. Layout is simple left-to-right with kerning.

use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, GlyphId, OutlineCurve, PxScale, ScaleFont};

pub struct Fonts {
    pub sans: FontRef<'static>,
    pub mono: FontRef<'static>,
}

pub fn fonts() -> &'static Fonts {
    static F: OnceLock<Fonts> = OnceLock::new();
    F.get_or_init(|| Fonts {
        sans: FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT)
            .expect("bundled sans font"),
        mono: FontRef::try_from_slice(epaint_default_fonts::HACK_REGULAR)
            .expect("bundled mono font"),
    })
}

fn face(mono: bool) -> &'static FontRef<'static> {
    let f = fonts();
    if mono { &f.mono } else { &f.sans }
}

/// Text size in px is the font's ascent-to-descent height, as egui uses it.
pub fn measure(text: &str, size: f32, mono: bool) -> f32 {
    let f = face(mono).as_scaled(PxScale::from(size));
    let mut w = 0.0;
    let mut prev: Option<GlyphId> = None;
    for ch in text.chars() {
        let g = f.glyph_id(ch);
        if let Some(p) = prev {
            w += f.kern(p, g);
        }
        w += f.h_advance(g);
        prev = Some(g);
    }
    w
}

/// `(ascent, descent)` in px; descent is negative.
pub fn metrics(size: f32, mono: bool) -> (f32, f32) {
    let f = face(mono).as_scaled(PxScale::from(size));
    (f.ascent(), f.descent())
}

/// Truncates with an ellipsis to fit `max_w`.
pub fn fit(text: &str, size: f32, mono: bool, max_w: f32) -> Option<String> {
    if measure(text, size, mono) <= max_w {
        return Some(text.to_string());
    }
    let chars: Vec<char> = text.chars().collect();
    for n in (1..chars.len()).rev() {
        let s: String = chars[..n].iter().collect::<String>() + "\u{2026}";
        if measure(&s, size, mono) <= max_w {
            return Some(s);
        }
    }
    None
}

pub enum PathOp {
    Move(f32, f32),
    Line(f32, f32),
    Quad(f32, f32, f32, f32),
    Cubic(f32, f32, f32, f32, f32, f32),
    Close,
}

/// Glyph outlines of `text` with the baseline origin at `(x, y)`, in px, y down.
pub fn outline(text: &str, size: f32, mono: bool, x: f32, y: f32) -> Vec<PathOp> {
    let font = face(mono);
    let f = font.as_scaled(PxScale::from(size));
    let (sx, sy) = (f.h_scale_factor(), f.v_scale_factor());
    let mut ops = Vec::new();
    let mut pen = x;
    let mut prev: Option<GlyphId> = None;
    for ch in text.chars() {
        let g = f.glyph_id(ch);
        if let Some(p) = prev {
            pen += f.kern(p, g);
        }
        if let Some(o) = font.outline(g) {
            let tx = |p: ab_glyph::Point| (pen + p.x * sx, y - p.y * sy);
            let mut last: Option<(f32, f32)> = None;
            for c in &o.curves {
                let (start, end) = match c {
                    OutlineCurve::Line(a, b) => (*a, *b),
                    OutlineCurve::Quad(a, _, b) => (*a, *b),
                    OutlineCurve::Cubic(a, _, _, b) => (*a, *b),
                };
                let s = tx(start);
                if last.is_none_or(|l| (l.0 - s.0).abs() > 1e-4 || (l.1 - s.1).abs() > 1e-4) {
                    if last.is_some() {
                        ops.push(PathOp::Close);
                    }
                    ops.push(PathOp::Move(s.0, s.1));
                }
                match c {
                    OutlineCurve::Line(_, b) => {
                        let b = tx(*b);
                        ops.push(PathOp::Line(b.0, b.1));
                    }
                    OutlineCurve::Quad(_, c1, b) => {
                        let (c1, b) = (tx(*c1), tx(*b));
                        ops.push(PathOp::Quad(c1.0, c1.1, b.0, b.1));
                    }
                    OutlineCurve::Cubic(_, c1, c2, b) => {
                        let (c1, c2, b) = (tx(*c1), tx(*c2), tx(*b));
                        ops.push(PathOp::Cubic(c1.0, c1.1, c2.0, c2.1, b.0, b.1));
                    }
                }
                last = Some(tx(end));
            }
            if last.is_some() {
                ops.push(PathOp::Close);
            }
        }
        pen += f.h_advance(g);
        prev = Some(g);
    }
    ops
}

/// Baseline origin of a text primitive anchored at `pos` (in the unrotated frame).
pub fn origin(
    pos: [f32; 2],
    text: &str,
    size: f32,
    mono: bool,
    h: crate::scene::HAlign,
    v: crate::scene::VAlign,
) -> (f32, f32) {
    use crate::scene::{HAlign, VAlign};
    let w = measure(text, size, mono);
    let (asc, desc) = metrics(size, mono);
    let x = match h {
        HAlign::Left => pos[0],
        HAlign::Center => pos[0] - w / 2.0,
        HAlign::Right => pos[0] - w,
    };
    let y = match v {
        VAlign::Top => pos[1] + asc,
        VAlign::Middle => pos[1] + (asc + desc) / 2.0,
        VAlign::Baseline => pos[1],
        VAlign::Bottom => pos[1] + desc,
    };
    (x, y)
}

/// Greedy word wrap to `max_w`.
pub fn wrap(text: &str, size: f32, mono: bool, max_w: f32) -> Vec<String> {
    let mut lines = vec![];
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let cand = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };
        if measure(&cand, size, mono) <= max_w || cur.is_empty() {
            cur = cand;
        } else {
            lines.push(std::mem::replace(&mut cur, word.to_string()));
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}
