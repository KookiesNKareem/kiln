//! Scene -> SVG text (diffable: fixed float formatting, primitives in scene order, no timestamps).

use std::fmt::Write as _;

use crate::scene::{Color, HAlign, Prim, Scene, Stroke, VAlign};

/// Up to two decimals, trailing zeros trimmed, `-0` normalized.
pub fn num(x: f32) -> String {
    let mut s = format!("{:.2}", x);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" { "0".into() } else { s }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn fill_attr(c: Option<Color>) -> String {
    match c {
        None => " fill=\"none\"".into(),
        Some(c) if c.a == 255 => format!(" fill=\"{}\"", c.hex()),
        Some(c) => format!(
            " fill=\"{}\" fill-opacity=\"{}\"",
            c.hex(),
            num(f32::from(c.a) / 255.0)
        ),
    }
}

fn stroke_attr(s: Option<&Stroke>) -> String {
    match s {
        None => String::new(),
        Some(s) => {
            let mut a = format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                s.color.hex(),
                num(s.width)
            );
            if s.color.a != 255 {
                write!(
                    a,
                    " stroke-opacity=\"{}\"",
                    num(f32::from(s.color.a) / 255.0)
                )
                .unwrap();
            }
            if s.dash > 0.0 {
                write!(a, " stroke-dasharray=\"{}\"", num(s.dash)).unwrap();
            }
            a
        }
    }
}

fn pts(p: &[[f32; 2]]) -> String {
    p.iter()
        .map(|q| format!("{},{}", num(q[0]), num(q[1])))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn to_svg(s: &Scene) -> String {
    let mut o = String::new();
    writeln!(
        o,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" font-family=\"Ubuntu, 'DejaVu Sans', Arial, sans-serif\">",
        w = num(s.width),
        h = num(s.height)
    )
    .unwrap();
    writeln!(
        o,
        "<rect width=\"100%\" height=\"100%\"{}/>",
        fill_attr(Some(s.background))
    )
    .unwrap();
    let mut clip_id = 0;
    let mut open_groups = 0;
    for p in &s.prims {
        match p {
            Prim::ClipPush(r) => {
                clip_id += 1;
                writeln!(
                    o,
                    "<clipPath id=\"c{clip_id}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath><g clip-path=\"url(#c{clip_id})\">",
                    num(r.x),
                    num(r.y),
                    num(r.w),
                    num(r.h)
                )
                .unwrap();
                open_groups += 1;
            }
            Prim::ClipPop => {
                if open_groups > 0 {
                    o.push_str("</g>\n");
                    open_groups -= 1;
                }
            }
            Prim::Rect {
                r,
                fill,
                stroke,
                radius,
                ..
            } => {
                let rr = if *radius > 0.0 {
                    format!(" rx=\"{}\"", num(*radius))
                } else {
                    String::new()
                };
                writeln!(
                    o,
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"{rr}{}{}/>",
                    num(r.x),
                    num(r.y),
                    num(r.w),
                    num(r.h),
                    fill_attr(*fill),
                    stroke_attr(stroke.as_ref())
                )
                .unwrap();
            }
            Prim::Line { pts: p, stroke, .. } => {
                writeln!(
                    o,
                    "<polyline points=\"{}\" fill=\"none\"{}/>",
                    pts(p),
                    stroke_attr(Some(stroke))
                )
                .unwrap();
            }
            Prim::Polygon {
                pts: p,
                fill,
                stroke,
                ..
            } => {
                writeln!(
                    o,
                    "<polygon points=\"{}\"{}{}/>",
                    pts(p),
                    fill_attr(*fill),
                    stroke_attr(stroke.as_ref())
                )
                .unwrap();
            }
            Prim::Circle {
                c, r, fill, stroke, ..
            } => {
                writeln!(
                    o,
                    "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}{}/>",
                    num(c[0]),
                    num(c[1]),
                    num(*r),
                    fill_attr(*fill),
                    stroke_attr(stroke.as_ref())
                )
                .unwrap();
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
                let anchor = match h {
                    HAlign::Left => "start",
                    HAlign::Center => "middle",
                    HAlign::Right => "end",
                };
                let (asc, desc) = crate::text::metrics(*size, *mono);
                let dy = match v {
                    VAlign::Top => asc,
                    VAlign::Middle => (asc + desc) / 2.0,
                    VAlign::Baseline => 0.0,
                    VAlign::Bottom => desc,
                };
                let rot = if *vertical {
                    format!(" transform=\"rotate(-90 {} {})\"", num(pos[0]), num(pos[1]))
                } else {
                    String::new()
                };
                let fam = if *mono {
                    " font-family=\"Hack, 'DejaVu Sans Mono', monospace\""
                } else {
                    ""
                };
                writeln!(
                    o,
                    "<text x=\"{}\" y=\"{}\" font-size=\"{}\" text-anchor=\"{anchor}\"{}{fam}{rot}>{}</text>",
                    num(pos[0]),
                    num(pos[1] + dy),
                    num(*size),
                    fill_attr(Some(*color)),
                    esc(text)
                )
                .unwrap();
            }
        }
    }
    for _ in 0..open_groups {
        o.push_str("</g>\n");
    }
    o.push_str("</svg>\n");
    o
}
