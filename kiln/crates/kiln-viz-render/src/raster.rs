//! Scene -> PNG with tiny-skia (CPU, fixed anti-aliasing): byte-identical output for identical scenes.

use tiny_skia::{
    FillRule, LineCap, LineJoin, Mask, Paint, PathBuilder, Pixmap, StrokeDash, Transform,
};

use crate::scene::{Color, Prim, Rect, Scene, Stroke};
use crate::text::{self, PathOp};

fn paint(c: Color) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color_rgba8(c.r, c.g, c.b, c.a);
    p.anti_alias = true;
    p
}

fn sk_stroke(s: &Stroke) -> tiny_skia::Stroke {
    tiny_skia::Stroke {
        width: s.width,
        line_cap: LineCap::Butt,
        line_join: LineJoin::Miter,
        dash: (s.dash > 0.0)
            .then(|| StrokeDash::new(vec![s.dash, s.dash], 0.0))
            .flatten(),
        ..Default::default()
    }
}

fn intersect(a: Rect, b: Rect) -> Rect {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    Rect::new(
        x,
        y,
        (a.right().min(b.right()) - x).max(0.0),
        (a.bottom().min(b.bottom()) - y).max(0.0),
    )
}

/// Renders the scene at `scale` device pixels per logical pixel.
pub fn to_pixmap(s: &Scene, scale: f32) -> Pixmap {
    let (w, h) = (
        (s.width * scale).ceil().max(1.0) as u32,
        (s.height * scale).ceil().max(1.0) as u32,
    );
    let mut pm = Pixmap::new(w, h).expect("non-zero pixmap");
    let bg = s.background;
    pm.fill(tiny_skia::Color::from_rgba8(bg.r, bg.g, bg.b, bg.a));
    let tf = Transform::from_scale(scale, scale);
    let mut clips: Vec<Rect> = vec![];
    let mut mask: Option<Mask> = None;
    let make_mask = |r: Rect| -> Option<Mask> {
        let mut m = Mask::new(w, h)?;
        let rr = tiny_skia::Rect::from_xywh(r.x, r.y, r.w.max(0.01), r.h.max(0.01))?;
        m.fill_path(&PathBuilder::from_rect(rr), FillRule::Winding, false, tf);
        Some(m)
    };
    for p in &s.prims {
        let m = mask.as_ref();
        match p {
            Prim::ClipPush(r) => {
                let r = clips.last().map_or(*r, |c| intersect(*c, *r));
                clips.push(r);
                mask = make_mask(r);
            }
            Prim::ClipPop => {
                clips.pop();
                mask = clips.last().and_then(|r| make_mask(*r));
            }
            Prim::Rect {
                r,
                fill,
                stroke,
                radius,
                ..
            } => {
                let Some(path) = rect_path(*r, *radius) else {
                    continue;
                };
                if let Some(f) = fill {
                    pm.fill_path(&path, &paint(*f), FillRule::Winding, tf, m);
                }
                if let Some(st) = stroke {
                    pm.stroke_path(&path, &paint(st.color), &sk_stroke(st), tf, m);
                }
            }
            Prim::Line { pts, stroke, .. } => {
                let mut pb = PathBuilder::new();
                for (i, q) in pts.iter().enumerate() {
                    if i == 0 {
                        pb.move_to(q[0], q[1])
                    } else {
                        pb.line_to(q[0], q[1])
                    }
                }
                if let Some(path) = pb.finish() {
                    pm.stroke_path(&path, &paint(stroke.color), &sk_stroke(stroke), tf, m);
                }
            }
            Prim::Polygon {
                pts, fill, stroke, ..
            } => {
                let mut pb = PathBuilder::new();
                for (i, q) in pts.iter().enumerate() {
                    if i == 0 {
                        pb.move_to(q[0], q[1])
                    } else {
                        pb.line_to(q[0], q[1])
                    }
                }
                pb.close();
                let Some(path) = pb.finish() else { continue };
                if let Some(f) = fill {
                    pm.fill_path(&path, &paint(*f), FillRule::Winding, tf, m);
                }
                if let Some(st) = stroke {
                    pm.stroke_path(&path, &paint(st.color), &sk_stroke(st), tf, m);
                }
            }
            Prim::Circle {
                c, r, fill, stroke, ..
            } => {
                let Some(path) = PathBuilder::from_circle(c[0], c[1], r.max(0.01)) else {
                    continue;
                };
                if let Some(f) = fill {
                    pm.fill_path(&path, &paint(*f), FillRule::Winding, tf, m);
                }
                if let Some(st) = stroke {
                    pm.stroke_path(&path, &paint(st.color), &sk_stroke(st), tf, m);
                }
            }
            Prim::Text {
                pos,
                text: t,
                size,
                color,
                h,
                v,
                mono,
                vertical,
            } => {
                let (x, y) = text::origin(*pos, t, *size, *mono, *h, *v);
                let ops = text::outline(t, *size, *mono, x, y);
                let mut pb = PathBuilder::new();
                for op in ops {
                    match op {
                        PathOp::Move(x, y) => pb.move_to(x, y),
                        PathOp::Line(x, y) => pb.line_to(x, y),
                        PathOp::Quad(a, b, c, d) => pb.quad_to(a, b, c, d),
                        PathOp::Cubic(a, b, c, d, e, f) => pb.cubic_to(a, b, c, d, e, f),
                        PathOp::Close => pb.close(),
                    }
                }
                let Some(path) = pb.finish() else { continue };
                let t = if *vertical {
                    tf.pre_concat(Transform::from_rotate_at(-90.0, pos[0], pos[1]))
                } else {
                    tf
                };
                pm.fill_path(&path, &paint(*color), FillRule::Winding, t, m);
            }
        }
    }
    pm
}

fn rect_path(r: Rect, radius: f32) -> Option<tiny_skia::Path> {
    if radius <= 0.0 {
        return Some(PathBuilder::from_rect(tiny_skia::Rect::from_xywh(
            r.x,
            r.y,
            r.w.max(0.0),
            r.h.max(0.0),
        )?));
    }
    let k = radius.min(r.w / 2.0).min(r.h / 2.0);
    let mut pb = PathBuilder::new();
    pb.move_to(r.x + k, r.y);
    pb.line_to(r.right() - k, r.y);
    pb.quad_to(r.right(), r.y, r.right(), r.y + k);
    pb.line_to(r.right(), r.bottom() - k);
    pb.quad_to(r.right(), r.bottom(), r.right() - k, r.bottom());
    pb.line_to(r.x + k, r.bottom());
    pb.quad_to(r.x, r.bottom(), r.x, r.bottom() - k);
    pb.line_to(r.x, r.y + k);
    pb.quad_to(r.x, r.y, r.x + k, r.y);
    pb.close();
    pb.finish()
}

pub fn to_png(s: &Scene, scale: f32) -> Vec<u8> {
    to_pixmap(s, scale).encode_png().expect("png encodes")
}
