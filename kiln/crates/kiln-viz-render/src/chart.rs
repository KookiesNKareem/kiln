//! Shared chart pieces: unit formatting (05 §4.4), axes with linear/log ticks, legends, titles.

use crate::scene::{Color, HAlign, Rect, Scene, Stroke, VAlign};
use crate::theme::{ColorMap, Theme};

fn sig3(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    let s = if a >= 100.0 {
        format!("{x:.0}")
    } else if a >= 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.2}")
    };
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

fn scaled(x: f64, units: &[(f64, &str)]) -> String {
    let a = x.abs();
    for &(k, u) in units {
        if a >= k * 0.9995 {
            return format!("{} {u}", sig3(x / k));
        }
    }
    let (k, u) = units[units.len() - 1];
    format!("{} {u}", sig3(x / k))
}

pub fn fmt_time(s: f64) -> String {
    scaled(
        s,
        &[
            (1.0, "s"),
            (1e-3, "ms"),
            (1e-6, "us"),
            (1e-9, "ns"),
            (1e-12, "ps"),
        ],
    )
}

pub fn fmt_bytes(b: f64) -> String {
    scaled(
        b,
        &[
            (1e12, "TB"),
            (1e9, "GB"),
            (1e6, "MB"),
            (1e3, "KB"),
            (1.0, "B"),
        ],
    )
}

pub fn fmt_bw(b: f64) -> String {
    scaled(
        b,
        &[
            (1e12, "TB/s"),
            (1e9, "GB/s"),
            (1e6, "MB/s"),
            (1e3, "KB/s"),
            (1.0, "B/s"),
        ],
    )
}

pub fn fmt_flops(f: f64) -> String {
    scaled(
        f,
        &[
            (1e15, "PFLOP/s"),
            (1e12, "TFLOP/s"),
            (1e9, "GFLOP/s"),
            (1e6, "MFLOP/s"),
            (1.0, "FLOP/s"),
        ],
    )
}

pub fn fmt_energy(j: f64) -> String {
    scaled(
        j,
        &[
            (1.0, "J"),
            (1e-3, "mJ"),
            (1e-6, "uJ"),
            (1e-9, "nJ"),
            (1e-12, "pJ"),
        ],
    )
}

pub fn fmt_power(w: f64) -> String {
    scaled(w, &[(1e3, "kW"), (1.0, "W"), (1e-3, "mW")])
}

pub fn fmt_num(x: f64) -> String {
    let a = x.abs();
    if a >= 1e6 || (a > 0.0 && a < 1e-2) {
        format!("{x:.2e}")
    } else {
        sig3(x)
    }
}

pub fn fmt_pct(x: f64) -> String {
    format!("{:.0}%", 100.0 * x)
}

#[derive(Clone, Copy, Debug)]
pub struct Axis {
    pub min: f64,
    pub max: f64,
    pub log: bool,
    pub px0: f32,
    pub px1: f32,
}

impl Axis {
    pub fn new(min: f64, max: f64, log: bool, px0: f32, px1: f32) -> Axis {
        let (mut min, mut max) = (min, max);
        if log {
            min = min.max(1e-300);
            max = max.max(min * 10.0);
        } else if (max - min).abs() < 1e-300 {
            max = min + 1.0;
        }
        Axis {
            min,
            max,
            log,
            px0,
            px1,
        }
    }

    /// Log axis padded to whole decades around the data.
    pub fn log_fit(lo: f64, hi: f64, px0: f32, px1: f32) -> Axis {
        let lo = lo.max(1e-300);
        let hi = hi.max(lo);
        Axis::new(
            10f64.powf(lo.log10().floor()),
            10f64.powf(hi.log10().ceil().max(lo.log10().floor() + 1.0)),
            true,
            px0,
            px1,
        )
    }

    pub fn map(&self, v: f64) -> f32 {
        let t = if self.log {
            (v.max(1e-300).log10() - self.min.log10()) / (self.max.log10() - self.min.log10())
        } else {
            (v - self.min) / (self.max - self.min)
        };
        self.px0 + (self.px1 - self.px0) * t as f32
    }

    pub fn ticks(&self) -> Vec<f64> {
        if self.log {
            let (a, b) = (
                self.min.log10().floor() as i32,
                self.max.log10().ceil() as i32,
            );
            let step = ((b - a) as f64 / 8.0).ceil().max(1.0) as i32;
            (a..=b)
                .step_by(step as usize)
                .map(|e| 10f64.powi(e))
                .filter(|v| *v >= self.min * 0.999 && *v <= self.max * 1.001)
                .collect()
        } else {
            let span = self.max - self.min;
            let raw = span / 6.0;
            let mag = 10f64.powf(raw.log10().floor());
            let step = [1.0, 2.0, 2.5, 5.0, 10.0]
                .iter()
                .map(|m| m * mag)
                .find(|s| span / s <= 7.0)
                .unwrap_or(mag * 10.0);
            let mut v = (self.min / step).ceil() * step;
            let mut out = vec![];
            while v <= self.max + step * 1e-9 {
                out.push(if v.abs() < step * 1e-9 { 0.0 } else { v });
                v += step;
            }
            out
        }
    }
}

pub fn title(s: &mut Scene, th: &Theme, text: &str, sub: &str) {
    s.text([16.0, 12.0], text, 18.0, th.fg, HAlign::Left, VAlign::Top);
    if !sub.is_empty() {
        s.text([16.0, 36.0], sub, 12.0, th.muted, HAlign::Left, VAlign::Top);
    }
}

/// Grid, ticks and axis titles around `plot`.
#[allow(clippy::too_many_arguments)]
pub fn axes(
    s: &mut Scene,
    th: &Theme,
    plot: Rect,
    x: &Axis,
    y: &Axis,
    xlabel: &str,
    ylabel: &str,
    xfmt: &dyn Fn(f64) -> String,
    yfmt: &dyn Fn(f64) -> String,
) {
    let grid = Stroke::solid(1.0, th.grid);
    for v in x.ticks() {
        let px = x.map(v);
        s.seg([px, plot.y], [px, plot.bottom()], grid);
        s.text(
            [px, plot.bottom() + 6.0],
            xfmt(v),
            11.0,
            th.muted,
            HAlign::Center,
            VAlign::Top,
        );
    }
    for v in y.ticks() {
        let py = y.map(v);
        s.seg([plot.x, py], [plot.right(), py], grid);
        s.text(
            [plot.x - 6.0, py],
            yfmt(v),
            11.0,
            th.muted,
            HAlign::Right,
            VAlign::Middle,
        );
    }
    s.rect(
        plot,
        None,
        Some(Stroke::solid(1.0, th.outline)),
        crate::scene::Hit::None,
    );
    s.text(
        [plot.x + plot.w / 2.0, plot.bottom() + 24.0],
        xlabel,
        12.0,
        th.fg,
        HAlign::Center,
        VAlign::Top,
    );
    s.text_full(
        [plot.x - 62.0, plot.y + plot.h / 2.0],
        ylabel,
        12.0,
        th.fg,
        HAlign::Center,
        VAlign::Bottom,
        false,
        true,
    );
}

/// Vertical gradient legend; `fmt_t` labels a normalized position t in [0, 1].
pub fn gradient_legend(
    s: &mut Scene,
    th: &Theme,
    r: Rect,
    cmap: &dyn Fn(f64) -> Color,
    label: &str,
    fmt_t: &dyn Fn(f64) -> String,
) {
    s.text(
        [r.x, r.y - 6.0],
        label,
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Bottom,
    );
    let n = 32;
    for i in 0..n {
        let t = 1.0 - (i as f64 + 0.5) / n as f64;
        let h = r.h / n as f32;
        s.fill(Rect::new(r.x, r.y + i as f32 * h, r.w, h + 0.5), cmap(t));
    }
    s.rect(
        r,
        None,
        Some(Stroke::solid(1.0, th.outline)),
        crate::scene::Hit::None,
    );
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let y = r.bottom() - (t as f32) * r.h;
        s.seg(
            [r.right(), y],
            [r.right() + 4.0, y],
            Stroke::solid(1.0, th.outline),
        );
        s.text(
            [r.right() + 6.0, y],
            fmt_t(t),
            10.0,
            th.muted,
            HAlign::Left,
            VAlign::Middle,
        );
    }
}

pub fn cmap_fn(m: ColorMap) -> impl Fn(f64) -> Color {
    move |t| m.at(t)
}

/// Swatch list; returns the y below the last item.
pub fn swatches(s: &mut Scene, th: &Theme, x: f32, y: f32, items: &[(Color, String)]) -> f32 {
    let mut yy = y;
    for (c, label) in items {
        s.rect(
            Rect::new(x, yy, 12.0, 12.0),
            Some(*c),
            Some(Stroke::solid(0.5, th.outline)),
            crate::scene::Hit::None,
        );
        s.text(
            [x + 18.0, yy + 6.0],
            label.clone(),
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Middle,
        );
        yy += 18.0;
    }
    yy
}

/// "Not available" banner in place of an empty pane (05 §3.3).
pub fn banner(s: &mut Scene, th: &Theme, r: Rect, msg: &str, hint: &str) {
    s.rect(
        r,
        Some(th.panel),
        Some(Stroke::dashed(1.0, th.outline, 4.0)),
        crate::scene::Hit::None,
    );
    s.text(
        [r.x + r.w / 2.0, r.y + r.h / 2.0 - 10.0],
        msg,
        14.0,
        th.fg,
        HAlign::Center,
        VAlign::Middle,
    );
    if !hint.is_empty() {
        s.text_full(
            [r.x + r.w / 2.0, r.y + r.h / 2.0 + 12.0],
            hint,
            12.0,
            th.muted,
            HAlign::Center,
            VAlign::Middle,
            true,
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units() {
        assert_eq!(fmt_time(12.371e-3), "12.4 ms");
        assert_eq!(fmt_time(4.2e-7), "420 ns");
        assert_eq!(fmt_bytes(1.5e9), "1.5 GB");
        assert_eq!(fmt_flops(3.12e14), "312 TFLOP/s");
        assert_eq!(fmt_bw(1.5552e12), "1.56 TB/s");
        assert_eq!(fmt_energy(0.0), "0 pJ");
    }

    #[test]
    fn ticks() {
        let a = Axis::log_fit(3e10, 2e14, 0.0, 100.0);
        assert_eq!(a.ticks(), vec![1e10, 1e11, 1e12, 1e13, 1e14, 1e15]);
        let l = Axis::new(0.0, 1.0, false, 0.0, 100.0);
        assert_eq!(l.ticks().len(), 6);
    }
}
