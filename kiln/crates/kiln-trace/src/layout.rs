//! Deterministic hierarchy layout used as the floorplan until kiln-phys places blocks (04): every placed
//! container's children are packed into its outline by a squarified treemap weighted by a nominal size per
//! kind. Not physical: sizes are relative and the manifest marks it `floorplan_source = "unplaced"`.

use crate::trace::{FloorplanRow, NONE_U32, ResourceRow, Trace};

pub const UNPLACED: &str = "unplaced";

/// Nominal relative area of a resource row by kind; 0 = not drawn.
fn leaf_weight(kind: &str, capacity_b: Option<f64>) -> f64 {
    match kind {
        "unit_matrix" | "unit_cim" => 4.0,
        "unit_vector" | "nmp_unit" => 2.0,
        "unit_scalar" | "unit_special" | "block" => 1.0,
        "memory" => {
            let kib = capacity_b.unwrap_or(65536.0) / 1024.0;
            (kib.log2() - 3.0).clamp(0.5, 24.0)
        }
        "mem_stack" => 160.0,
        "router" | "port" => 0.5,
        _ => 0.0,
    }
}

fn is_container(kind: &str) -> bool {
    matches!(
        kind,
        "system" | "host" | "board" | "package" | "die" | "cluster" | "switch"
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// Floorplan rows for every drawable resource (containers, units, memories, blocks, routers, ports).
pub fn unplaced_floorplan(t: &Trace) -> Vec<FloorplanRow> {
    let n = t.resources.len();
    let kinds: Vec<&str> = t.resources.iter().map(|r| t.resource_kind(r)).collect();
    let children = t.children();
    let mut weight = vec![0.0; n];
    // Children have larger indices than parents only when sorted by path, which the builder guarantees;
    // walk in reverse post-order via an explicit order to be safe.
    let order = postorder(&t.resources, &children);
    for &i in &order {
        let k = kinds[i];
        weight[i] = if is_container(k) {
            children[i].iter().map(|&c| weight[c as usize]).sum()
        } else {
            leaf_weight(k, t.resources[i].capacity_b)
        };
    }
    let mut rects: Vec<Option<Rect>> = vec![None; n];
    let roots: Vec<usize> = (0..n)
        .filter(|&i| t.resources[i].parent.is_none() && weight[i] > 0.0)
        .collect();
    let total: f64 = roots.iter().map(|&i| weight[i]).sum();
    if total <= 0.0 {
        return vec![];
    }
    // 1 weight unit ~ 1 mm^2 so labels and scale bars read sensibly.
    let side = total.sqrt() * 1000.0;
    let root = Rect {
        x: 0.0,
        y: 0.0,
        w: side,
        h: side,
    };
    let mut items: Vec<(usize, f64)> = roots.iter().map(|&i| (i, weight[i])).collect();
    squarify(&mut items, root, &mut rects);
    let mut stack: Vec<usize> = roots.clone();
    while let Some(i) = stack.pop() {
        let (Some(r), true) = (rects[i], is_container(kinds[i])) else {
            continue;
        };
        let pad = 0.02 * r.w.min(r.h);
        let head = (0.06 * r.h).min(0.25 * r.w).max(pad);
        let inner = Rect {
            x: r.x + pad,
            y: r.y + head,
            w: (r.w - 2.0 * pad).max(0.0),
            h: (r.h - head - pad).max(0.0),
        };
        let mut items: Vec<(usize, f64)> = children[i]
            .iter()
            .map(|&c| (c as usize, weight[c as usize]))
            .filter(|x| x.1 > 0.0)
            .collect();
        squarify(&mut items, inner, &mut rects);
        stack.extend(items.iter().map(|x| x.0));
    }
    let die = nearest(t, &kinds, "die");
    let mut out: Vec<FloorplanRow> = (0..n)
        .filter_map(|i| {
            let r = rects[i]?;
            Some(FloorplanRow {
                resource: i as u32,
                die: die[i],
                layer: 0,
                x_um: r.x,
                y_um: r.y,
                w_um: r.w,
                h_um: r.h,
                poly: None,
                rotation: 0,
                source: 0,
                block: block_of_kind(kinds[i]),
                area_um2: None,
                leak_w: None,
            })
        })
        .collect();
    out.sort_by_key(|r| r.resource);
    out
}

/// `floorplan.block` code of a resource kind (default enum order).
pub fn block_of_kind(kind: &str) -> u8 {
    match kind {
        k if k.starts_with("unit_") || k == "nmp_unit" => 1,
        "memory" | "bank" => 2,
        "router" | "network" | "channel" => 3,
        "port" => 4,
        "mem_stack" => 5,
        "sequencer" | "dma" => 6,
        _ => 0,
    }
}

fn postorder(rows: &[ResourceRow], children: &[Vec<u32>]) -> Vec<usize> {
    let mut out = Vec::with_capacity(rows.len());
    let mut stack: Vec<(usize, bool)> = (0..rows.len())
        .rev()
        .filter(|&i| rows[i].parent.is_none())
        .map(|i| (i, false))
        .collect();
    while let Some((i, done)) = stack.pop() {
        if done {
            out.push(i);
        } else {
            stack.push((i, true));
            stack.extend(children[i].iter().rev().map(|&c| (c as usize, false)));
        }
    }
    out
}

/// Index of the nearest ancestor-or-self of the given kind, per row.
pub fn nearest(t: &Trace, kinds: &[&str], kind: &str) -> Vec<u32> {
    (0..t.resources.len())
        .map(|i| {
            let mut cur = Some(i as u32);
            while let Some(c) = cur {
                if kinds[c as usize] == kind {
                    return c;
                }
                cur = t.resources[c as usize].parent;
            }
            NONE_U32
        })
        .collect()
}

/// Squarified treemap (Bruls, Huizing, van Wijk 2000). Items keep their given order (stable, so arrays stay
/// in index order); `rects[i]` is filled for every item.
fn squarify(items: &mut [(usize, f64)], r: Rect, rects: &mut [Option<Rect>]) {
    let total: f64 = items.iter().map(|x| x.1).sum();
    if items.is_empty() || total <= 0.0 || r.w <= 0.0 || r.h <= 0.0 {
        return;
    }
    let scale = r.w * r.h / total;
    let areas: Vec<f64> = items.iter().map(|x| x.1 * scale).collect();
    let mut rest = r;
    let mut start = 0;
    while start < items.len() {
        let short = rest.w.min(rest.h);
        let mut end = start + 1;
        let mut best = worst(&areas[start..end], short);
        while end < items.len() {
            let w = worst(&areas[start..=end], short);
            if w > best {
                break;
            }
            best = w;
            end += 1;
        }
        let row_area: f64 = areas[start..end].iter().sum();
        let horizontal = rest.w >= rest.h;
        let thick = if short > 0.0 { row_area / short } else { 0.0 };
        let mut off = 0.0;
        for k in start..end {
            let len = if thick > 0.0 { areas[k] / thick } else { 0.0 };
            rects[items[k].0] = Some(if horizontal {
                Rect {
                    x: rest.x,
                    y: rest.y + off,
                    w: thick,
                    h: len,
                }
            } else {
                Rect {
                    x: rest.x + off,
                    y: rest.y,
                    w: len,
                    h: thick,
                }
            });
            off += len;
        }
        rest = if horizontal {
            Rect {
                x: rest.x + thick,
                y: rest.y,
                w: (rest.w - thick).max(0.0),
                h: rest.h,
            }
        } else {
            Rect {
                x: rest.x,
                y: rest.y + thick,
                w: rest.w,
                h: (rest.h - thick).max(0.0),
            }
        };
        start = end;
    }
}

fn worst(row: &[f64], short: f64) -> f64 {
    let s: f64 = row.iter().sum();
    let (mx, mn) = row
        .iter()
        .fold((f64::MIN, f64::MAX), |(a, b), &x| (a.max(x), b.min(x)));
    if s <= 0.0 || mn <= 0.0 {
        return f64::INFINITY;
    }
    let s2 = s * s;
    let w2 = short * short;
    (w2 * mx / s2).max(s2 / (w2 * mn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squarify_tiles_exactly() {
        let mut items: Vec<(usize, f64)> = (0..7).map(|i| (i, 1.0 + i as f64)).collect();
        let mut rects = vec![None; 7];
        squarify(
            &mut items,
            Rect {
                x: 0.0,
                y: 0.0,
                w: 6.0,
                h: 4.0,
            },
            &mut rects,
        );
        let area: f64 = rects.iter().map(|r| r.unwrap().w * r.unwrap().h).sum();
        assert!((area - 24.0).abs() < 1e-9);
        for (i, r) in rects.iter().enumerate() {
            let r = r.unwrap();
            assert!((r.w * r.h - 24.0 * (1.0 + i as f64) / 28.0).abs() < 1e-9);
            assert!(
                r.x >= -1e-9 && r.y >= -1e-9 && r.x + r.w <= 6.0 + 1e-9 && r.y + r.h <= 4.0 + 1e-9
            );
        }
    }
}
