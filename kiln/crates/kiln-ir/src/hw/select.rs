//! Reference and selector syntax (01 §14.4): parsing, globbing, index sets, and `{i}` interpolation.

use crate::common::Diagnostic;

#[derive(Clone, Debug, PartialEq)]
pub enum Start {
    /// Nearest enclosing scope where the first segment matches.
    Lexical,
    /// `/...` from the system root.
    Root,
    /// `^.` repeated `n` times.
    Up(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Seg {
    AnyDepth,
    Pat { name: String, index: Option<Vec<Axis>> },
}

/// One `;`-separated axis of an index set; empty = `*`.
pub type Axis = Vec<Span>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Span {
    One(u32),
    Range(u32, u32),
    All,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Selector {
    pub start: Start,
    pub segs: Vec<Seg>,
}

fn bad(s: &str, why: &str) -> Diagnostic {
    Diagnostic::error("E-IR-0206", format!("malformed reference/selector {s:?}: {why}"))
        .hint("paths are dotted ids; patterns: name*, name[3], name[1;2], name[0..4], name[0,5], **, ^.name, /abs.path")
}

/// Splits on `.` outside brackets (so `hbm_if[0..3].mc*` has two segments).
pub fn split_segments(s: &str) -> Vec<&str> {
    let mut out = vec![];
    let (mut depth, mut start) = (0i32, 0);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'[' => depth += 1,
            b']' => depth -= 1,
            b'.' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

impl Selector {
    pub fn parse(s: &str) -> Result<Self, Diagnostic> {
        let mut rest = s.trim();
        let mut start = Start::Lexical;
        if let Some(r) = rest.strip_prefix('/') {
            start = Start::Root;
            rest = r;
        } else {
            let mut ups = 0;
            while let Some(r) = rest.strip_prefix("^.") {
                ups += 1;
                rest = r;
            }
            if ups > 0 {
                start = Start::Up(ups);
            }
        }
        if rest.is_empty() {
            return Err(bad(s, "empty path"));
        }
        let segs = split_segments(rest).into_iter().map(|seg| parse_seg(s, seg)).collect::<Result<_, _>>()?;
        Ok(Self { start, segs })
    }
}

fn parse_seg(full: &str, seg: &str) -> Result<Seg, Diagnostic> {
    if seg == "**" {
        return Ok(Seg::AnyDepth);
    }
    if seg.is_empty() {
        return Err(bad(full, "empty segment"));
    }
    let (name, index) = match seg.split_once('[') {
        Some((n, idx)) => {
            let idx = idx.strip_suffix(']').ok_or_else(|| bad(full, "unclosed '['"))?;
            (n, Some(idx.split(';').map(|a| parse_axis(full, a)).collect::<Result<_, _>>()?))
        }
        None => (seg, None),
    };
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'*')) {
        return Err(bad(full, &format!("invalid segment {seg:?}")));
    }
    Ok(Seg::Pat { name: name.to_owned(), index })
}

fn parse_axis(full: &str, a: &str) -> Result<Axis, Diagnostic> {
    let a = a.trim();
    if a == "*" {
        return Ok(vec![Span::All]);
    }
    let num = |x: &str| x.trim().parse::<u32>().map_err(|_| bad(full, &format!("bad index {x:?}")));
    a.split(',')
        .map(|part| match part.split_once("..") {
            Some((lo, hi)) => Ok(Span::Range(num(lo)?, num(hi)?)),
            None => Ok(Span::One(num(part)?)),
        })
        .collect()
}

pub fn glob(pat: &str, s: &str) -> bool {
    match pat.split_once('*') {
        None => pat == s,
        Some((head, tail)) => {
            let Some(rest) = s.strip_prefix(head) else { return false };
            (0..=rest.len()).any(|k| rest.is_char_boundary(k) && glob(tail, &rest[k..]))
        }
    }
}

/// Does an index set select an instance with linear index `i` and grid coordinate `coord`?
pub fn index_matches(axes: &[Axis], i: u32, coord: &[u32]) -> bool {
    let hit = |axis: &Axis, v: u32| {
        axis.iter().any(|sp| match *sp {
            Span::All => true,
            Span::One(x) => x == v,
            Span::Range(lo, hi) => (lo..hi).contains(&v),
        })
    };
    match axes {
        [single] => hit(single, i),
        many => many.len() == coord.len() && many.iter().zip(coord).all(|(ax, &c)| hit(ax, c)),
    }
}

/// Largest explicit index in an index set, for E-IR-0211 range checks.
pub fn max_index(axes: &[Axis]) -> Vec<Option<u32>> {
    axes.iter()
        .map(|ax| {
            ax.iter()
                .filter_map(|sp| match *sp {
                    Span::One(x) => Some(x),
                    Span::Range(_, hi) => hi.checked_sub(1),
                    Span::All => None,
                })
                .max()
        })
        .collect()
}

/// Instance variables visible to `{...}` interpolation.
#[derive(Clone, Debug, Default)]
pub struct Vars {
    pub own: InstVars,
    pub up: Vec<InstVars>,
}

#[derive(Clone, Debug, Default)]
pub struct InstVars {
    pub i: u32,
    pub n: u32,
    pub coord: Vec<u32>,
}

/// Substitutes `{i}`, `{r}`, `{c}`, `{n}` and `{^i}` (`^^i` for further ancestors) in a reference string.
pub fn interpolate(s: &str, vars: &Vars) -> Result<String, Diagnostic> {
    if !s.contains('{') {
        return Ok(s.to_owned());
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = rest[open..].find('}').ok_or_else(|| bad(s, "unclosed '{'"))? + open;
        let name = rest[open + 1..close].trim();
        let ups = name.bytes().take_while(|&b| b == b'^').count();
        let v = if ups == 0 { Some(&vars.own) } else { vars.up.get(ups - 1) };
        let v = v.ok_or_else(|| {
            Diagnostic::error("E-IR-0204", format!("{s:?}: '{name}' has no enclosing replicated ancestor"))
        })?;
        let val = match &name[ups..] {
            "i" => v.i,
            "n" => v.n,
            "r" => v.coord.first().copied().unwrap_or(v.i),
            "c" => v.coord.get(1).copied().unwrap_or(0),
            other => {
                return Err(Diagnostic::error("E-IR-0204", format!("{s:?}: unknown instance variable '{other}'"))
                    .hint("interpolation supports {i} {r} {c} {n} and ^-prefixed ancestor forms like {^i}"));
            }
        };
        out.push_str(&val.to_string());
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_forms() {
        let s = Selector::parse("gpc[0..3].tpc*.sm*.l1").unwrap();
        assert_eq!(s.start, Start::Lexical);
        assert_eq!(s.segs.len(), 4);
        assert_eq!(Selector::parse("/board.gpu").unwrap().start, Start::Root);
        assert_eq!(Selector::parse("^.^.smem").unwrap().start, Start::Up(2));
        assert_eq!(Selector::parse("**.l1").unwrap().segs[0], Seg::AnyDepth);
        assert!(Selector::parse("a..b").is_err());
        assert!(Selector::parse("a[1").is_err());
    }

    #[test]
    fn globs_and_indices() {
        assert!(glob("sm*", "sm12"));
        assert!(glob("gpc0_*", "gpc0_3"));
        assert!(!glob("gpc0_*", "gpc1_3"));
        assert!(glob("*", "x"));
        let Seg::Pat { index: Some(ax), .. } = &Selector::parse("t[0..2;*]").unwrap().segs[0] else { panic!() };
        assert!(index_matches(ax, 9, &[1, 7]));
        assert!(!index_matches(ax, 9, &[2, 7]));
        let Seg::Pat { index: Some(ax), .. } = &Selector::parse("s[0,5,9]").unwrap().segs[0] else { panic!() };
        assert!(index_matches(ax, 5, &[]) && !index_matches(ax, 6, &[]));
    }

    #[test]
    fn interpolation() {
        let vars = Vars {
            own: InstVars { i: 2, n: 4, coord: vec![1, 0] },
            up: vec![InstVars { i: 3, n: 6, coord: vec![] }],
        };
        assert_eq!(interpolate("cc[{i}].up", &vars).unwrap(), "cc[2].up");
        assert_eq!(interpolate("/board.gpu.hbm{^i}", &vars).unwrap(), "/board.gpu.hbm3");
        assert_eq!(interpolate("t{r}_{c}", &vars).unwrap(), "t1_0");
        assert!(interpolate("x{^^i}", &vars).is_err());
    }
}
