//! Symbolic dimension expressions (02 §2.2) with exact rational evaluation.

use std::cmp::Ordering;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i128,
    den: i128,
}

impl Rational {
    pub const ZERO: Self = Self { num: 0, den: 1 };
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// `num/den` in lowest terms; `None` when `den == 0` or the reduced value does not fit `i128`.
    pub fn new(num: i128, den: i128) -> Option<Self> {
        Self::reduce(num.signum() * den.signum(), num.unsigned_abs(), den.unsigned_abs())
    }

    fn reduce(sign: i128, num: u128, den: u128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let g = gcd(num, den).max(1);
        let num = i128::try_from(num / g).ok()?;
        Some(Self { num: if sign < 0 { -num } else { num }, den: i128::try_from(den / g).ok()? })
    }

    pub const fn int(v: i128) -> Self {
        Self { num: v, den: 1 }
    }

    pub const fn num(self) -> i128 {
        self.num
    }

    pub const fn den(self) -> i128 {
        self.den
    }

    pub const fn is_integer(self) -> bool {
        self.den == 1
    }

    pub fn floor(self) -> Self {
        Self::int(self.num.div_euclid(self.den))
    }

    pub fn ceil(self) -> Self {
        Self::int(self.num.div_euclid(self.den) + i128::from(self.num.rem_euclid(self.den) != 0))
    }

    pub fn checked_add(self, o: Self) -> Option<Self> {
        let g = gcd(self.den.unsigned_abs(), o.den.unsigned_abs()) as i128;
        let (a, b) = (self.den / g, o.den / g);
        Self::new(self.num.checked_mul(b)?.checked_add(o.num.checked_mul(a)?)?, self.den.checked_mul(b)?)
    }

    pub fn checked_mul(self, o: Self) -> Option<Self> {
        let (g1, g2) = (gcd(self.num.unsigned_abs(), o.den.unsigned_abs()).max(1) as i128, gcd(o.num.unsigned_abs(), self.den.unsigned_abs()).max(1) as i128);
        Self::new((self.num / g1).checked_mul(o.num / g2)?, (self.den / g2).checked_mul(o.den / g1)?)
    }

    /// `None` on division by zero or overflow.
    pub fn checked_div(self, o: Self) -> Option<Self> {
        if o.num == 0 {
            return None;
        }
        self.checked_mul(Self::reduce(o.num.signum(), o.den.unsigned_abs(), o.num.unsigned_abs())?)
    }

    /// The value as a non-negative integer, if it is one.
    pub fn to_u64(self) -> Option<u64> {
        (self.den == 1)
            .then(|| u64::try_from(self.num).ok())
            .flatten()
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl std::ops::Mul for Rational {
    type Output = Self;
    /// Panics on overflow; [`Rational::checked_mul`] reports it.
    fn mul(self, o: Self) -> Self {
        self.checked_mul(o).expect("rational overflow")
    }
}

impl Ord for Rational {
    fn cmp(&self, o: &Self) -> Ordering {
        // Continued-fraction comparison: exact, no products.
        let (mut a, mut b, mut c, mut d) = (self.num, self.den, o.num, o.den);
        let mut flip = false;
        loop {
            let (qa, qc) = (a.div_euclid(b), c.div_euclid(d));
            if qa != qc {
                let ord = qa.cmp(&qc);
                return if flip { ord.reverse() } else { ord };
            }
            let (ra, rc) = (a.rem_euclid(b), c.rem_euclid(d));
            match (ra == 0, rc == 0) {
                (true, true) => return Ordering::Equal,
                (true, false) => return if flip { Ordering::Greater } else { Ordering::Less },
                (false, true) => return if flip { Ordering::Less } else { Ordering::Greater },
                (false, false) => {
                    (a, b, c, d) = (b, ra, d, rc);
                    flip = !flip;
                }
            }
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({}/{})", self.num, self.den)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegField {
    Count,
    QLen,
    KvLen,
    CountXQ,
    CountXKv,
}

impl SegField {
    const ALL: [(Self, &'static str); 5] = [
        (Self::Count, "count"),
        (Self::QLen, "q_len"),
        (Self::KvLen, "kv_len"),
        (Self::CountXQ, "count_x_q"),
        (Self::CountXKv, "count_x_kv"),
    ];

    pub fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(f, _)| *f == self)
            .map(|(_, n)| *n)
            .expect("all fields named")
    }

    fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().find(|(_, n)| *n == s).map(|(f, _)| *f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegAgg {
    Sum,
    Max,
}

/// A dimension expression. Always held in canonical form (§11.5): constants folded, sums and products flattened
/// and sorted, so structural equality is canonical equality.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Num(Rational),
    Sym(String),
    Seg {
        sym: String,
        agg: SegAgg,
        field: SegField,
    },
    Add(Vec<DimExpr>),
    Mul(Vec<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Ceil(Box<DimExpr>),
    Floor(Box<DimExpr>),
    Min(Box<DimExpr>, Box<DimExpr>),
    Max(Box<DimExpr>, Box<DimExpr>),
}

/// Symbol values for evaluation.
pub trait DimEnv {
    fn sym(&self, name: &str) -> Option<u64>;
    fn seg(&self, sym: &str, agg: SegAgg, field: SegField) -> Option<u64>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    Unbound(String),
    DivByZero,
    Overflow,
}

impl DimExpr {
    pub fn int(v: u64) -> Self {
        Self::Num(Rational::int(v.into()))
    }

    pub fn sym(name: impl Into<String>) -> Self {
        Self::Sym(name.into())
    }

    pub fn parse(src: &str) -> Result<Self, String> {
        let toks = tokenize(src)?;
        let mut p = Parser { toks, pos: 0 };
        let e = p.expr()?;
        if p.pos != p.toks.len() {
            return Err(format!("unexpected trailing input in {src:?}"));
        }
        e.canonical().ok_or_else(|| format!("constant overflow in {src:?}"))
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Num(r) => r.to_u64(),
            _ => None,
        }
    }

    pub fn eval(&self, env: &dyn DimEnv) -> Result<Rational, EvalError> {
        Ok(match self {
            Self::Num(r) => *r,
            Self::Sym(s) => Rational::int(
                env.sym(s)
                    .ok_or_else(|| EvalError::Unbound(s.clone()))?
                    .into(),
            ),
            Self::Seg { sym, agg, field } => Rational::int(
                env.seg(sym, *agg, *field)
                    .ok_or_else(|| EvalError::Unbound(sym.clone()))?
                    .into(),
            ),
            Self::Add(v) => v.iter().try_fold(Rational::ZERO, |a, e| {
                a.checked_add(e.eval(env)?).ok_or(EvalError::Overflow)
            })?,
            Self::Mul(v) => v.iter().try_fold(Rational::ONE, |a, e| {
                a.checked_mul(e.eval(env)?).ok_or(EvalError::Overflow)
            })?,
            Self::Div(a, b) => {
                let (x, y) = (a.eval(env)?, b.eval(env)?);
                if y == Rational::ZERO {
                    return Err(EvalError::DivByZero);
                }
                x.checked_div(y).ok_or(EvalError::Overflow)?
            }
            Self::Ceil(a) => a.eval(env)?.ceil(),
            Self::Floor(a) => a.eval(env)?.floor(),
            Self::Min(a, b) => a.eval(env)?.min(b.eval(env)?),
            Self::Max(a, b) => a.eval(env)?.max(b.eval(env)?),
        })
    }

    fn visit(&self, f: &mut impl FnMut(&Self)) {
        f(self);
        match self {
            Self::Num(_) | Self::Sym(_) | Self::Seg { .. } => {}
            Self::Add(v) | Self::Mul(v) => v.iter().for_each(|e| e.visit(f)),
            Self::Div(a, b) | Self::Min(a, b) | Self::Max(a, b) => {
                a.visit(f);
                b.visit(f);
            }
            Self::Ceil(a) | Self::Floor(a) => a.visit(f),
        }
    }

    /// Size symbols referenced (deduplicated, in first-use order).
    pub fn size_symbols(&self, out: &mut Vec<String>) {
        self.visit(&mut |e| {
            if let Self::Sym(s) = e
                && !out.contains(s)
            {
                out.push(s.clone());
            }
        });
    }

    /// Segments symbols referenced through `sum(..)`/`max(..)`.
    pub fn segment_symbols(&self, out: &mut Vec<String>) {
        self.visit(&mut |e| {
            if let Self::Seg { sym, .. } = e
                && !out.contains(sym)
            {
                out.push(sym.clone());
            }
        });
    }

    /// Canonical form; `None` when constant folding overflows.
    pub fn canonical(self) -> Option<Self> {
        Some(match self {
            Self::Add(v) => {
                let mut c = Rational::ZERO;
                let mut terms = Vec::new();
                for e in v {
                    match e.canonical()? {
                        Self::Add(inner) => {
                            for t in inner {
                                match t {
                                    Self::Num(r) => c = c.checked_add(r)?,
                                    t => terms.push(t),
                                }
                            }
                        }
                        Self::Num(r) => c = c.checked_add(r)?,
                        t => terms.push(t),
                    }
                }
                terms.sort_by_cached_key(|t| t.to_string());
                if c != Rational::ZERO {
                    terms.insert(0, Self::Num(c));
                }
                match terms.len() {
                    0 => Self::Num(Rational::ZERO),
                    1 => terms.pop().expect("one term"),
                    _ => Self::Add(terms),
                }
            }
            Self::Mul(v) => {
                let mut c = Rational::ONE;
                let mut fs = Vec::new();
                for e in v {
                    match e.canonical()? {
                        Self::Mul(inner) => {
                            for t in inner {
                                match t {
                                    Self::Num(r) => c = c.checked_mul(r)?,
                                    t => fs.push(t),
                                }
                            }
                        }
                        Self::Num(r) => c = c.checked_mul(r)?,
                        t => fs.push(t),
                    }
                }
                if c == Rational::ZERO {
                    return Some(Self::Num(c));
                }
                fs.sort_by_cached_key(|t| t.to_string());
                if c != Rational::ONE || fs.is_empty() {
                    fs.insert(0, Self::Num(c));
                }
                if fs.len() == 1 {
                    fs.pop().expect("one factor")
                } else {
                    Self::Mul(fs)
                }
            }
            Self::Div(a, b) => match (a.canonical()?, b.canonical()?) {
                (Self::Num(x), Self::Num(y)) if y != Rational::ZERO => Self::Num(x.checked_div(y)?),
                (x, Self::Num(y)) if y != Rational::ZERO => {
                    Self::Mul(vec![Self::Num(Rational::ONE.checked_div(y)?), x]).canonical()?
                }
                (x, y) => Self::Div(Box::new(x), Box::new(y)),
            },
            Self::Ceil(a) => match a.canonical()? {
                Self::Num(r) => Self::Num(r.ceil()),
                x => Self::Ceil(Box::new(x)),
            },
            Self::Floor(a) => match a.canonical()? {
                Self::Num(r) => Self::Num(r.floor()),
                x => Self::Floor(Box::new(x)),
            },
            Self::Min(a, b) => minmax(*a, *b, false)?,
            Self::Max(a, b) => minmax(*a, *b, true)?,
            e => e,
        })
    }
}

fn minmax(a: DimExpr, b: DimExpr, is_max: bool) -> Option<DimExpr> {
    let (a, b) = (a.canonical()?, b.canonical()?);
    if let (DimExpr::Num(x), DimExpr::Num(y)) = (&a, &b) {
        return Some(DimExpr::Num(if is_max { (*x).max(*y) } else { (*x).min(*y) }));
    }
    let (a, b) = if a.to_string() <= b.to_string() {
        (a, b)
    } else {
        (b, a)
    };
    Some(if is_max {
        DimExpr::Max(Box::new(a), Box::new(b))
    } else {
        DimExpr::Min(Box::new(a), Box::new(b))
    })
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Num(r) => write!(f, "{r}"),
            Self::Sym(s) => f.write_str(s),
            Self::Seg { sym, agg, field } => {
                let a = if *agg == SegAgg::Sum { "sum" } else { "max" };
                write!(f, "{a}({sym}.{})", field.name())
            }
            Self::Add(v) => {
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" + ")?;
                    }
                    write!(f, "{e}")?;
                }
                Ok(())
            }
            Self::Mul(v) => {
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        f.write_str("*")?;
                    }
                    if matches!(e, Self::Add(_)) {
                        write!(f, "({e})")?
                    } else {
                        write!(f, "{e}")?
                    }
                }
                Ok(())
            }
            Self::Div(a, b) => write!(f, "({a})/({b})"),
            Self::Ceil(a) => write!(f, "ceil({a})"),
            Self::Floor(a) => write!(f, "floor({a})"),
            Self::Min(a, b) => write!(f, "min({a}, {b})"),
            Self::Max(a, b) => write!(f, "max({a}, {b})"),
        }
    }
}

impl From<u64> for DimExpr {
    fn from(v: u64) -> Self {
        Self::int(v)
    }
}

impl From<&str> for DimExpr {
    fn from(s: &str) -> Self {
        Self::parse(s).unwrap_or_else(|e| panic!("invalid dim expression {s:?}: {e}"))
    }
}

impl Serialize for DimExpr {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.as_u64() {
            Some(v) => s.serialize_u64(v),
            None => s.collect_str(self),
        }
    }
}

impl<'de> Deserialize<'de> for DimExpr {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Src {
            Int(u64),
            Str(String),
        }
        match Src::deserialize(d)? {
            Src::Int(v) => Ok(Self::int(v)),
            Src::Str(s) => Self::parse(&s).map_err(serde::de::Error::custom),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(Rational),
    Ident(String),
    Punct(char),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let int: i128 = src[start..i]
                .parse()
                .map_err(|_| format!("number too large in {src:?}"))?;
            if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
                let fs = i + 1;
                i = fs;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                let too_long = || format!("decimal too long in {src:?}");
                let scale = 10i128.checked_pow((i - fs) as u32).ok_or_else(too_long)?;
                let frac: i128 = src[fs..i].parse().map_err(|_| too_long())?;
                let num = int.checked_mul(scale).and_then(|n| n.checked_add(frac)).ok_or_else(too_long)?;
                out.push(Tok::Num(Rational::new(num, scale).ok_or_else(too_long)?));
            } else {
                out.push(Tok::Num(Rational::int(int)));
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(src[start..i].to_string()));
        } else if "+-*/(),.".contains(c) {
            out.push(Tok::Punct(c));
            i += 1;
        } else {
            return Err(format!("unexpected character {c:?} in {src:?}"));
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Punct(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(format!("expected {c:?} at token {}", self.pos))
        }
    }

    fn expr(&mut self) -> Result<DimExpr, String> {
        let mut terms = vec![self.term()?];
        loop {
            if self.eat('+') {
                terms.push(self.term()?);
            } else if self.eat('-') {
                terms.push(DimExpr::Mul(vec![
                    DimExpr::Num(Rational::int(-1)),
                    self.term()?,
                ]));
            } else {
                break;
            }
        }
        Ok(if terms.len() == 1 {
            terms.pop().expect("one")
        } else {
            DimExpr::Add(terms)
        })
    }

    fn term(&mut self) -> Result<DimExpr, String> {
        let mut acc = self.factor()?;
        loop {
            if self.eat('*') {
                acc = DimExpr::Mul(vec![acc, self.factor()?]);
            } else if self.eat('/') {
                acc = DimExpr::Div(Box::new(acc), Box::new(self.factor()?));
            } else {
                return Ok(acc);
            }
        }
    }

    fn factor(&mut self) -> Result<DimExpr, String> {
        match self.toks.get(self.pos).cloned() {
            Some(Tok::Num(r)) => {
                self.pos += 1;
                Ok(DimExpr::Num(r))
            }
            Some(Tok::Punct('-')) => {
                self.pos += 1;
                Ok(DimExpr::Mul(vec![
                    DimExpr::Num(Rational::int(-1)),
                    self.factor()?,
                ]))
            }
            Some(Tok::Punct('(')) => {
                self.pos += 1;
                let e = self.expr()?;
                self.expect(')')?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                self.pos += 1;
                if !self.eat('(') {
                    return Ok(DimExpr::Sym(name));
                }
                let r = self.call(&name)?;
                self.expect(')')?;
                Ok(r)
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }

    fn call(&mut self, name: &str) -> Result<DimExpr, String> {
        if matches!(name, "sum" | "max")
            && let (Some(Tok::Ident(sym)), Some(Tok::Punct('.')), Some(Tok::Ident(field))) = (
                self.toks.get(self.pos).cloned(),
                self.toks.get(self.pos + 1),
                self.toks.get(self.pos + 2).cloned(),
            )
        {
            let field = SegField::parse(&field)
                .ok_or_else(|| format!("unknown segment field {field:?}"))?;
            self.pos += 3;
            let agg = if name == "sum" {
                SegAgg::Sum
            } else {
                SegAgg::Max
            };
            return Ok(DimExpr::Seg { sym, agg, field });
        }
        let a = self.expr()?;
        let two = |p: &mut Self| -> Result<DimExpr, String> {
            p.expect(',')?;
            p.expr()
        };
        Ok(match name {
            "ceil" => DimExpr::Ceil(Box::new(a)),
            "floor" => DimExpr::Floor(Box::new(a)),
            "min" => DimExpr::Min(Box::new(a), Box::new(two(self)?)),
            "max" => DimExpr::Max(Box::new(a), Box::new(two(self)?)),
            "ceil_div" => DimExpr::Ceil(Box::new(DimExpr::Div(Box::new(a), Box::new(two(self)?)))),
            _ => return Err(format!("unknown function {name:?}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Env;
    impl DimEnv for Env {
        fn sym(&self, n: &str) -> Option<u64> {
            match n {
                "T" => Some(8),
                "L" => Some(32),
                _ => None,
            }
        }
        fn seg(&self, _: &str, agg: SegAgg, f: SegField) -> Option<u64> {
            Some(match (agg, f) {
                (SegAgg::Max, SegField::KvLen) => 2048,
                _ => 8,
            })
        }
    }

    fn ev(s: &str) -> Rational {
        DimExpr::parse(s).unwrap().eval(&Env).unwrap()
    }

    #[test]
    fn evaluates_grammar() {
        assert_eq!(ev("T*4096"), Rational::int(32768));
        assert_eq!(ev("1.25*T"), Rational::int(10));
        assert_eq!(ev("ceil(5/4*T*2/8)"), Rational::int(3));
        assert_eq!(ev("ceil_div(T, 3) + floor(T/3)"), Rational::int(5));
        assert_eq!(ev("max(seqs.kv_len) - min(T, L)"), Rational::int(2040));
        assert_eq!(ev("max(T, L)"), Rational::int(32));
        assert_eq!(ev("sum(seqs.count_x_q)"), Rational::int(8));
        assert_eq!(
            DimExpr::parse("X").unwrap().eval(&Env),
            Err(EvalError::Unbound("X".into()))
        );
    }

    #[test]
    fn overflow_is_an_error_not_a_wrap() {
        assert!(DimExpr::parse("18446744073709551616*18446744073709551616+1").is_err());
        assert!(DimExpr::parse("170141183460469231731687303715884105727 + 1").is_err());
        let big = DimExpr::parse("170141183460469231731687303715884105727*T").unwrap();
        assert_eq!(big.eval(&Env), Err(EvalError::Overflow));
        assert!(Rational::new(1, 3).unwrap() < Rational::new(i128::MAX / 2, i128::MAX / 2 + 1).unwrap());
    }

    #[test]
    fn canonical_is_order_insensitive_and_stable() {
        let a = DimExpr::parse("L*2 + T + 3 - 1").unwrap();
        let b = DimExpr::parse("2 + T + 2*L").unwrap();
        assert_eq!(a, b);
        assert_eq!(DimExpr::parse(&a.to_string()).unwrap(), a);
        assert_eq!(DimExpr::parse("4096*1").unwrap().as_u64(), Some(4096));
        assert_eq!(DimExpr::parse("(5/4)*T").unwrap().to_string(), "(5/4)*T");
        assert_eq!(
            serde_json::to_string(&DimExpr::parse("2*2048").unwrap()).unwrap(),
            "4096"
        );
    }
}
