//! `=expr` expressions (01 §14.3): total, deterministic f64 arithmetic with dimension checking.

use serde_json::Value;

use super::quantity::{Dim, apply_suffix, parse_literal, to_literal};
use crate::common::Diagnostic;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q {
    pub v: f64,
    pub dim: Dim,
}

impl Q {
    fn num(v: f64) -> Self {
        Self { v, dim: Dim::NONE }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64, Dim),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
}

const OPS: &[&str] = &["<=", ">=", "==", "!=", "<", ">", "+", "-", "*", "/", "%", "^"];
const INSTANCE_VARS: &[&str] = &["i", "r", "c", "n"];

fn err(expr: &str, msg: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::error("E-IR-0204", format!("expression {expr:?}: {msg}"))
}

fn lex(src: &str) -> Result<Vec<Tok>, Diagnostic> {
    let b = src.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let v: f64 = src[start..i].parse().map_err(|_| err(src, format!("bad number {:?}", &src[start..i])))?;
            let ustart = i;
            while i < b.len() && b[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let mut unit = &src[ustart..i];
            if i + 1 < b.len() && b[i] == b'/' && b[i + 1].is_ascii_alphabetic() {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_alphanumeric() {
                    j += 1;
                }
                if apply_suffix(1.0, &src[ustart..j]).is_some() {
                    unit = &src[ustart..j];
                    i = j;
                }
            }
            if unit.is_empty() {
                out.push(Tok::Num(v, Dim::NONE));
            } else {
                let (q, d) = apply_suffix(v, unit).ok_or_else(|| err(src, format!("unknown unit {unit:?}")))?;
                out.push(Tok::Num(q, d));
            }
        } else if c.is_ascii_alphabetic() || c == '_' || c == '^' && b.get(i + 1).is_some_and(u8::is_ascii_alphabetic) {
            let start = i;
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(src[start..i].to_owned()));
        } else if c == '(' {
            out.push(Tok::LParen);
            i += 1;
        } else if c == ')' {
            out.push(Tok::RParen);
            i += 1;
        } else if c == ',' {
            out.push(Tok::Comma);
            i += 1;
        } else if let Some(op) = OPS.iter().find(|op| src[i..].starts_with(**op)) {
            out.push(Tok::Op(op));
            i += op.len();
        } else {
            return Err(err(src, format!("unexpected character {c:?}")));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    src: &'a str,
    lookup: &'a dyn Fn(&str) -> Option<Value>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat_op(&mut self, ops: &[&str]) -> Option<&'static str> {
        match self.peek() {
            Some(Tok::Op(o)) if ops.contains(o) => {
                let o = *o;
                self.pos += 1;
                Some(o)
            }
            _ => None,
        }
    }

    fn same_dim(&self, a: Q, b: Q, op: &str) -> Result<(), Diagnostic> {
        if a.dim == b.dim {
            Ok(())
        } else {
            Err(err(self.src, format!("'{op}' needs equal dimensions, got {} and {}", a.dim, b.dim))
                .hint("operands of + - % min max and comparisons must have the same unit"))
        }
    }

    fn expr(&mut self) -> Result<Q, Diagnostic> {
        let a = self.additive()?;
        if let Some(op) = self.eat_op(&["<=", ">=", "==", "!=", "<", ">"]) {
            let b = self.additive()?;
            self.same_dim(a, b, op)?;
            let r = match op {
                "<=" => a.v <= b.v,
                ">=" => a.v >= b.v,
                "==" => a.v == b.v,
                "!=" => a.v != b.v,
                "<" => a.v < b.v,
                _ => a.v > b.v,
            };
            return Ok(Q::num(f64::from(u8::from(r))));
        }
        Ok(a)
    }

    fn additive(&mut self) -> Result<Q, Diagnostic> {
        let mut a = self.term()?;
        while let Some(op) = self.eat_op(&["+", "-"]) {
            let b = self.term()?;
            self.same_dim(a, b, op)?;
            a.v = if op == "+" { a.v + b.v } else { a.v - b.v };
        }
        Ok(a)
    }

    fn term(&mut self) -> Result<Q, Diagnostic> {
        let mut a = self.unary()?;
        while let Some(op) = self.eat_op(&["*", "/", "%"]) {
            let b = self.unary()?;
            a = match op {
                "*" => Q { v: a.v * b.v, dim: a.dim * b.dim },
                _ if b.v == 0.0 => return Err(err(self.src, "division by zero")),
                "/" => Q { v: a.v / b.v, dim: a.dim / b.dim },
                _ => {
                    self.same_dim(a, b, "%")?;
                    Q { v: a.v % b.v, dim: a.dim }
                }
            };
        }
        Ok(a)
    }

    fn unary(&mut self) -> Result<Q, Diagnostic> {
        if self.eat_op(&["-"]).is_some() {
            let q = self.unary()?;
            return Ok(Q { v: -q.v, dim: q.dim });
        }
        let base = self.atom()?;
        if self.eat_op(&["^"]).is_some() {
            let e = self.unary()?;
            return self.pow(base, e);
        }
        Ok(base)
    }

    fn pow(&self, base: Q, e: Q) -> Result<Q, Diagnostic> {
        if !e.dim.is_none() {
            return Err(err(self.src, "exponent must be dimensionless"));
        }
        if base.dim.is_none() {
            return Ok(Q::num(base.v.powf(e.v)));
        }
        if e.v.fract() != 0.0 || e.v.abs() > 8.0 {
            return Err(err(self.src, "a quantity can only be raised to a small integer power"));
        }
        Ok(Q { v: base.v.powf(e.v), dim: base.dim.scale(e.v as i8) })
    }

    fn atom(&mut self) -> Result<Q, Diagnostic> {
        let tok = self.toks.get(self.pos).cloned().ok_or_else(|| err(self.src, "unexpected end"))?;
        self.pos += 1;
        match tok {
            Tok::Num(v, dim) => Ok(Q { v, dim }),
            Tok::LParen => {
                let q = self.expr()?;
                self.expect_rparen()?;
                Ok(q)
            }
            Tok::Ident(name) if self.peek() == Some(&Tok::LParen) => {
                self.pos += 1;
                let mut args = vec![];
                if self.peek() != Some(&Tok::RParen) {
                    loop {
                        args.push(self.expr()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                }
                self.expect_rparen()?;
                self.call(&name, &args)
            }
            Tok::Ident(name) => self.name(&name),
            other => Err(err(self.src, format!("unexpected token {other:?}"))),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), Diagnostic> {
        if self.peek() == Some(&Tok::RParen) {
            self.pos += 1;
            Ok(())
        } else {
            Err(err(self.src, "expected ')'"))
        }
    }

    fn name(&self, name: &str) -> Result<Q, Diagnostic> {
        if INSTANCE_VARS.contains(&name.trim_start_matches('^')) && (self.lookup)(name).is_none() {
            return Err(err(self.src, format!("instance variable '{name}' is not supported in expressions"))
                .hint("use \"{i}\" interpolation inside reference/selector strings, or `vary` for per-instance values"));
        }
        let v = (self.lookup)(name).ok_or_else(|| {
            err(self.src, format!("unknown name '{name}'")).hint("declare it in `params` or the template's `params`")
        })?;
        value_to_q(&v).ok_or_else(|| err(self.src, format!("param '{name}' = {v} is not numeric")))
    }

    fn call(&self, f: &str, args: &[Q]) -> Result<Q, Diagnostic> {
        let arity = |n: usize| {
            if args.len() == n { Ok(()) } else { Err(err(self.src, format!("{f}() takes {n} argument(s)"))) }
        };
        let one = |op: fn(f64) -> f64| -> Result<Q, Diagnostic> {
            arity(1)?;
            Ok(Q { v: op(args[0].v), dim: args[0].dim })
        };
        match f {
            "min" | "max" => {
                let first = *args.first().ok_or_else(|| err(self.src, format!("{f}() needs arguments")))?;
                args.iter().try_fold(first, |acc, &q| {
                    self.same_dim(acc, q, f)?;
                    Ok(if (f == "min") == (q.v < acc.v) { q } else { acc })
                })
            }
            "ceil" => one(f64::ceil),
            "floor" => one(f64::floor),
            "round" => one(f64::round),
            "abs" => one(f64::abs),
            "sqrt" => {
                arity(1)?;
                let dim = args[0].dim.halve().ok_or_else(|| err(self.src, "sqrt of a quantity with odd dimension"))?;
                Ok(Q { v: args[0].v.sqrt(), dim })
            }
            "log2" => {
                arity(1)?;
                if !args[0].dim.is_none() {
                    return Err(err(self.src, "log2 needs a dimensionless argument"));
                }
                Ok(Q::num(args[0].v.log2()))
            }
            "pow" => {
                arity(2)?;
                self.pow(args[0], args[1])
            }
            "if" => {
                arity(3)?;
                self.same_dim(args[1], args[2], "if")?;
                Ok(if args[0].v != 0.0 { args[1] } else { args[2] })
            }
            _ => Err(err(self.src, format!("unknown function '{f}'"))
                .hint("functions: min max ceil floor round sqrt log2 pow abs if")),
        }
    }
}

pub fn value_to_q(v: &Value) -> Option<Q> {
    match v {
        Value::Number(n) => n.as_f64().map(Q::num),
        Value::String(s) => parse_literal(s).map(|(v, dim)| Q { v, dim }),
        Value::Bool(b) => Some(Q::num(f64::from(u8::from(*b)))),
        _ => None,
    }
}

pub fn is_expr(s: &str) -> bool {
    s.starts_with('=')
}

/// Evaluates `"=..."`. A bare single name substitutes the param value verbatim, whatever its JSON type.
pub fn eval(src: &str, lookup: &dyn Fn(&str) -> Option<Value>) -> Result<Value, Diagnostic> {
    let body = src.strip_prefix('=').unwrap_or(src).trim();
    let toks = lex(body)?;
    if let [Tok::Ident(name)] = toks.as_slice() {
        let p = Parser { toks: vec![], pos: 0, src: body, lookup };
        return match lookup(name) {
            Some(v) => Ok(v),
            None => p.name(name).map(|q| to_literal(q.v, q.dim)),
        };
    }
    let mut p = Parser { toks, pos: 0, src: body, lookup };
    let q = p.expr()?;
    if p.pos != p.toks.len() {
        return Err(err(body, "trailing input"));
    }
    if !q.v.is_finite() {
        return Err(err(body, "result is not finite"));
    }
    Ok(to_literal(q.v, q.dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn env(name: &str) -> Option<Value> {
        match name {
            "n_mxu" => Some(json!(4)),
            "mxu_edge" => Some(json!(128)),
            "clk" => Some(json!("1500MHz")),
            "cap" => Some(json!("512KiB")),
            "precs" => Some(json!(["fp32@1"])),
            _ => None,
        }
    }

    fn ev(s: &str) -> Result<Value, Diagnostic> {
        eval(s, &env)
    }

    #[test]
    fn arithmetic_and_units() {
        assert_eq!(ev("= n_mxu * 4").unwrap(), json!(16));
        assert_eq!(ev("= 2 * mxu_edge * mxu_edge * 2").unwrap(), json!(65536));
        assert_eq!(ev("= 2 * cap").unwrap(), json!("1048576B"));
        assert_eq!(ev("= clk * 2").unwrap(), json!("3000000000Hz"));
        assert_eq!(ev("= 40MiB / 2").unwrap(), json!("20971520B"));
        assert_eq!(ev("= 1.41GHz * 4").unwrap(), json!("5640000000Hz"));
        assert_eq!(ev("= 3.2Gbps * 1024 / 8").unwrap(), json!("409600000000b/s"));
        assert_eq!(ev("= max(2, 3) + if(n_mxu > 2, 1, 0) + 2^3").unwrap(), json!(12));
        assert_eq!(ev("= ceil(7 / 2) + floor(log2(1024)) + sqrt(16) + -1").unwrap(), json!(17));
    }

    #[test]
    fn verbatim_substitution() {
        assert_eq!(ev("=precs").unwrap(), json!(["fp32@1"]));
        assert_eq!(ev("=clk").unwrap(), json!("1500MHz"));
    }

    #[test]
    fn errors_are_e0204() {
        for bad in ["= 1 / 0", "= nope + 1", "= clk + cap", "= 1 +", "= i * 2", "= foo(1)", "= precs + 1"] {
            assert_eq!(ev(bad).unwrap_err().code, "E-IR-0204", "{bad}");
        }
    }
}
