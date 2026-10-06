//! Unit newtypes and quantity literals (01 §2). IR values are SI base units; authoring accepts `"<num><suffix>"`.

use std::fmt;
use std::ops::{Div, Mul};

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::diag::de_error;
use crate::common::Diagnostic;

/// Exponents over [bytes, bits, seconds, joules, volts, micrometres].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Dim(pub [i8; 6]);

impl Dim {
    pub const NONE: Dim = Dim([0; 6]);
    pub const BYTES: Dim = Dim([1, 0, 0, 0, 0, 0]);
    pub const BITS: Dim = Dim([0, 1, 0, 0, 0, 0]);
    pub const SECONDS: Dim = Dim([0, 0, 1, 0, 0, 0]);
    pub const HZ: Dim = Dim([0, 0, -1, 0, 0, 0]);
    pub const BYTES_PER_S: Dim = Dim([1, 0, -1, 0, 0, 0]);
    pub const BITS_PER_S: Dim = Dim([0, 1, -1, 0, 0, 0]);
    pub const JOULES: Dim = Dim([0, 0, 0, 1, 0, 0]);
    pub const J_PER_BYTE: Dim = Dim([-1, 0, 0, 1, 0, 0]);
    pub const J_PER_BIT: Dim = Dim([0, -1, 0, 1, 0, 0]);
    pub const WATTS: Dim = Dim([0, 0, -1, 1, 0, 0]);
    pub const VOLTS: Dim = Dim([0, 0, 0, 0, 1, 0]);
    pub const UM: Dim = Dim([0, 0, 0, 0, 0, 1]);
    pub const UM2: Dim = Dim([0, 0, 0, 0, 0, 2]);

    pub fn scale(self, k: i8) -> Dim {
        Dim(self.0.map(|e| e * k))
    }

    pub fn halve(self) -> Option<Dim> {
        self.0.iter().all(|e| e % 2 == 0).then(|| Dim(self.0.map(|e| e / 2)))
    }

    pub fn is_none(self) -> bool {
        self == Dim::NONE
    }

    /// Canonical base-unit suffix and scale used when an expression result is written back as a literal.
    fn base_suffix(self) -> Option<(&'static str, f64)> {
        BASE_SUFFIXES.iter().find(|(d, ..)| *d == self).map(|(_, s, f)| (*s, *f))
    }
}

impl Mul for Dim {
    type Output = Dim;
    fn mul(self, o: Dim) -> Dim {
        Dim(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }
}

impl Div for Dim {
    type Output = Dim;
    fn div(self, o: Dim) -> Dim {
        Dim(std::array::from_fn(|i| self.0[i] - o.0[i]))
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((s, _)) = self.base_suffix() {
            return f.write_str(s);
        }
        let names = ["B", "b", "s", "J", "V", "um"];
        let parts: Vec<String> =
            self.0.iter().zip(names).filter(|(e, _)| **e != 0).map(|(e, n)| format!("{n}^{e}")).collect();
        if parts.is_empty() { f.write_str("dimensionless") } else { f.write_str(&parts.join("*")) }
    }
}

const BASE_SUFFIXES: &[(Dim, &str, f64)] = &[
    (Dim::BYTES, "B", 1.0),
    (Dim::BITS, "b", 1.0),
    (Dim::SECONDS, "s", 1.0),
    (Dim::HZ, "Hz", 1.0),
    (Dim::BYTES_PER_S, "B/s", 1.0),
    (Dim::BITS_PER_S, "b/s", 1.0),
    (Dim::JOULES, "J", 1.0),
    (Dim::J_PER_BYTE, "J/B", 1.0),
    (Dim::WATTS, "W", 1.0),
    (Dim::VOLTS, "V", 1.0),
    (Dim::UM, "um", 1.0),
    (Dim::UM2, "mm2", 1e6),
];

/// Every accepted suffix, in global base units (Mm2 is stored in um^2 globally).
/// (suffix, multiplier, divisor, dimension); sub-unit prefixes divide so `12ns` is exactly `12 / 1e9`.
const SUFFIXES: &[(&str, f64, f64, Dim)] = &[
    ("B", 1.0, 1.0, Dim::BYTES),
    ("KB", 1e3, 1.0, Dim::BYTES),
    ("MB", 1e6, 1.0, Dim::BYTES),
    ("GB", 1e9, 1.0, Dim::BYTES),
    ("TB", 1e12, 1.0, Dim::BYTES),
    ("KiB", 1024.0, 1.0, Dim::BYTES),
    ("MiB", 1048576.0, 1.0, Dim::BYTES),
    ("GiB", 1073741824.0, 1.0, Dim::BYTES),
    ("TiB", 1099511627776.0, 1.0, Dim::BYTES),
    ("b", 1.0, 1.0, Dim::BITS),
    ("Kib", 1024.0, 1.0, Dim::BITS),
    ("Mib", 1048576.0, 1.0, Dim::BITS),
    ("Hz", 1.0, 1.0, Dim::HZ),
    ("kHz", 1e3, 1.0, Dim::HZ),
    ("MHz", 1e6, 1.0, Dim::HZ),
    ("GHz", 1e9, 1.0, Dim::HZ),
    ("B/s", 1.0, 1.0, Dim::BYTES_PER_S),
    ("KB/s", 1e3, 1.0, Dim::BYTES_PER_S),
    ("MB/s", 1e6, 1.0, Dim::BYTES_PER_S),
    ("GB/s", 1e9, 1.0, Dim::BYTES_PER_S),
    ("TB/s", 1e12, 1.0, Dim::BYTES_PER_S),
    ("GiB/s", 1073741824.0, 1.0, Dim::BYTES_PER_S),
    ("TiB/s", 1099511627776.0, 1.0, Dim::BYTES_PER_S),
    ("b/s", 1.0, 1.0, Dim::BITS_PER_S),
    ("Mb/s", 1e6, 1.0, Dim::BITS_PER_S),
    ("Gb/s", 1e9, 1.0, Dim::BITS_PER_S),
    ("Gbps", 1e9, 1.0, Dim::BITS_PER_S),
    ("s", 1.0, 1.0, Dim::SECONDS),
    ("ms", 1.0, 1e3, Dim::SECONDS),
    ("us", 1.0, 1e6, Dim::SECONDS),
    ("ns", 1.0, 1e9, Dim::SECONDS),
    ("ps", 1.0, 1e12, Dim::SECONDS),
    ("J", 1.0, 1.0, Dim::JOULES),
    ("mJ", 1.0, 1e3, Dim::JOULES),
    ("uJ", 1.0, 1e6, Dim::JOULES),
    ("nJ", 1.0, 1e9, Dim::JOULES),
    ("pJ", 1.0, 1e12, Dim::JOULES),
    ("fJ", 1.0, 1e15, Dim::JOULES),
    ("J/B", 1.0, 1.0, Dim::J_PER_BYTE),
    ("pJ/B", 1.0, 1e12, Dim::J_PER_BYTE),
    ("pJ/b", 1.0, 1e12, Dim::J_PER_BIT),
    ("W", 1.0, 1.0, Dim::WATTS),
    ("mW", 1.0, 1e3, Dim::WATTS),
    ("kW", 1e3, 1.0, Dim::WATTS),
    ("V", 1.0, 1.0, Dim::VOLTS),
    ("mV", 1.0, 1e3, Dim::VOLTS),
    ("um", 1.0, 1.0, Dim::UM),
    ("mm", 1e3, 1.0, Dim::UM),
    ("mm2", 1e6, 1.0, Dim::UM2),
    ("um2", 1.0, 1.0, Dim::UM2),
    ("cyc", 1.0, 1.0, Dim::NONE),
];

/// Converts `v` in units of `suffix` to global base units.
pub fn apply_suffix(v: f64, suffix: &str) -> Option<(f64, Dim)> {
    SUFFIXES.iter().find(|(s, ..)| *s == suffix).map(|(_, mul, div, d)| (v * mul / div, *d))
}

/// Parses `"<number><suffix>"` (or a bare numeric string) into global base units.
pub fn parse_literal(s: &str) -> Option<(f64, Dim)> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return v.is_finite().then_some((v, Dim::NONE));
    }
    SUFFIXES
        .iter()
        .filter_map(|(suf, mul, div, d)| {
            let num = s.strip_suffix(suf)?;
            let v: f64 = num.trim_end().parse().ok()?;
            (v.is_finite() && !num.is_empty()).then_some((suf.len(), v * mul / div, *d))
        })
        .max_by_key(|(len, ..)| *len)
        .map(|(_, v, d)| (v, d))
}

/// Writes an expression result back as a JSON value: bare number when dimensionless, else a base-unit literal.
pub fn to_literal(v: f64, dim: Dim) -> Value {
    if dim.is_none() {
        return number_value(v);
    }
    match dim.base_suffix() {
        Some((suf, scale)) => Value::String(format!("{}{suf}", v / scale)),
        None => Value::String(format!("{v}[{dim}]")),
    }
}

pub fn number_value(v: f64) -> Value {
    if v.fract() == 0.0 && v.abs() < 9.007_199_254_740_992e15 {
        Value::from(v as i64)
    } else {
        serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
    }
}

pub trait Quantity: Sized {
    const DIM: Dim;
    const EXAMPLE: &'static str;
    /// Size of this type's base unit in global base units.
    const SCALE: f64 = 1.0;
    fn from_base(v: f64) -> Result<Self, Diagnostic>;

    fn from_u64(v: u64) -> Result<Self, Diagnostic> {
        Self::from_base(v as f64)
    }

    /// Exact parse that bypasses `f64` (integer quantities), when `s` admits one.
    fn parse_exact(_s: &str) -> Option<Result<Self, Diagnostic>> {
        None
    }

    fn convert(v: f64, dim: Dim) -> Option<f64> {
        (dim == Self::DIM || dim.is_none()).then_some(if dim.is_none() { v } else { v / Self::SCALE })
    }

    fn parse_str(s: &str) -> Result<Self, Diagnostic> {
        let mismatch = |what: &str| {
            Diagnostic::error("E-IR-0108", format!("{what} {s:?}: expected {} (e.g. {:?})", Self::DIM, Self::EXAMPLE))
                .hint(format!("write a {} quantity such as {:?} or a bare number in base units", Self::DIM, Self::EXAMPLE))
        };
        if let Some(r) = Self::parse_exact(s) {
            return r;
        }
        let (v, dim) = parse_literal(s).ok_or_else(|| mismatch("unrecognized quantity"))?;
        Self::from_base(Self::convert(v, dim).ok_or_else(|| mismatch("quantity has the wrong dimension:"))?)
    }
}

struct QVisitor<T>(std::marker::PhantomData<T>);

impl<T: Quantity> Visitor<'_> for QVisitor<T> {
    type Value = T;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a number in base units or a quantity string like {:?}", T::EXAMPLE)
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<T, E> {
        T::from_u64(v).map_err(de_error)
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<T, E> {
        u64::try_from(v).map_or_else(|_| T::from_base(v as f64), T::from_u64).map_err(de_error)
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<T, E> {
        T::from_base(v).map_err(de_error)
    }
    fn visit_str<E: de::Error>(self, s: &str) -> Result<T, E> {
        T::parse_str(s).map_err(de_error)
    }
}

macro_rules! float_quantity {
    ($(#[$m:meta])* $name:ident, $dim:expr, $ex:literal $(, scale = $scale:expr)?) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize)]
        #[serde(transparent)]
        pub struct $name(pub f64);

        impl Quantity for $name {
            const DIM: Dim = $dim;
            const EXAMPLE: &'static str = $ex;
            $(const SCALE: f64 = $scale;)?
            fn from_base(v: f64) -> Result<Self, Diagnostic> {
                if v.is_finite() {
                    Ok(Self(if v == 0.0 { 0.0 } else { v }))
                } else {
                    Err(Diagnostic::error("E-IR-0103", format!("non-finite {} value", stringify!($name))))
                }
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                d.deserialize_any(QVisitor::<Self>(std::marker::PhantomData))
            }
        }

        impl Mul<f64> for $name {
            type Output = $name;
            fn mul(self, k: f64) -> $name {
                $name(self.0 * k)
            }
        }
    };
}

float_quantity!(Hz, Dim::HZ, "1410MHz");
float_quantity!(BytesPerSec, Dim::BYTES_PER_S, "1555GB/s");
float_quantity!(
    /// Only in fields named `*_bits_per_s`.
    BitsPerSec,
    Dim::BITS_PER_S,
    "2.43Gbps"
);
float_quantity!(Seconds, Dim::SECONDS, "12ns");
float_quantity!(Joules, Dim::JOULES, "3.9pJ");
float_quantity!(Watts, Dim::WATTS, "400W");
float_quantity!(Volts, Dim::VOLTS, "0.75V");
float_quantity!(Um, Dim::UM, "33mm");
float_quantity!(Mm2, Dim::UM2, "826mm2", scale = 1e6);
float_quantity!(
    /// Cycles of the owning entity's clock.
    Cycles,
    Dim::NONE,
    "4cyc"
);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize)]
#[serde(transparent)]
pub struct JoulesPerByte(pub f64);

impl Quantity for JoulesPerByte {
    const DIM: Dim = Dim::J_PER_BYTE;
    const EXAMPLE: &'static str = "3.9pJ/b";
    fn from_base(v: f64) -> Result<Self, Diagnostic> {
        Ok(Self(v))
    }
    fn convert(v: f64, dim: Dim) -> Option<f64> {
        match dim {
            Dim::J_PER_BYTE => Some(v),
            Dim::J_PER_BIT => Some(v * 8.0),
            d if d.is_none() => Some(v),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for JoulesPerByte {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(QVisitor::<Self>(std::marker::PhantomData))
    }
}

macro_rules! int_quantity {
    ($(#[$m:meta])* $name:ident, $dim:expr, $ex:literal) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize)]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl Quantity for $name {
            const DIM: Dim = $dim;
            const EXAMPLE: &'static str = $ex;
            /// Floats must be whole and strictly below 2^64; integer literals take the exact path instead.
            fn from_base(v: f64) -> Result<Self, Diagnostic> {
                if v >= 0.0 && v.fract() == 0.0 && v < TWO_64 {
                    Ok(Self(v as u64))
                } else {
                    Err(not_whole(v, Self::DIM))
                }
            }
            fn from_u64(v: u64) -> Result<Self, Diagnostic> {
                Ok(Self(v))
            }
            fn parse_exact(s: &str) -> Option<Result<Self, Diagnostic>> {
                exact_int(s, Self::DIM).map(|v| v.map(Self).ok_or_else(|| not_whole_str(s, Self::DIM)))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                d.deserialize_any(QVisitor::<Self>(std::marker::PhantomData))
            }
        }
    };
}

const TWO_64: f64 = 18_446_744_073_709_551_616.0;

fn not_whole(v: impl fmt::Display, dim: Dim) -> Diagnostic {
    Diagnostic::error("E-IR-0110", format!("{v} is not a non-negative whole number of {dim} below 2^64"))
        .hint("byte and bit quantities must convert to an integer (\"1.5KiB\" is fine, \"0.3B\" is not)")
}

fn not_whole_str(s: &str, dim: Dim) -> Diagnostic {
    not_whole(format!("{s:?}"), dim)
}

/// `"<digits>"` or `"<digits><integer suffix of dim>"` in base units without going through `f64`: `Some(None)`
/// when it overflows `u64`, `None` when `s` is not of that form.
fn exact_int(s: &str, dim: Dim) -> Option<Option<u64>> {
    let s = s.trim();
    let digits = |n: &str| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit());
    let (num, mul) = if digits(s) {
        (s, 1.0)
    } else {
        SUFFIXES.iter().filter(|(_, _, div, d)| *d == dim && *div == 1.0).find_map(|(suf, mul, ..)| {
            let n = s.strip_suffix(suf)?.trim_end();
            digits(n).then_some((n, *mul))
        })?
    };
    Some(num.parse::<u128>().ok().and_then(|n| n.checked_mul(mul as u128)).and_then(|n| u64::try_from(n).ok()))
}

int_quantity!(Bytes, Dim::BYTES, "40MiB");
int_quantity!(
    /// Only in fields named `*_bits`.
    Bits,
    Dim::BITS,
    "1024"
);

impl Bytes {
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

impl Mul<Hz> for Bytes {
    type Output = BytesPerSec;
    fn mul(self, f: Hz) -> BytesPerSec {
        BytesPerSec(self.0 as f64 * f.0)
    }
}

impl Div<Hz> for BytesPerSec {
    type Output = f64;
    fn div(self, f: Hz) -> f64 {
        self.0 / f.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn de<T: for<'a> Deserialize<'a>>(v: Value) -> Result<T, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }

    #[test]
    fn literals_convert_to_base_units() {
        assert_eq!(de::<Bytes>("40MiB".into()).unwrap(), Bytes(40 << 20));
        assert_eq!(de::<Bytes>("1.5KiB".into()).unwrap(), Bytes(1536));
        assert_eq!(de::<Bytes>(Value::from(4096)).unwrap(), Bytes(4096));
        assert_eq!(de::<Hz>("1410MHz".into()).unwrap(), Hz(1.41e9));
        assert_eq!(de::<BytesPerSec>("1555GB/s".into()).unwrap(), BytesPerSec(1.555e12));
        assert_eq!(de::<BitsPerSec>("2.43Gbps".into()).unwrap(), BitsPerSec(2.43e9));
        assert_eq!(de::<JoulesPerByte>("3.9pJ/b".into()).unwrap().0, 3.9e-12 * 8.0);
        assert_eq!(de::<Mm2>("826mm2".into()).unwrap(), Mm2(826.0));
        assert_eq!(de::<Um>("33mm".into()).unwrap(), Um(33000.0));
        assert_eq!(de::<Cycles>("4cyc".into()).unwrap(), Cycles(4.0));
        assert_eq!(de::<Seconds>("12ns".into()).unwrap().0, 12e-9);
    }

    #[test]
    fn dimension_and_integer_errors() {
        assert!(de::<Bytes>("1.4GHz".into()).unwrap_err().contains("E-IR-0108"));
        assert!(de::<Bytes>("0.3B".into()).unwrap_err().contains("E-IR-0110"));
        assert!(de::<Hz>("fast".into()).unwrap_err().contains("E-IR-0108"));
    }

    #[test]
    fn integer_quantities_are_exact_and_bounded() {
        assert_eq!(de::<Bytes>(Value::from(9_007_199_254_740_993u64)).unwrap(), Bytes(9_007_199_254_740_993));
        assert_eq!(de::<Bytes>(Value::from(u64::MAX)).unwrap(), Bytes(u64::MAX));
        assert_eq!(de::<Bytes>("9007199254740993B".into()).unwrap(), Bytes(9_007_199_254_740_993));
        assert_eq!(de::<Bytes>("9007199254740993".into()).unwrap(), Bytes(9_007_199_254_740_993));
        assert_eq!(de::<Bytes>("18446744073709551615B".into()).unwrap(), Bytes(u64::MAX));
        assert_eq!(de::<Bits>("3Kib".into()).unwrap(), Bits(3072));
        for s in ["18446744073709551616B", "18446744073709551616", "16777216TiB", "1.8446744073709552e19B"] {
            assert!(de::<Bytes>(s.into()).unwrap_err().contains("E-IR-0110"), "{s}");
        }
        assert!(de::<Bytes>(Value::from(1.8446744073709552e19)).unwrap_err().contains("E-IR-0110"));
        assert!(de::<Bytes>(Value::from(-1)).unwrap_err().contains("E-IR-0110"));
    }

    #[test]
    fn literal_round_trip() {
        assert_eq!(to_literal(1.5e9, Dim::HZ), Value::from("1500000000Hz"));
        assert_eq!(to_literal(826e6, Dim::UM2), Value::from("826mm2"));
        assert_eq!(to_literal(4.0, Dim::NONE), Value::from(4));
        let Value::String(s) = to_literal(1.41e9 * 4.0, Dim::HZ) else { panic!() };
        assert!(de::<Bytes>(s.into()).unwrap_err().contains("E-IR-0108"));
    }
}
