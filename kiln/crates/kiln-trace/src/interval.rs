use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};

/// `{low, central, high}` ordered numerically (`low <= central <= high`), 03 §9.1, 06 §6.3.
///
/// Corners are named by throughput: the `Low` corner yields `high` time, energy, power and area.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub low: f64,
    pub central: f64,
    pub high: f64,
}

impl Interval {
    pub fn new(low: f64, central: f64, high: f64) -> Result<Self, Diagnostic> {
        let i = Self { low, central, high };
        if i.is_valid() {
            Ok(i)
        } else {
            Err(Diagnostic::error(
                "E-TRACE-INTERVAL",
                format!("invalid interval [{low}, {central}, {high}]"),
            )
            .hint("intervals must be finite with low <= central <= high"))
        }
    }

    pub const fn point(x: f64) -> Self {
        Self {
            low: x,
            central: x,
            high: x,
        }
    }

    /// Orders values from the three corner evaluations into an interval. A NaN corner yields NaN ends, so the
    /// failed evaluation stays visible to [`Interval::is_valid`].
    pub fn from_corners(central: f64, low_corner: f64, high_corner: f64) -> Self {
        if [central, low_corner, high_corner]
            .iter()
            .any(|x| x.is_nan())
        {
            return Self {
                low: f64::NAN,
                central,
                high: f64::NAN,
            };
        }
        Self {
            low: central.min(low_corner).min(high_corner),
            central,
            high: central.max(low_corner).max(high_corner),
        }
    }

    pub fn is_valid(&self) -> bool {
        [self.low, self.central, self.high]
            .iter()
            .all(|x| x.is_finite())
            && self.low <= self.central
            && self.central <= self.high
    }

    pub fn width(&self) -> f64 {
        self.high - self.low
    }

    /// `(high - low) / central`, 0 for a zero-centred interval (06 §6.3 `score_rel_width`).
    pub fn rel_width(&self) -> f64 {
        if self.central == 0.0 {
            0.0
        } else {
            self.width() / self.central.abs()
        }
    }

    pub fn contains(&self, x: f64) -> bool {
        self.low <= x && x <= self.high
    }

    /// `k / self`, e.g. tokens/s from a time interval; ends swap.
    pub fn recip_scaled(&self, k: f64) -> Self {
        Self {
            low: k / self.high,
            central: k / self.central,
            high: k / self.low,
        }
    }

    pub fn scaled(&self, k: f64) -> Self {
        Self::from_corners(self.central * k, self.low * k, self.high * k)
    }

    pub fn value(&self, corner: Corner) -> f64 {
        match corner {
            Corner::Low => self.low,
            Corner::Central => self.central,
            Corner::High => self.high,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Corner {
    Central,
    Low,
    High,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalMethod {
    None,
    #[default]
    Sensitivity,
    Corners,
    Sampled,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rejects_misordered_and_nonfinite() {
        assert!(Interval::new(1.0, 2.0, 3.0).is_ok());
        assert_eq!(
            Interval::new(2.0, 1.0, 3.0).unwrap_err().code,
            "E-TRACE-INTERVAL"
        );
        assert!(Interval::new(f64::NAN, 1.0, 1.0).is_err());
    }

    #[test]
    fn nan_corner_is_invalid() {
        assert!(!Interval::from_corners(2.0, f64::NAN, 3.0).is_valid());
        assert!(!Interval::from_corners(f64::NAN, 1.0, 3.0).is_valid());
        assert!(!Interval::from_corners(2.0, 1.0, f64::NAN).is_valid());
    }

    #[test]
    fn recip_swaps_ends() {
        let t = Interval::new(1.0, 2.0, 4.0).unwrap();
        assert_eq!(
            t.recip_scaled(8.0),
            Interval {
                low: 2.0,
                central: 4.0,
                high: 8.0
            }
        );
        assert_eq!(t.rel_width(), 1.5);
    }

    proptest! {
        #[test]
        fn corners_always_valid_and_round_trip(c in -1e12f64..1e12, a in -1e12f64..1e12, b in -1e12f64..1e12) {
            let i = Interval::from_corners(c, a, b);
            prop_assert!(i.is_valid());
            prop_assert!(i.contains(a) && i.contains(b) && i.contains(c));
            let back: Interval = serde_json::from_str(&serde_json::to_string(&i).unwrap()).unwrap();
            prop_assert_eq!(back, i);
        }
    }
}
