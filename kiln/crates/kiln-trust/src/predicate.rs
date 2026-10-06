//! T2 expected responses (06 §12.2): what a whole-step result must do after a transform.

/// The parts of a `kiln.result/1` phase (06 §6.3) that the T2 relations read.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub time_s: f64,
    /// Time attributed to each bound kind (`bound_breakdown` fraction x `time_s`); sums to `time_s`.
    pub t_offchip: f64,
    pub t_compute: f64,
    pub energy_j: f64,
    pub area_mm2: f64,
    pub static_power_w: f64,
    /// The phase's full `bound_breakdown` (fractions of `time_s`), for reporting.
    pub bound_breakdown: std::collections::BTreeMap<String, f64>,
}

impl Outcome {
    pub fn t_other(&self) -> f64 {
        self.time_s - self.t_offchip - self.t_compute
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub pass: bool,
    pub expected: f64,
    pub got: f64,
    pub detail: String,
}

fn rel(got: f64, expected: f64, tol: f64, what: &str) -> Verdict {
    let err = (got / expected - 1.0).abs();
    Verdict { pass: err <= tol, expected, got, detail: format!("{what}: rel err {err:.4} vs tol {tol}") }
}

/// M1 / M3 / M8: `time' = T_other + T_part * k` where `T_part` is the scaled bound's share before the transform.
pub fn scaled_part(base: &Outcome, new: &Outcome, part: f64, k: f64, tol: f64) -> Verdict {
    rel(new.time_s, base.time_s - part + part * k, tol, "time vs T_other + k T_part")
}

/// M2: compute scaling on a memory-bound step leaves time within `[lo, hi]` of the original.
pub fn ratio_in(base: &Outcome, new: &Outcome, lo: f64, hi: f64) -> Verdict {
    let r = new.time_s / base.time_s;
    Verdict { pass: (lo..=hi).contains(&r), expected: hi, got: r, detail: format!("time ratio {r:.4} in [{lo}, {hi}]") }
}

/// M5: an idle unit never speeds the step up and strictly costs area and static power.
pub fn idle_unit(base: &Outcome, new: &Outcome) -> Verdict {
    let pass = new.time_s >= base.time_s && new.area_mm2 > base.area_mm2 && new.static_power_w > base.static_power_w;
    Verdict {
        pass,
        expected: base.time_s,
        got: new.time_s,
        detail: format!(
            "time {} -> {}, area {} -> {}, static {} -> {}",
            base.time_s, new.time_s, base.area_mm2, new.area_mm2, base.static_power_w, new.static_power_w
        ),
    }
}

/// M6: bit-identical numbers.
pub fn identical(base: &Outcome, new: &Outcome) -> Verdict {
    let fields = [
        ("time", base.time_s, new.time_s),
        ("energy", base.energy_j, new.energy_j),
        ("t_offchip", base.t_offchip, new.t_offchip),
        ("t_compute", base.t_compute, new.t_compute),
        ("area", base.area_mm2, new.area_mm2),
    ];
    let diff: Vec<String> =
        fields.iter().filter(|(_, a, b)| a.to_bits() != b.to_bits()).map(|(k, a, b)| format!("{k} {a:e} -> {b:e}")).collect();
    let detail = if diff.is_empty() { "bit-identical time, energy, bounds, area".into() } else { format!("differs: {}", diff.join(", ")) };
    Verdict { pass: diff.is_empty(), expected: base.time_s, got: new.time_s, detail }
}

/// M7 with an ideal wire model and fixed placement: time and energy unchanged to `tol` relative.
pub fn unchanged(base: &Outcome, new: &Outcome, tol: f64) -> Verdict {
    let t = rel(new.time_s, base.time_s, tol, "time");
    let e = rel(new.energy_j, base.energy_j, tol, "energy");
    Verdict { pass: t.pass && e.pass, expected: base.time_s, got: new.time_s, detail: format!("{}; {}", t.detail, e.detail) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(time: f64, off: f64, comp: f64) -> Outcome {
        Outcome {
            time_s: time,
            t_offchip: off,
            t_compute: comp,
            energy_j: 1.0,
            area_mm2: 100.0,
            static_power_w: 10.0,
            bound_breakdown: Default::default(),
        }
    }

    #[test]
    fn scaled_part_matches_closed_form() {
        let base = o(10.0, 8.0, 1.0);
        assert!(scaled_part(&base, &o(6.0, 4.0, 1.0), base.t_offchip, 0.5, 0.05).pass);
        assert!(!scaled_part(&base, &o(10.0, 8.0, 1.0), base.t_offchip, 0.5, 0.05).pass);
        assert!(scaled_part(&base, &o(6.25, 4.0, 1.0), base.t_offchip, 0.5, 0.05).pass);
        assert!(!scaled_part(&base, &o(6.4, 4.0, 1.0), base.t_offchip, 0.5, 0.05).pass);
        assert_eq!(base.t_other(), 1.0);
    }

    #[test]
    fn ratio_idle_identical_unchanged() {
        let base = o(10.0, 9.0, 0.5);
        assert!(ratio_in(&base, &o(9.8, 9.0, 0.25), 0.97, 1.0).pass);
        assert!(!ratio_in(&base, &o(9.0, 9.0, 0.25), 0.97, 1.0).pass);
        let mut idle = base.clone();
        idle.area_mm2 += 1.0;
        idle.static_power_w += 0.1;
        assert!(idle_unit(&base, &idle).pass);
        idle.time_s -= 1e-12;
        assert!(!idle_unit(&base, &idle).pass);
        assert!(!idle_unit(&base, &base).pass, "free area/power is a failure");
        assert!(identical(&base, &base.clone()).pass);
        let mut b = base.clone();
        b.time_s = f64::from_bits(b.time_s.to_bits() + 1);
        assert!(!identical(&base, &b).pass);
        assert!(unchanged(&base, &b, 1e-9).pass);
    }
}
