//! Block-level area/energy models (04 §4.2-§4.4, §4.11, §7.4): datapath lanes, SRAM macros, register files,
//! routers and PHYs. Values here are structural (kappa = 1); calibrated factors are applied by `characterize`.

use kiln_ir::precision::{Precision, PrecisionKind};
use serde::Serialize;

use crate::tables::{Datapath, TechNode, WireClassId};

/// One datapath lane at 45 nm, 0.9 V (Horowitz anchors), before kappa factors and node scaling.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lane {
    pub area_um2: f64,
    pub energy_pj: f64,
    /// Operand/pipeline register bits (clock-tree load).
    pub ff_bits: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Fmt {
    m: f64,
    e: f64,
    float: bool,
    mx: bool,
}

fn fmt(dp: &Datapath, p: Precision) -> Fmt {
    let p = p.compute();
    let name = p.name();
    let base = name.strip_prefix("mx").unwrap_or(name);
    let key = match base {
        "fp4" => "fp4_e2m1",
        "fp8_e4m3" | "fp8_e5m2" | "fp6_e3m2" | "fp6_e2m3" => base,
        "nvfp4" => "fp4_e2m1",
        _ => base,
    };
    let mx = matches!(p.kind(), PrecisionKind::Mx | PrecisionKind::BlockFloat);
    match dp.formats.get(key) {
        Some(&(m, e)) if p.is_float() || mx && !base.starts_with("int") => Fmt { m, e, float: true, mx },
        _ => Fmt { m: f64::from(p.element_bits()), e: 0.0, float: false, mx },
    }
}

fn clog2(x: f64) -> f64 {
    x.max(1.0).log2().ceil()
}

/// fp multiplier anchored to Horowitz fp16 (11x11 significands), scaled by the significand product.
fn fp_mult(dp: &Datapath, ma: f64, mb: f64) -> (f64, f64) {
    let (a16, e16) = dp.anchor("fp16_mult");
    let r = (ma * mb / 121.0).powf(dp.c("fp_mult_exp"));
    (a16 * r, e16 * r)
}

/// fp adder of total width `bits`, anchored to Horowitz fp16/fp32 adds.
fn fp_add(dp: &Datapath, bits: f64) -> (f64, f64) {
    let (a16, e16) = dp.anchor("fp16_add");
    (a16 * (bits / 16.0).powf(dp.c("fp_add_exp")), e16 * (bits / 16.0).powf(dp.c("fp_add_e_exp")))
}

/// 04 §4.2 structural dot-product lane for `a x b -> acc` at dot length `l` with `extra_ff` holding-register bits
/// (dataflow modes, §4.9). The fp multiplier and fp accumulator are anchored to the Horowitz fp rows (the bit-level
/// partial-product form underestimates rounding/normalization logic by ~3x for fp); integer terms are per bit.
pub fn mac_lane(dp: &Datapath, a: Precision, b: Precision, acc: Precision, l: f64, extra_ff: f64) -> Lane {
    let (fa, fb, fc) = (fmt(dp, a), fmt(dp, b), fmt(dp, acc));
    let l = l.max(1.0);
    let float = fa.float || fb.float;
    let (a_add, e_add) = (dp.c("a_add_um2"), dp.c("e_add_fj") * 1e-3);
    let (a_mux, e_mux) = (dp.c("a_mux_um2"), dp.c("e_mux_fj") * 1e-3);
    let (a_ff, e_ff) = (dp.c("a_ff_um2"), dp.c("e_ff_fj") * 1e-3);
    let g = dp.c("guard_bits");
    let mut area = 0.0;
    let mut energy = 0.0;
    let w_prod;
    if float {
        let (am, em) = fp_mult(dp, fa.m, fb.m);
        area += am;
        energy += em;
        let emax = fa.e.max(fb.e);
        area += a_add * (emax + 1.0);
        energy += e_add * (emax + 1.0);
        w_prod = fa.m + fb.m + g;
        area += a_mux * w_prod * clog2(w_prod);
        energy += e_mux * w_prod * clog2(w_prod);
    } else {
        let a_pp = if fa.m.max(fb.m) <= 12.0 { dp.c("a_pp_small_um2") } else { dp.c("a_pp_large_um2") };
        area += a_pp * fa.m * fb.m;
        energy += dp.c("e_pp_fj") * 1e-3 * fa.m * fb.m;
        w_prod = fa.m + fb.m;
    }
    let tree = (w_prod + clog2(l)) * (l - 1.0) / l;
    area += a_add * tree;
    energy += e_add * tree;
    let w_acc = if fc.float { fc.m + fc.e } else { fc.m };
    if fc.float {
        let (aa, ea) = fp_add(dp, w_acc);
        area += aa / l;
        energy += ea / l;
        if float {
            area += dp.c("a_cmp_um2") * fa.e.max(fb.e);
        }
    } else {
        area += a_add * w_acc / l;
        energy += e_add * w_acc / l;
    }
    let ff = f64::from(a.element_bits()) + f64::from(b.element_bits()) + w_acc / l + extra_ff;
    area += a_ff * ff;
    energy += e_ff * ff;
    if fa.mx || fb.mx {
        area += a_add * 9.0 / dp.c("mx_block");
        energy += e_add * 9.0 / dp.c("mx_block");
    }
    Lane { area_um2: area, energy_pj: energy * dp.c("alpha_dp"), ff_bits: ff }
}

/// Elementwise FMA lane of `dtype` (vector units): the DPU model with L = 1, accumulate in `dtype`.
pub fn fma_lane(dp: &Datapath, dtype: Precision) -> Lane {
    let acc = if dtype.is_float() { dtype } else { Precision::Int32 };
    mac_lane(dp, dtype, dtype, acc, 1.0, 0.0)
}

/// Node scaling of a 45 nm lane: (area um2, energy J at V_nom).
pub fn scale(dp: &Datapath, n: &TechNode, lane: Lane) -> (f64, f64) {
    let a = lane.area_um2 * n.a_ge_um2 / dp.c("a_ge45_um2");
    let e = lane.energy_pj * 1e-12 * dp.c("s_45_n7") * n.s_e;
    (a, e)
}

/// Bits per cycle a 64-bit Horowitz SRAM read moves (anchor of the per-bit SRAM energy).
const E_S0_PJ_PER_BIT_45: f64 = 0.156;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sram {
    pub area_um2: f64,
    pub eta: f64,
    /// Read/write energy per bit at V_nom, kappa = 1 (J).
    pub e_read_bit: f64,
    pub e_write_bit: f64,
    pub t_acc_s: f64,
    pub leak_w: f64,
    pub bits: f64,
}

/// Port mix of one bank.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ports {
    pub read: u32,
    pub write: u32,
    pub rw: u32,
}

impl Ports {
    pub fn total(&self) -> u32 {
        (self.read + self.write + self.rw).max(1)
    }

    /// Cell area factor (04 §4.3): 1RW 1.0, 1R1W (8T) 1.4, 2 ports 2.0, more ports (1 + 0.414 (P - 1))^2.
    pub fn f_port(&self) -> f64 {
        match (self.total(), self.read, self.write) {
            (1, _, _) => 1.0,
            (2, 1, 1) => 1.4,
            (p, _, _) => (1.0 + 0.414 * f64::from(p - 1)).powi(2),
        }
    }

    pub fn f_port_e(&self) -> f64 {
        match (self.total(), self.read, self.write) {
            (1, _, _) => 1.0,
            (2, 1, 1) => 1.15,
            (p, _, _) => 1.0 + 0.3 * f64::from(p - 1),
        }
    }
}

/// Repeated-wire energy per bit per um at energy-aware sizing (c_w + c_rep ~ 1.35 c), toggle factor 0.25, at V.
pub fn wire_e_bit_per_um(n: &TechNode, class: WireClassId, v: f64) -> f64 {
    0.25 * 1.35 * n.wire(class).c_ff_um * 1e-15 * v * v
}

/// SRAM macro (04 §4.3): `banks` banks of `bits / banks`, bank word `word_bits`, `ports` per bank, cell factor
/// `cell_k` (HC / eDRAM / MRAM cells).
pub fn sram(n: &TechNode, dp: &Datapath, bits: f64, word_bits: f64, banks: u32, ports: Ports, cell_k: f64) -> Sram {
    let banks = banks.max(1);
    let fp = ports.f_port();
    let cell = n.cell_um2 * fp * cell_k;
    let h_c = (cell / n.cell_aspect).sqrt();
    let w_c = n.cell_aspect * h_c;
    let cb = (bits / f64::from(banks)).max(1.0);
    let w = word_bits.clamp(1.0, cb);
    let mut r = 512.0f64;
    while r > 16.0 && r * w > cb {
        r /= 2.0;
    }
    let mut cc = (cb / r).max(w);
    let n_sub = (cc / 1024.0).ceil().max(1.0);
    cc /= n_sub;
    let a_sub = (r * h_c + n.n_hp * h_c) * (cc * w_c + n.n_wp * w_c);
    let a_bank = n_sub * a_sub * (1.0 + n.g_ctrl);
    let area = f64::from(banks) * a_bank + n.a_fix_um2 * (1.0 + 0.25 * f64::from(banks - 1));
    let side = area.sqrt();
    let e_wire = wire_e_bit_per_um(n, WireClassId::Intermediate, n.vdd_nom) * 0.5 * (a_bank.sqrt());
    let e_read = E_S0_PJ_PER_BIT_45 * 1e-12 * dp.c("s_45_n7") * n.s_e * (cb / 65536.0).sqrt() * ports.f_port_e() + e_wire;
    let tau = crate::wire::tau_s_per_um(n, WireClassId::Intermediate, crate::wire::Sizing::Energy);
    let t_acc = (n.t_s0_ns + n.t_s1_ns * cb.log2()) * 1e-9 + tau * 0.5 * side;
    Sram {
        area_um2: area,
        eta: bits * n.cell_um2 / area,
        e_read_bit: e_read,
        e_write_bit: 1.1 * e_read,
        t_acc_s: t_acc,
        leak_w: n.p_ls_mw_per_mib * 1e-3 * bits / (1u64 << 20) as f64 * fp * cell_k.min(1.0),
        bits,
    }
}

/// Flop or latch register array (04 §4.4).
pub fn flop_array(n: &TechNode, bits: f64, word_bits: f64, ports: Ports, latch: bool) -> Sram {
    let p = f64::from(ports.total());
    let a_bit = if latch { n.a_bit_latch_um2 } else { n.a_bit_flop_um2 } * (1.0 + n.k_w * (p - 1.0)).powi(2);
    let entries = (bits / word_bits.max(1.0)).max(1.0);
    let e = n.e_rf0_fj * 1e-15 * (1.0 + 0.3 * (p - 1.0)) * (entries / 64.0).sqrt().max(0.25);
    Sram {
        area_um2: a_bit * bits,
        eta: 1.0,
        e_read_bit: e,
        e_write_bit: 1.2 * e,
        t_acc_s: 0.0,
        leak_w: 0.0,
        bits,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Router {
    pub area_um2: f64,
    /// Logic part of `area_um2` (buffers + allocators) for leakage.
    pub logic_um2: f64,
    pub e_flit_j: f64,
    pub flit_bits: f64,
    pub n_pipe: u32,
    pub t_cycle_min_s: f64,
    pub ff_bits: f64,
}

pub const P_FLAT: u32 = 8;

/// Routing layers per direction available to a crossbar's buses (intermediate + semi-global pairs).
fn layers_per_dir(n: &TechNode) -> f64 {
    ((n.wire(WireClassId::Intermediate).layers + n.wire(WireClassId::SemiGlobal).layers) / 2.0).max(1.0)
}

/// NoC router (04 §7.4, ORION/DSENT structure). The matrix crossbar is wire-limited with its buses stacked over
/// the intermediate and semi-global layers of one direction; a radix above [`P_FLAT`] is built as `ceil(P / 8)`
/// radix-8 tiles whose outputs merge over the block (the flat matrix for P > 8 grows as P^2 W^2).
pub fn router(n: &TechNode, p: u32, v: u32, b: u32, w: f64, n_pipe: u32, util: f64) -> Router {
    let p = p.max(2);
    let pv = f64::from(p * v.max(1));
    let a_bit = n.a_bit_flop_um2 * (1.0 + n.k_w).powi(2);
    let buf_bits = f64::from(p) * f64::from(v.max(1)) * f64::from(b.max(1)) * w;
    let a_buf = buf_bits * a_bit;
    let p_x = 2.0 * n.wire(WireClassId::SemiGlobal).pitch_nm * 1e-3;
    let tile = |k: u32| (f64::from(k) * w * p_x / layers_per_dir(n)).powi(2);
    let a_xbar = if p <= P_FLAT { tile(p) } else { f64::from(p.div_ceil(P_FLAT)) * tile(P_FLAT) };
    let a_alloc = 12.0 * pv * pv * n.a_ge_um2;
    let logic = (a_buf + a_alloc) / util;
    let line = f64::from(p.min(P_FLAT)) * w * p_x / layers_per_dir(n);
    let e_buf = 2.2 * n.e_rf0_fj * 1e-15 * w;
    let e_xbar = w * 0.25 * n.wire(WireClassId::SemiGlobal).c_ff_um * 1e-15 * line * 2.0 * n.vdd_nom * n.vdd_nom;
    let e_arb = 50.0 * pv.log2().max(1.0) * n.e_rf0_fj * 1e-15;
    let fo4 = 3.45 * n.r0c0_ps * 1e-12;
    Router {
        area_um2: logic + a_xbar,
        logic_um2: logic,
        e_flit_j: e_buf + e_xbar + e_arb,
        flit_bits: w,
        n_pipe,
        t_cycle_min_s: (8.0 + 3.0 * pv.log2()) * fo4,
        ff_bits: buf_bits,
    }
}

/// Shared bus hub (01 `bus` topology): a `p`-input mux tree of `w` bits plus a round-robin arbiter; no buffers.
pub fn bus(n: &TechNode, p: u32, w: f64, util: f64) -> Router {
    let p = p.max(2);
    let mux_ge = 2.3 * w * f64::from(p - 1) + 50.0 * f64::from(p);
    let logic = mux_ge * n.a_ge_um2 / util;
    Router {
        area_um2: logic,
        logic_um2: logic,
        e_flit_j: w * f64::from(p).log2().ceil() * 0.5 * n.e_rf0_fj * 1e-15,
        flit_bits: w,
        n_pipe: 1,
        t_cycle_min_s: 0.0,
        ff_bits: w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::Tables;

    fn lane(a: Precision, b: Precision, c: Precision, l: f64) -> Lane {
        mac_lane(&Tables::get().datapath, a, b, c, l, 0.0)
    }

    #[test]
    fn format_ordering_matches_04_4_2() {
        use Precision::*;
        let i8 = lane(Int8, Int8, Int32, 8.0);
        let bf = lane(Bf16, Bf16, Fp32, 8.0);
        let fp16 = lane(Fp16, Fp16, Fp32, 8.0);
        let fp8 = lane(Fp8E4m3, Fp8E4m3, Fp32, 8.0);
        assert!(i8.area_um2 < bf.area_um2 && bf.area_um2 < fp16.area_um2, "{i8:?} {bf:?} {fp16:?}");
        assert!(i8.energy_pj < bf.energy_pj && bf.energy_pj < fp16.energy_pj);
        let r = fp8.energy_pj / bf.energy_pj;
        assert!((0.3..0.75).contains(&r), "fp8/bf16 energy {r}");
        let mx4 = lane(Mxfp4, Mxfp4, Fp32, 8.0);
        let (am, _) = fp_mult(&Tables::get().datapath, 2.0, 2.0);
        assert!(am < 0.25 * mx4.area_um2, "MXFP4 lane dominated by alignment/tree/accumulate, not the multiplier");
    }

    #[test]
    fn n7_lane_sanity() {
        let t = Tables::get();
        let n7 = t.node("tsmc_n7").unwrap();
        let (a, e) = scale(&t.datapath, n7, lane(Precision::Bf16, Precision::Bf16, Precision::Fp32, 8.0));
        assert!((20.0..200.0).contains(&a), "bf16 lane {a} um2");
        assert!((0.02e-12..0.2e-12).contains(&e), "bf16 lane {e} J");
    }

    #[test]
    fn sram_shape_and_scaling() {
        let t = Tables::get();
        let n7 = t.node("tsmc_n7").unwrap();
        let big = sram(n7, &t.datapath, 8.0 * 1048576.0 * 8.0, 512.0, 16, Ports { rw: 1, ..Default::default() }, 1.0);
        assert!((0.5..0.95).contains(&big.eta), "large macro efficiency {}", big.eta);
        let small = sram(n7, &t.datapath, 4096.0, 64.0, 1, Ports { rw: 1, ..Default::default() }, 1.0);
        assert!(small.eta < 0.4, "small macro periphery-dominated {}", small.eta);
        let k8 = sram(n7, &t.datapath, 65536.0, 64.0, 1, Ports { rw: 1, ..Default::default() }, 1.0);
        let k256 = sram(n7, &t.datapath, 2097152.0, 64.0, 1, Ports { rw: 1, ..Default::default() }, 1.0);
        assert!(k256.e_read_bit > 3.0 * k8.e_read_bit && k256.t_acc_s > k8.t_acc_s);
        let two = sram(n7, &t.datapath, 65536.0, 64.0, 1, Ports { read: 1, write: 1, rw: 0 }, 1.0);
        assert!(two.area_um2 > k8.area_um2);
    }

    #[test]
    fn router_area_monotone_in_radix() {
        let t = Tables::get();
        let n7 = t.node("tsmc_n7").unwrap();
        let mut last = 0.0;
        for p in 2..64 {
            let r = router(n7, p, 2, 4, 256.0, 2, 0.65);
            assert!(r.area_um2 >= last, "radix {p}");
            last = r.area_um2;
        }
    }
}
