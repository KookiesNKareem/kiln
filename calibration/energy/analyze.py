"""Fit per-unit energies from an energy_bench run and compare with kiln.

Usage: python analyze.py <run.json> [ncu.log [ref_summary.json]] -> <run>_summary.json and <run>_table.md next to the run file.
"""
import csv
import io
import json
import statistics
import sys

# kiln-phys a100_sxm4_40gb at V_nom 0.75 V (Phys::new, working tree 2026-10-06; see README).
KILN = {
    "rf_read_pJ_per_B": 0.3290, "rf_write_pJ_per_B": 0.3619,
    "l1_smem_read_pJ_per_B": 0.1601, "l2_read_pJ_per_B": 0.8627, "hbm_pJ_per_B": 36.805,
    "mac_pJ": {"bf16": 0.1233, "fp16": 0.1821, "tf32": 0.2102, "int8": 0.0447},
    "fp32_elem_pJ": 0.7663,
    "board_idle_W": 69.07, "board_busy_noact_W": 156.83, "board_busy_fullact_noenergy_W": 250.44,
    "clock_tree_full_W": 117.70, "clock_tree_gated_W": 35.31, "control_W": 77.52, "dram_background_W": 7.5,
    "board_fixed_W": 8.88, "eta_vr": 0.90,
}
UNITS = {"ffma": "fp32_fma", "smem_read": "smem_byte", "l1_read": "l1_byte", "l2_read": "l2_byte",
         "hbm_read": "hbm_byte", "hbm_copy": "hbm_byte"}


def fit(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def spread(v):
    return {"mean": statistics.fmean(v), "min": min(v), "max": max(v), "sd": statistics.pstdev(v) if len(v) > 1 else 0.0,
            "n": len(v)}


def parse_ncu(path):
    out, cur = {}, None
    for line in open(path):
        if line.startswith("=== "):
            cur = line[4:].split()[0]
            out[cur] = {}
        elif cur and line.startswith('"0"'):
            r = next(csv.reader(io.StringIO(line)))
            try:
                out[cur][r[12]] = float(r[14].replace(",", ""))
            except ValueError:
                out[cur][r[12]] = r[14]
            out[cur]["_kernel"] = r[4][:80]
    return out


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    recs = d["records"]
    reps = sorted({r["rep"] for r in recs})
    get = lambda rep, b, lv: next((r for r in recs if r["rep"] == rep and r["bench"] == b and r["level"] == lv), None)
    S = {"source": path, "device": d["device"], "clock_lock": d.get("clock_lock"), "energy_counter": d.get("energy_counter"),
         "kiln": KILN, "baselines": {}, "components": {}}
    for b, lv in [("idle", "none"), ("spin_1thread", "full"), ("spin_nap", "full"), ("spin_busy", "half"), ("spin_busy", "full")]:
        rs = [get(rep, b, lv) for rep in reps]
        rs = [r for r in rs if r]
        if rs:
            S["baselines"][f"{b}_{lv}"] = {"power_w": spread([r["power_w"] for r in rs]),
                                           "sm_mhz": spread([r["sm_mhz_mean"] for r in rs]),
                                           "temp_c": spread([r["temp_c_end"] for r in rs])}
    p_active = {rep: get(rep, "spin_1thread", "full")["power_w"] for rep in reps}
    p_spin = {rep: get(rep, "spin_busy", "full")["power_w"] for rep in reps}

    benches = list(UNITS) + sorted({r["bench"] for r in recs if r["bench"].startswith("gemm_")})
    comp = S["components"]
    for b in benches:
        unit = UNITS.get(b, "mac")
        levels = [lv for lv in ["full", "half", "duty50", "duty25"] if any(r["bench"] == b and r["level"] == lv for r in recs)]
        if "full" not in levels:
            continue
        per_rep = []
        for rep in reps:
            L = {lv: get(rep, b, lv) for lv in levels}
            full = L["full"]
            P = lambda lv: L[lv]["power_w"]
            X = lambda lv: L[lv]["throughput"][unit]
            row = {"rep": rep,
                   "over_spin_pJ": (P("full") - p_spin[rep]) / X("full") * 1e12,
                   "over_active_idle_pJ": (P("full") - p_active[rep]) / X("full") * 1e12,
                   "duty_pJ": (P("full") - P("duty50")) / (X("full") - X("duty50")) * 1e12,
                   "levels": {lv: {"power_w": P(lv), "throughput": X(lv), "sm_mhz": L[lv]["sm_mhz_mean"],
                                   "sm_mhz_min": L[lv]["sm_mhz_min"], "temp_c": L[lv]["temp_c_end"],
                                   "clock_event_reasons": L[lv]["clock_event_reasons_union"]} for lv in levels}}
            if "half" in L:
                row["half_pJ"] = (P("full") - P("half")) / (X("full") - X("half")) * 1e12
            if "duty25" in L:
                row["duty50_vs_25_pJ"] = (P("duty50") - P("duty25")) / (X("duty50") - X("duty25")) * 1e12
            per_rep.append(row)
        c = {"unit": unit, "per_rep": per_rep}
        for k in ["over_spin_pJ", "over_active_idle_pJ", "duty_pJ", "half_pJ", "duty50_vs_25_pJ"]:
            if k in per_rep[0]:
                c[k] = spread([r[k] for r in per_rep])
        for lv in levels:
            c[f"{lv}_power_w"] = spread([r["levels"][lv]["power_w"] for r in per_rep])
            c[f"{lv}_throughput"] = spread([r["levels"][lv]["throughput"] for r in per_rep])
            c[f"{lv}_sm_mhz"] = spread([r["levels"][lv]["sm_mhz"] for r in per_rep])
        comp[b] = c

    if len(sys.argv) > 2:
        S["ncu"] = parse_ncu(sys.argv[2])
    ncu = S.get("ncu", {})

    e_ffma = comp["ffma"]["over_spin_pJ"]["mean"]
    for b in [x for x in ["smem_read", "l1_read"] if x in comp]:
        comp[b]["lop3_per_byte"] = 1 / 8
        comp[b]["alu_upper_bound_pJ_per_B"] = e_ffma / 8

    # cuBLAS GEMM decomposition with ncu traffic per call, at the over-spin per-byte energies measured above.
    if len(sys.argv) > 3:  # per-byte energies from another run (e.g. the full-suite run at the same clock)
        ref = json.load(open(sys.argv[3]))["components"]
        e = {k: ref[k]["over_spin_pJ"]["mean"] for k in ["smem_read", "l2_read", "hbm_read"]}
    else:
        e = {k: comp[k]["over_spin_pJ"]["mean"] for k in ["smem_read", "l2_read", "hbm_read"] if k in comp}
    for g in ["gemm_bf16", "gemm_int8"]:
        n = ncu.get(g, {})
        if "lts__t_bytes.sum" not in n or len(e) < 3 or g not in comp:
            continue
        macs = 8192**3
        smem_b = n.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 0) * 128
        l2_b = n.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 0)
        dram_b = n.get("dram__bytes_read.sum", 0) + n.get("dram__bytes_write.sum", 0)
        parts = {"smem_pJ_per_mac": smem_b * e["smem_read"] / macs, "l2_to_sm_pJ_per_mac": l2_b * e["l2_read"] / macs,
                 "hbm_beyond_l2_pJ_per_mac": dram_b * (e["hbm_read"] - e["l2_read"]) / macs}
        comp[g]["traffic_per_call"] = {"smem_ld_bytes_est": smem_b, "l2_to_sm_bytes": l2_b, "dram_bytes": dram_b,
                                       "hmma_inst": n.get("sm__inst_executed_pipe_tensor_op_hmma.sum")}
        comp[g]["memory_parts"] = parts
        comp[g]["datapath_residual_pJ_per_mac"] = comp[g]["over_spin_pJ"]["mean"] - sum(parts.values())
        comp[g]["ncu_note"] = ("smem bytes = shared-ld wavefronts x 128 B (ldmatrix); L2 bytes = global-ld bytes at L1TEX "
                               "(cp.async); per-byte energies are the over-spin values of the LDS/LDG microbenchmarks; "
                               "smem fills by cp.async, RF traffic and epilogue stores are not subtracted, so the residual "
                               "is tensor datapath + RF + operand staging")

    out = path.replace(".json", "_summary.json")
    json.dump(S, open(out, "w"), indent=1)

    rows = [("idle (no kernel)", "idle_none"), ("active idle (1-thread spin)", "spin_1thread_full"),
            ("all SMs resident, nanosleep", "spin_nap_full"), ("all SMs busy spin, half SMs", "spin_busy_half"),
            ("all SMs busy spin (issue+control)", "spin_busy_full")]
    L = ["| baseline | measured board W (min-max) | SM MHz | kiln board W |", "|---|---|---|---|"]
    kb = {"idle_none": f"{KILN['board_idle_W']:.0f} (clocks gated, busy=0)",
          "spin_1thread_full": f"{KILN['board_idle_W']:.0f}",
          "spin_nap_full": f"{KILN['board_busy_noact_W']:.0f} (busy=1, activity=0)",
          "spin_busy_full": f"{KILN['board_busy_noact_W']:.0f}-{KILN['board_busy_fullact_noenergy_W']:.0f} (busy=1, act 0..1, no op energy)"}
    for name, k in rows:
        if k in S["baselines"]:
            p = S["baselines"][k]["power_w"]
            L.append(f"| {name} | {p['mean']:.1f} ({p['min']:.1f}-{p['max']:.1f}) | {S['baselines'][k]['sm_mhz']['mean']:.0f} | {kb.get(k, '-')} |")
    L += ["", "| component | over busy-spin: (P_full - P_spin)/thr, pJ (min-max) | all-in duty slope (full vs 50%) | half-SM slope | full: W, throughput/s, SM MHz | kiln A100 (V_nom 0.75 V) |",
          "|---|---|---|---|---|---|"]
    kr, kw = KILN["rf_read_pJ_per_B"], KILN["rf_write_pJ_per_B"]
    kiln_col = {"ffma": f"fp32 op {KILN['fp32_elem_pJ']:.3f}; with RF 12 B rd + 4 B wr: {KILN['fp32_elem_pJ'] + 12 * kr + 4 * kw:.2f}",
                "smem_read": f"{KILN['l1_smem_read_pJ_per_B']:.3f} (+RF wr {kw:.3f})",
                "l1_read": f"{KILN['l1_smem_read_pJ_per_B']:.3f} (+RF wr {kw:.3f})",
                "l2_read": f"{KILN['l2_read_pJ_per_B']:.3f} (+NoC, +L1 fill)",
                "hbm_read": f"{KILN['hbm_pJ_per_B']:.1f} (+L2 path)", "hbm_copy": f"{KILN['hbm_pJ_per_B']:.1f} (+L2 path)"}
    for g in ["bf16", "fp16", "tf32", "int8"]:
        kiln_col[f"gemm_{g}"] = f"{KILN['mac_pJ'][g]:.3f} per MAC (datapath only)"
    fmt = lambda x: f"{x['mean']:.3g} ({x['min']:.3g}-{x['max']:.3g})"
    for b, c in comp.items():
        extra = ""
        if "datapath_residual_pJ_per_mac" in c:
            extra = f"; minus ncu-counted SMEM/L2/HBM traffic: {c['datapath_residual_pJ_per_mac']:.2f}"
        if "alu_upper_bound_pJ_per_B" in c:
            extra = f"; LOP3 share <= {c['alu_upper_bound_pJ_per_B']:.2f}"
        duty = fmt(c["duty_pJ"]) + (f"; 50% vs 25%: {fmt(c['duty50_vs_25_pJ'])}" if "duty50_vs_25_pJ" in c else "")
        half = fmt(c["half_pJ"]) if "half_pJ" in c else "-"
        L.append(f"| {b} ({c['unit']}) | {fmt(c['over_spin_pJ'])}{extra} | {duty} | {half} | "
                 f"{c['full_power_w']['mean']:.0f} W, {c['full_throughput']['mean']:.3e}, {c['full_sm_mhz']['mean']:.0f} | {kiln_col.get(b, '-')} |")
    open(path.replace(".json", "_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
