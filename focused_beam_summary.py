# -*- coding: utf-8 -*-
"""Print the key tables of focused_beam.json (used to build focused_beam.log)."""
import json, math, numpy as np, sys
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "focused_beam.json", encoding="utf-8"))
ANOM = D["meta"]["anomaly"]
print("meta:", {k: D["meta"][k] for k in ("python","numpy","scipy","sympy","os","anomaly","seed")})
print()
print("### stage `field` -- Maxwell residuals (normalised by a0; k=1)")
f = D["field"]
for tag in ("pulse_order1","pulse_order3","long_order1","long_order3"):
    s = f[tag+"_scaling"]
    print(" %-14s slopes: divE=%.2f ampere=%.2f Ez=%.2f" % (tag, s["divE"]["slope"], s["ampere"]["slope"], s["Ez"]["slope"]))
    for r in f[tag]:
        print("    eps=%.3f  Ez/E=%.4g  |divE|=%.3g  |curlB-dtE|=%.3g  |divB|=%.1g  |curlE+dtB|=%.1g"
              % (r["eps"], r["Ez_rel"], r["divE_rel"], r["amp_rel"], r["divB"], r["far"]))
print("  plane-wave limit (eps->0) max|E-E_pw| :", [(p["eps"], p["max_abs_err"]) for p in f["plane_wave_limit"]])
print()
print("### stage `conv` -- tolerance convergence and invariants")
for row in D["conv"]["rows"]:
    print(" case eps=%.2f a0=%g gamma=%g delta=%g" % (row["eps"], row["a0"], row["gamma"], row["delta"]))
    for s in row["tol_scan"]:
        print("   rtol=%.0e nfev=%5d  |dr1|=%.3g rel=%.3g  |dr0|=%.3g  S.u=%.1g S.S=%.1g orth=%.1g"
              % (s["rtol"], s["nfev"], s["r1_err_vs_tightest"], s["r1_rel_err"], s["r0_err_vs_tightest"], s["su"], s["ss"], s["orth"]))
print("  invariants along trajectory:", D["conv"]["invariants"])
print()
print("### stage `eps` -- anomaly-resolved rotation vs diffraction parameter")
for tag in [k for k in D["eps"] if not k.endswith("_scaling") and k != "wall_s"]:
    rows = D["eps"][tag]
    b0 = rows[0]["beam"]
    print(" [%s] gamma=%g a0=%g delta=%g N=%g phi0=%g" % (tag, rows[0]["gamma"], b0["a0"], b0["delta"], b0["N"], b0["phi0"]))
    print("   eps    w0/lam   |r0|        |r1|        r2_z        -A/2        dev_rel   du_perp")
    for r in rows:
        b = r["beam"]
        print("   %-6.3f %-8.3f %-11.5g %-11.5g %-11.7g %-11.7g %-9.4g %.4g"
              % (b["eps"], b["w0_over_lam"], r["r0_abs"], r["r1_abs"], r["r2"][2], r["r2_pw_z"], r["r2_dev_rel"], r["du_perp"]))
    s = D["eps"][tag+"_scaling"]
    for k in s:
        print("     slope[%s]=%.2f  local=%s" % (k, s[k]["slope"], [round(x[2],2) for x in s[k]["local"]]))
print()
print("### stage `order` -- field-model systematics (order 3/c0=0 vs c0=1 vs order 1)")
for r in D["order"]["rows"]:
    print("   eps=%.2f order=%d c0=%g |r0|=%.6g |r1|=%.6g du_perp=%.4g" % (r["eps"], r["order"], r["c0"], r["r0_abs"], r["r1_abs"], r["du_perp"]))
print()
print("### stage `xcheck` -- physical anomaly a=%.6g" % ANOM)
for r in D["xcheck"]["rows"]:
    print("   eps=%.3f delta=%g |dr|=%.6g  |a r1 + a^2 r2|=%.6g  |pw|=%.6g  rel_err=%.2g  dr/pw=%.5g  |r0|=%.4g"
          % (r["eps"], r["delta"], r["dr_abs"], r["pred_abs"], r["pw_abs"], r["rel_err_vs_pred"], r["ratio_to_pw"], r["r0_abs"]))
print()
print("### stage `cep` -- 16-point CEP scan")
for tag in [k for k in D["cep"] if k != "wall_s"]:
    o = D["cep"][tag]
    print(" [%s] gamma=%g a0=%g delta=%g eps=%g N=%g" % (tag, o["gamma"], o["a0"], o["delta"], o["eps"], o["N"]))
    for k in ("r0","r1","r2"):
        print("   %s: mean=%s  A1=%s  A2=%s" % (k, ["%.5g"%x for x in o[k+"_mean"]], ["%.4g"%x for x in o[k+"_A1"]], ["%.2g"%x for x in o[k+"_A2"]]))
        print("      |.|mean=%.6g  |.|cep_half=%.4g  rel_cep=%.4g" % (o[k+"_abs_mean"], o[k+"_abs_ptp_half"], o[k+"_rel_cep"]))
print()
print("### stage `cepscan` -- CEP-odd amplitude A1 (4-point scans)")
cs = D["cepscan"]
for key, lab in (("lin_eps","eps"),("lin_a0","a0"),("lin_N","N"),("lin_gamma","gamma"),("lin_b","b_over_w0"),("cir_b","b_over_w0")):
    print(" [%s]" % key)
    for r in cs[key]:
        print("   %s=%-8g eps=%.3g a0=%g gamma=%g N=%g b/w0=%g | A1(r0)=%.5g A1(r1)=%.5g A1(r2)=%.3g |r0|mean=%.4g rel_cep=%.4g"
              % (lab, r[lab], r["eps"], r["a0"], r["gamma"], r["N"], r["b_over_w0"], r["r0_A1_abs"], r["r1_A1_abs"], r["r2_A1_abs"], r["r0_abs_mean"], r["r0_rel_cep"]))
    if key+"_scaling" in cs:
        s = cs[key+"_scaling"]
        print("     slope r0=%.2f local=%s" % (s["r0"]["slope"], [round(x[2],2) for x in s["r0"]["local"]]))
        print("     slope r1=%.2f local=%s" % (s["r1"]["slope"], [round(x[2],2) for x in s["r1"]["local"]]))
print()
print("### stage `scans`")
for key, lab in (("a0","a0"),("gamma","gamma"),("N","N"),("b","b_over_w0"),("delta","delta")):
    print(" [%s]" % key)
    for r in D["scans"][key]:
        b = r["beam"]
        v = r.get(lab, b.get(lab))
        print("   %s=%-8g | |r0|=%.6g |r1|=%.6g r2_z=%.6g -A/2=%.6g du_perp=%.4g  r0=(%s)"
              % (lab, v, r["r0_abs"], r["r1_abs"], r["r2"][2], r["r2_pw_z"], r["du_perp"], ", ".join("%.3g"%x for x in r["r0"])))
    if key+"_scaling" in D["scans"]:
        s = D["scans"][key+"_scaling"]
        print("     slopes:", {k: round(s[k]["slope"],3) for k in s})
print()
print("### stage `a0big` -- CEP amplitude at large a0 (linear polarisation)")
ab = D["a0big"]
for key, lab in (("a0","a0"),("eps_at_a08","eps"),("gamma_at_a08","gamma")):
    print(" [%s]" % key)
    for r in ab[key]:
        print("   %s=%-7g eps=%.3g gamma=%g | A1(r0)=%.6g A1(r1)=%.6g |r0|mean=%.5g rel_cep=%.4g du_perp=%.4g"
              % (lab, r[lab], r["eps"], r["gamma"], r["r0_A1_abs"], r["r1_A1_abs"], r["r0_abs_mean"], r["r0_rel_cep"], r["du_perp"]))
    if key+"_scaling" in ab:
        for q, v in ab[key+"_scaling"].items():
            print("     slope %s = %.2f  local=%s" % (q, v["slope"], [round(x[2],2) for x in v["local"]]))
print()
print("### stage `window` -- analytic estimates only")
for r in D["window"]["rows"]:
    print("   gamma=%-7g a0=%-5g N=%g eps=%g w0/lam=%.2f | chi=%.3g R_C=%.3g N_gam=%.3g P_sf=%.3g"
          % (r["gamma"], r["a0"], r["N"], r["eps"], r["w0_over_lam"], r["chi"], r["R_C"], r["N_gamma"], r["P_radflip"]))
