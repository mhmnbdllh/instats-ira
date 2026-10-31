#!/usr/bin/env python3
"""Reproduce worked examples on published data with the application's
engine (modules/reliability.py).
Usage: published_examples.py <repo_dir> <out_json>

Sources and reference values
  Fleiss, Cohen & Everitt (1969), Psychol. Bull. 72, 323-327, Tables 1-2:
      N = 200, 3 x 3 table of proportions; unweighted kappa = .429,
      Var (general) = .002885, Var (null) = .003082; weighted kappa = .508.
  Fleiss (1971), Psychol. Bull. 76, 378-382, psychiatric-diagnosis data
      (30 patients x 6 raters, 5 categories) as distributed with the R
      package irr (data set `diagnoses`); reference values returned by
      irr 0.85, kappam.fleiss(diagnoses, detail = TRUE)
      (r_reference_values.R): kappa = .430; category kappas .245, .245,
      .520, .471, .566.
  Shrout & Fleiss (1979), Psychol. Bull. 86, 420-428, Tables 2 and 4:
      6 targets x 4 judges; ICC(1,1) .17, ICC(2,1) .29, ICC(3,1) .71,
      ICC(1,4) .44, ICC(2,4) .62, ICC(3,4) .91.
  Krippendorff (2011), Computing Krippendorff's alpha-reliability: 12 units x
      4 observers with missing values; alpha nominal .743, ordinal .815,
      interval .849, ratio .797.
"""
import sys, json
import numpy as np, pandas as pd
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from modules.reliability import run_multi_irar, _kappa_se_z_p_exact

res = {}

# --- Fleiss, Cohen & Everitt (1969) --------------------------------------
counts = np.array([[106, 10, 4], [22, 28, 10], [2, 12, 6]], dtype=float)  # proportions x 200; marginals .60/.30/.10 and .65/.25/.10
assert counts.sum() == 200
u = _kappa_se_z_p_exact(counts, np.eye(3))
W = np.array([[1.0, 0.0, 4 / 9], [0.0, 1.0, 2 / 3], [4 / 9, 2 / 3, 1.0]])
w = _kappa_se_z_p_exact(counts, W)
res["FCE1969"] = {
    "kappa": [u["kappa"], 0.429],
    "var_general": [u["se_general"] ** 2, 0.002885],
    "var_null": [u["se_null"] ** 2, 0.003082],
}
# Weighted example: the paper prints p_o(w) = .787 and p_e(w) = .567 and
# derives kappa_w = .508 from these ROUNDED terms; the unrounded value is
# .507. The two printed intermediate terms are checked here, and the
# engine's kappa_w is compared with the value obtained from them.
P = counts / counts.sum()
po_w = float((W * P).sum()); pe_w = float((W * np.outer(P.sum(1), P.sum(0))).sum())
res["FCE1969"]["po_weighted"] = [po_w, 0.787]
res["FCE1969"]["pe_weighted"] = [pe_w, 0.567]
res["FCE1969_extra"] = {"weighted_kappa_engine": w["kappa"], "weighted_kappa_published": 0.508,
                        "weighted_kappa_from_rounded_terms": (0.787 - 0.567) / (1 - 0.567)}

# --- Fleiss (1971) data; reference values: R irr 0.85 kappam.fleiss --------
diag = np.array([
    [4, 4, 4, 4, 4, 4], [2, 2, 2, 5, 5, 5], [2, 3, 3, 3, 3, 5], [5, 5, 5, 5, 5, 5],
    [2, 2, 2, 4, 4, 4], [1, 1, 3, 3, 3, 3], [3, 3, 3, 3, 5, 5], [1, 1, 3, 3, 3, 4],
    [1, 1, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [1, 4, 4, 4, 4, 4], [1, 2, 4, 4, 4, 4],
    [2, 2, 2, 3, 3, 3], [1, 4, 4, 4, 4, 4], [2, 2, 4, 4, 4, 5], [3, 3, 3, 3, 3, 5],
    [1, 1, 1, 4, 5, 5], [1, 1, 1, 1, 1, 2], [2, 2, 4, 4, 4, 4], [1, 3, 3, 5, 5, 5],
    [5, 5, 5, 5, 5, 5], [2, 4, 4, 4, 4, 4], [2, 2, 4, 5, 5, 5], [1, 1, 4, 4, 4, 4],
    [1, 4, 4, 4, 4, 5], [2, 2, 2, 2, 2, 4], [1, 1, 1, 1, 5, 5], [2, 2, 4, 4, 4, 4],
    [1, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5]])
df = pd.DataFrame(diag, columns=[f"P{i}" for i in range(1, 7)])
g = {"name": "Fleiss1971", "items": list(df.columns), "data_type": "nominal", "q_declared": 5, "scale_min": None}
c = run_multi_irar(df, [g], n_boot=0)["constructs"][0]
fk = c["fleiss_kappa"]
pub_cat = {1: 0.245, 2: 0.245, 3: 0.520, 4: 0.471, 5: 0.566}
res["Fleiss1971"] = {"kappa": [fk["kappa"], 0.430]}
for cat, v in pub_cat.items():
    res["Fleiss1971"][f"kappa_cat{cat}"] = [fk["kappa_per_category"][cat], v]
res["Fleiss1971_extra"] = {"se_null": fk["kappa_se"], "z": fk["kappa_z"], "se_general": fk["kappa_se_general"],
                           "ci_t": [fk["ci_lo_t"], fk["ci_hi_t"]]}

# --- Shrout & Fleiss (1979) -----------------------------------------------
sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
df = pd.DataFrame(sf, columns=["J1", "J2", "J3", "J4"])
g = {"name": "SF1979", "items": list(df.columns), "data_type": "interval", "q_declared": None, "scale_min": None}
c = run_multi_irar(df, [g], n_boot=0)["constructs"][0]
ic = c["icc"]
res["ShroutFleiss1979"] = {
    "ICC(1,1)": [ic["icc_1_1"], 0.17], "ICC(2,1)=ICC(A,1)": [ic["icc_2_1"], 0.29],
    "ICC(3,1)=ICC(C,1)": [ic["icc_3_1"], 0.71], "ICC(1,4)": [ic["icc_1_k"], 0.44],
    "ICC(2,4)=ICC(A,4)": [ic["icc_2_k"], 0.62], "ICC(3,4)=ICC(C,4)": [ic["icc_3_k"], 0.91],
}

# --- Krippendorff (2011) --------------------------------------------------
nan = np.nan
K = np.array([
    [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
    [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]]).T
df = pd.DataFrame(K, columns=["A", "B", "C", "D"])
pub = {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}
res["Krippendorff2011"] = {}
for dt, v in pub.items():
    q, smin = (5, 1) if dt == "ordinal" else ((5, None) if dt == "nominal" else (None, None))
    g = {"name": dt, "items": list(df.columns), "data_type": dt, "q_declared": q, "scale_min": smin}
    c = run_multi_irar(df, [g], n_boot=0)["constructs"][0]
    res["Krippendorff2011"][f"alpha_{dt}"] = [c["krippendorff"]["alpha"], v]
    res["Krippendorff2011"][f"n_pairable_{dt}"] = c["krippendorff"].get("n_units")

def rnd(v, ref):
    d = len(str(ref).split(".")[1]) if "." in str(ref) else 0
    return round(float(v), d)
ok = True
for src, d in res.items():
    if src.endswith("_extra"):
        print(src, d); continue
    for key, val in d.items():
        if not isinstance(val, list):
            print(f"  {src} {key}: {val}"); continue
        app, ref = val
        match = rnd(app, ref) == ref
        ok &= match
        print(f"{src:18s} {key:22s} app={app:.6f} rounded={rnd(app, ref)} reference={ref} {'OK' if match else 'MISMATCH'}")
res["all_match"] = bool(ok)
json.dump(res, open(out, "w"), indent=1, default=float)
print("ALL MATCH" if ok else "SOME MISMATCH")
