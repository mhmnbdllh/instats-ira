#!/usr/bin/env python3
"""Coverage of the application's 95% t-intervals for two-rater
dichotomous coefficients when category prevalence is skewed and n is
small: Cohen's kappa (general SE, Fleiss, Cohen & Everitt 1969), Scott's
pi (general SE, Gwet 2021) and Gwet's AC1 (jackknife SE). Ratings are
drawn from fixed joint distributions; the population value of each
coefficient is computed from the joint distribution itself.
Usage: skewed_prevalence_mc.py <repo> <out_json> <reps> <seed>"""
import sys, json, numpy as np
repo, out, R, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, repo)
import warnings; warnings.filterwarnings("ignore")
from modules.reliability import _cohen_kappa, _scott_pi, _gwet_ac1
cells = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
SETUPS = {"near-chance agreement, 90/10 prevalence": (0.81, 0.09, 0.09, 0.01),
          "high agreement, 90/10 prevalence": (0.85, 0.05, 0.05, 0.05),
          "moderate agreement, 75/25 prevalence": (0.65, 0.10, 0.10, 0.15)}
def population(j):
    p11, p10, p01, p00 = j
    a, b = p11 + p10, p11 + p01          # P(rater 1 = 1), P(rater 2 = 1)
    po = p11 + p00
    pe_c = a * b + (1 - a) * (1 - b)
    m = (a + b) / 2
    pe_s = m ** 2 + (1 - m) ** 2
    pe_g = 2 * m * (1 - m)               # (1/(q-1)) * sum pi(1-pi), q = 2
    return {"Cohen kappa": (po - pe_c) / (1 - pe_c), "Scott pi": (po - pe_s) / (1 - pe_s),
            "Gwet AC1": (po - pe_g) / (1 - pe_g)}
rng = np.random.default_rng(seed)
res = {}
for name, j in SETUPS.items():
    pop = population(j)
    for n in (50, 86, 200):
        hit = {k: 0 for k in pop}; cnt = {k: 0 for k in pop}
        for _ in range(R):
            r = cells[rng.choice(4, size=n, p=j)].astype(float)
            v = _cohen_kappa(r, ["a", "b"], data_type="dichotomous")
            v = (v or {}).get("variants", {}).get("unweighted", {})
            s = _scott_pi(r[:, 0], r[:, 1])
            g = _gwet_ac1(r, q_declared=2)
            for key, lo, hi in (("Cohen kappa", v.get("ci_lo_t"), v.get("ci_hi_t")),
                                ("Scott pi", s.get("ci_lo_t"), s.get("ci_hi_t")),
                                ("Gwet AC1", g.get("ci_lo_t"), g.get("ci_hi_t"))):
                if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
                    continue
                cnt[key] += 1; hit[key] += lo <= pop[key] <= hi
        res[f"{name} | n={n}"] = {k: {"population": round(pop[k], 4), "valid_reps": cnt[k],
                                      "coverage": round(hit[k] / cnt[k], 3)} for k in pop}
        print(name, n, {k: res[f'{name} | n={n}'][k]['coverage'] for k in pop}, flush=True)
json.dump({"R": R, "seed": seed, "results": res}, open(out, "w"), indent=1)
