# Reference values from two independent R implementations.
#  (1) irr (kappam.fleiss) on the Fleiss (1971) psychiatric-diagnosis data
#      (30 patients, 6 raters, 5 categories) distributed with irr as the data
#      set `diagnoses`: reference values for published_examples.py and for
#      the PUBLISHED-DATA checks of tests/test_irar_formulas.py.
#  (2) epiR (epi.ccc, Fisher z-transform interval) on configuration S7
#      (sample-instats_Interval-Ratio.csv, raters R1 and R2): reference values
#      for Lin's concordance correlation coefficient and its 95% interval.
# Usage (from the repository root):
#   Rscript validation/r_reference_values.R > validation/r_reference_values.txt
suppressPackageStartupMessages({library(irr); library(epiR)})
cat("R", as.character(getRversion()), "| irr", as.character(packageVersion("irr")),
    "| epiR", as.character(packageVersion("epiR")), "\n\n")
cat("(1) Fleiss (1971) data, irr::kappam.fleiss(diagnoses, detail = TRUE)\n")
data(diagnoses)
r <- kappam.fleiss(diagnoses, detail = TRUE)
cat(sprintf("kappa = %.6f, z = %.4f\n", r$value, r$statistic))
print(r$detail)
cat("\n(2) S7, epiR::epi.ccc(R1, R2, ci = 'z-transform', conf.level = 0.95)\n")
d <- read.csv("sample-instats_Interval-Ratio.csv", fileEncoding = "UTF-8-BOM")
cc <- epi.ccc(d$R1, d$R2, ci = "z-transform", conf.level = 0.95)
cat(sprintf("CCC = %.6f, 95%% CI [%.6f, %.6f], C_b = %.6f\n",
            cc$rho.c$est, cc$rho.c$lower, cc$rho.c$upper, cc$C.b))
