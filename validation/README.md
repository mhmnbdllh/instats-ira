# Validation scripts

These scripts reproduce every table and every number of the Method validation
section of the article on the Inter-Rater Agreement module
(`modules/reliability.py`, entry point `run_multi_irar`). Run them from the
repository root. Each script writes one output file; the output files in this
folder are the ones from which the published tables and numbers were built.

All scripts use the ten configurations S1–S10 of the three sample files:

| Configuration | File | Raters | Declared data type | q | Scale minimum |
|---|---|---|---|---|---|
| S1 | `sample-instats_Dichotomous.csv` | R1, R2 | dichotomous | 2 | – |
| S2 | `sample-instats_Dichotomous.csv` | R1–R4 | dichotomous | 2 | – |
| S3 | `sample-instats_Nominal-Ordinal.csv` | R1, R2 | nominal | 5 | – |
| S4 | `sample-instats_Nominal-Ordinal.csv` | R1–R4 | nominal | 5 | – |
| S5 | `sample-instats_Nominal-Ordinal.csv` | R1, R2 | ordinal | 5 | 1 |
| S6 | `sample-instats_Nominal-Ordinal.csv` | R1–R4 | ordinal | 5 | 1 |
| S7 | `sample-instats_Interval-Ratio.csv` | R1, R2 | interval | – | – |
| S8 | `sample-instats_Interval-Ratio.csv` | R1–R4 | interval | – | – |
| S9 | `sample-instats_Interval-Ratio.csv` | R1, R2 | ratio | – | – |
| S10 | `sample-instats_Interval-Ratio.csv` | R1–R4 | ratio | – | – |

| Script | What it does | Command | Output |
|---|---|---|---|
| `describe_data.py` | MD5 checksum, size, missing ratings and per-rater distribution of the three sample files | `python3 validation/describe_data.py . validation/describe_data.json` | `describe_data.json` |
| `run_scenarios.py` | Runs S1–S10 through `run_multi_irar` (alpha = 0.05, 1,000 bootstrap resamples) and stores every returned quantity, including the agreement tables and the ICC F-test table | `python3 validation/run_scenarios.py . validation/run_scenarios.json 0.05 1000` | `run_scenarios.json` |
| `published_examples.py` | Reproduces the numerical examples printed in Fleiss, Cohen & Everitt (1969), Shrout & Fleiss (1979) and Krippendorff (2011), and, for the Fleiss (1971) data, the values returned by the R package irr | `python3 validation/published_examples.py . validation/published_examples.json` | `published_examples.json` |
| `r_reference_values.R` | Reference values from two R implementations: irr (`kappam.fleiss`) for the Fleiss (1971) data and epiR (`epi.ccc`, Fisher z-transform interval) for Lin's CCC in S7 | `Rscript validation/r_reference_values.R > validation/r_reference_values.txt` | `r_reference_values.txt` |
| `validate_independent.py` | Compares every quantity reported for S1–S10 with scikit-learn, statsmodels, irrCAC, krippendorff, pingouin, SciPy and with formulas recomputed from the primary sources; recomputes the bootstrap interval of Krippendorff's alpha on the same resampled units; checks the dispatcher and the identities between configurations; deletes 10% of the ratings of every configuration (seed 7) and compares Krippendorff's alpha with the krippendorff package; stores the explanatory quantities cited in the article | `python3 validation/validate_independent.py . validation/validate_independent.json` | `validate_independent.json` |
| `coverage_sim.py` | Coverage of every interval and accuracy of every standard error in 1,000 resamples of 86 units per configuration (about 28 minutes on two cores; the archived run took 1,666 s) | `python3 validation/coverage_sim.py . validation/coverage_sim.json 1000 20261026` | `coverage_sim.json` |
| `skewed_prevalence_mc.py` | Coverage of the two-rater intervals under skewed category prevalence and small samples (synthetic data; Limitations) | `python3 validation/skewed_prevalence_mc.py . validation/skewed_prevalence_mc.json 2000 20261022` | `skewed_prevalence_mc.json` |
| `timing.py` | Median computation time of one analysis of S1–S10 with 1,000 and 10,000 bootstrap resamples | `python3 validation/timing.py . validation/timing.json` | `timing.json` |

The automated test suite is `tests/test_irar_formulas.py`:

```
python3 tests/test_irar_formulas.py validation/test_suite_counts.json > validation/test_suite_output.txt
```

`test_suite_output.txt` lists every check with its result; `test_suite_counts.json`
gives the number of checks of each kind.

## Requirements

The application's own requirements (`requirements.txt`) plus the reference
packages:

```
pip install -r tests/requirements-test.txt     # krippendorff, statsmodels
pip install --no-deps irrCAC==0.4.4            # used only by validate_independent.py
```

irrCAC 0.4.4 pins an old SciPy version; installing it with `--no-deps` keeps
the current SciPy and works for the functions used here.

`r_reference_values.R` needs R with the irr and epiR packages
(`install.packages(c("irr", "epiR"))`).

Versions used for the archived outputs: Python 3.13.16, NumPy 2.5.3,
pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1, pingouin 0.7.0,
statsmodels 0.15.0, irrCAC 0.4.4, krippendorff 0.9.0; R 4.3.3, irr 0.85, epiR 2.0.67.
