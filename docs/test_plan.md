# Test plan

Team 1 · Binary classification of AOI alarms into true defects and false calls · issue #34 · author: Melisa Subasi · updated for P2 (2 Nov 2026)

Numbers in brackets, such as [7], refer to the bibliography report ([`literature_report.pdf`](literature_report.pdf)). [7] is the core paper: Pfab & Rothering (2025).

## 1. What we test

1. **Experiments:** do the two next steps named in [7], a better threshold and retraining, lower the missed true defects on later production periods? This is the design of experiments in section 2.
2. **Code tests:** is each run correct? Data, split, threshold and retraining are checked by automatic tests (section 4).

We do not test a real AOI machine or camera images. They are not part of the project.

Terms: *true defect* is the positive class. *Missed true defects* = 1 − recall (called *slip* in [7]). *False calls released* = specificity, the share of false calls that skip the manual station (called *volume reduction* in [7]).

## 2. Design of experiments

The baseline is our reimplementation of [7] (`scripts/replicate_pfab2025.py`): trained once, threshold as in [7]. Every setting is compared with this baseline and with the published result of [7].

| Factor | n | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Level 6 |
|---|---|---|---|---|---|---|---|
| A · Threshold | 3 | as in [7], corrected | safety margin [14] | re-set per period [15] | | | |
| B · Retraining | 6 | never, as in [7] | per period | weekly | every 3 days | daily | on drift [16] |
| C · Retraining data | 2 | all data so far | recent window [12] | | | | |
| D · Model | 2 | random forest | balanced random forest | | | | |

- **A** and **B** are the two next steps named in [7]. **C** only matters when B is not *never*. **D** keeps the two models of [7], so every result can be compared with the paper.
- **Held constant:** the Siemens data and the split of [7] (first half 80/20 stratified, second half 5 periods in time order), the input features, the hyper-parameters logged by [7], the recall target of 99 % used to set the threshold.
- **Nuisance factor:** the random seed. Each setting is replicated with the 10 seeds of [7].
- **Responses:** missed true defects (%) and false calls released (%) for each of the 5 later periods, mean ± SD over the seeds. Reference lines from [7]: missed true defects ≤ 1 %, false calls released ≥ 40 %.
- **Design:** full factorial 3 × 6 × 2 × 2 = 72 settings × 10 seeds = 720 runs (with B = never, C has no effect, so fewer runs are needed).
- **Order:** first one factor at a time against the baseline (A, then B), then the full design.

## 3. How the code tests run

- **Unit tests** use a small set of made-up data, because the real data are not in the repository. They are written for the core code in P3 (#39) and then run on every push (CI).
- **Data checks** run on the real data after `scripts/download_data.py`.
- **Reproduction check:** `scripts/replicate_pfab2025.py` on the real data writes `results/replication_pfab2025.json`; its baseline is compared with the run logs of [7].

Status: **exists** = the test is in the repository now. **planned** = we add it in issue #39 (due 24 Nov).

## 4. Code tests

### Data

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| D1 | Labels are only 0 or 1 | List the values of column `class` | Only 0 and 1 | planned |
| D2 | No missing values | Count missing values in the feature columns | 0 missing values | planned |
| D3 | Records are in time order | Load the data, then compare the timestamps | Timestamps never go down | planned |
| D4 | A board is a true defect if one of its joints is | Group made-up joints by board | Board label = highest joint label | exists |
| D5 | The category columns are there | Look for each name in `CATEGORICAL` | All columns found | exists |
| D6 | We have the right data file | `scripts/download_data.py` checks the SHA-256 of `dataset.csv` and `mapping.json` | Checksums match, or the script stops | exists (in the script) |
| D7 | Our data facts are right | Run `scripts/data_facts.py` and compare with `results/data_facts.json` | 440,274 joints and 39,742 boards | planned |

### Core code (reimplementation of [7])

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| R1 | Same split as [7] | Split made-up data with a seed | First half 80/20 stratified; second half 5 periods in time order; no row in two parts | planned |
| R2 | Threshold keeps 99 % recall on its own data | 5-fold threshold on made-up scores | Recall ≥ 99 % in each fold | planned |
| R3 | Threshold belongs to the final model (corrected) | Fit the final model, then set the threshold | Threshold is computed from the final model, not from the last tuning step | planned |
| R4 | Retraining only uses earlier data | Record the rows used before each period | All rows come before the period | planned |
| R5 | Baseline reproduces [7] | Compare `results/replication_pfab2025.json` with the run logs of [7] | Same pattern; balanced RF on later periods 7.5 % in both (done); open gaps listed in section 5 | exists (manual) |

## 5. Responses and checks

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| K1 | Missed true defects per later period | From the result files, mean ± SD over 10 seeds | Lower than the baseline; reference line 1 % | baseline: 5–14 % |
| K2 | False calls released per later period | From the result files, mean ± SD over 10 seeds | Shown next to the baseline; reference line 40 % | baseline: 55.8 % (RF), 57.9 % (balanced RF) |
| K3 | Time order, not random split | Check how each number in the report was made | Every later-period number uses only earlier data for training | exists |
| K4 | Every number has a source | Compare each number in the report with a file in `results/` | No number without a file | manual check at each milestone |
| K5 | The limits are written down | Read the README and the report | They name: one production line, 132 days, labels that may be wrong, anonymised data, non-commercial licence, a line that changes over time | manual check at each milestone |

Open gaps of the baseline (checked in P3): random forest on later periods 9.1 % (ours) vs. 11.3 % ([7]); test set 1.5–1.9 % (ours) vs. 0.7 % ([7]).

If a setting does not reach the reference lines, we report what works, what does not, and why. That is a result, not a failed test. The test is that we measure it correctly.

## 6. Risks and their tests

| Risk | Test or check |
|---|---|
| The line changes over time, so the 1 % line may not hold later | The design of experiments (factors A and B), K1 |
| Our reimplementation differs from [7] | R1–R5 |
| Labels from operators may be wrong | D1, K5 |
| Only about 1 % of the joints are true defects | Missed true defects and false calls released instead of accuracy (K1, K2) |
| Only one line and 132 days | K5, D7 |
| Data licence is non-commercial, no personal data | D6; the data are never added to the repository (K5) |
| Results cannot be repeated | 10 seeds, `requirements.txt`, unit tests in CI (P3) |
| Numbers in the report differ from the result files | K4 |

## 7. Who and when

Melisa Subasi writes the test plan (#34), the tests (#39) and the test report (#47).

| Date | Milestone | Test work |
|---|---|---|
| 3 Nov | P2: core code and baseline | This plan. Baseline reproduced (R5). |
| 24 Nov | P3: features and testing | Run the design of experiments (A first, then B). Add D1, D2, D3, D7, R1–R4 (#39). |
| 8 Dec | P4: full evaluation | Full design; test report K1 to K5 (#47) |
| 22 Dec | P5: packaging | All tests pass on a clean install |
| 12 Jan | Final | Test plan and evaluation in the final report (#44) |
