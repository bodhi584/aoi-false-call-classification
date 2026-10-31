# Test plan

Team 1 · Binary classification of AOI alarms into true defects and false calls · issue #34 · author: Melisa Subasi

## 1. What we test

1. **Data:** the Siemens AOI data are correct and in time order.
2. **Classifier and threshold:** the three models run, and the threshold keeps at least 99 % of true defects.
3. **Day-by-day simulation:** each day is classified with a model that only knows earlier days.
4. **KPIs:** recall of at least 99 % and specificity of at least 20 %.

We do not test a real AOI machine or camera images. They are not part of the project.

Terms: *true defect* is the positive class. *Recall* is the share of true defects that are classified as true defect. *Specificity* is the share of false calls that are classified as false call, which means manual inspections saved.

## 2. How the tests run

- **Unit tests** use a small set of made-up data, because the real data are not in the repository. They run on every push, on Windows, macOS and Linux, with Python 3.11 and 3.12 (CI). File: `tests/test_classification.py`.
- **Data checks** run on the real data after `scripts/download_data.py`.
- **Simulation run:** `scripts/run_replay.py` on the real data (about 10 minutes). It writes `results/replay.json`.
- **KPI check:** at each milestone, we read the KPIs from the files in `results/`.

Status: **exists** = the test is in the repository now. **planned** = we add it in issue #39 (due 24 Nov).

## 3. Tests

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

### Classifier and threshold

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| C1 | The threshold rule works | `threshold_for` on a small example with `max_miss = 0` | No true defect is released; the expected number of false calls is released | exists |
| C2 | The rule works for 1 % | Made-up scores of 200 defective boards, `max_miss = 0.01` | At most 2 defective boards are released | planned |
| C3 | Recall is at least 99 % on the calibration days | After `fit`, classify the held-back calibration days | Recall ≥ 99 % | planned |
| C4 | The classifier finds false calls and keeps true defects | Train and classify made-up data with a clear signal | More than 30 % of false calls and fewer than 20 % of true defects are classified as false call | exists |
| C5 | All three models run | Train each model on made-up data | Output has `score`, `defective`, `predicted_false_call` | exists |
| C6 | Bad training data gives a clear error | Set all labels to 0 and call `fit` | `ValueError` | exists |
| C7 | Same seed gives the same result | Train the same model twice with the same seed | Same scores | planned |

C3 tests the rule on the calibration days. Whether 99 % also holds on later days is the KPI K1.

### Day-by-day simulation

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| S1 | The model only knows earlier days | In `replay`, compare the training day with the classified day | `trained_on_day <= day` in every row | exists |
| S2 | The calibration days come before the classified day | Record the days that `fit` receives in a replay of made-up data | All days are before the classified day | planned |
| S3 | The simulation goes on when a window has too few defects | Made-up data with no defects in one window | No error; the old classifier is kept | planned |
| S4 | The full run works on the real data | Run `scripts/run_replay.py` | No error; all six runs have the same number of boards | planned |

### KPIs and limits

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| K1 | Recall | 100 % minus `defects_released_pct` in `results/replay.json` | ≥ 99 % | open (now 96.3–97.8 %) |
| K2 | Specificity | `false_calls_released_pct` in `results/replay.json` | ≥ 20 % | open (now 6.5–15.0 %, at a lower recall) |
| K3 | Time order, not random split | Check how each number in the report was made | Every KPI comes from the time-order test | exists (`results/baseline.json` shows both) |
| K4 | Every number has a source | Compare each number in the report with a file in `results/` | No number without a file | manual check at each milestone |
| K5 | The limits are written down | Read the README and the report | They name: one production line, 132 days, labels that may be wrong, anonymised data, non-commercial licence, and a line that changes over time | manual check at each milestone |

If K1 or K2 is not reached at the end, we report what works, what does not, and why. That is a result, not a failed test. The test is that we measure it correctly.

## 4. Risks and their tests

| Risk (proposal) | Test or check |
|---|---|
| The line changes over time, so 99 % recall may not hold later | S1, S2, K1, and the weekly retraining test (#38) |
| Labels from operators may be wrong | D1, K5, and the error analysis in the report |
| Only about 1 % of the joints are true defects | We use recall and specificity, not accuracy (C2, C3, K1, K2) |
| Only one line and 132 days | K5, D7 |
| Data licence is non-commercial, no personal data | D6; the data are never added to the repository (K5) |
| Results cannot be repeated | C7, `requirements.txt`, CI on three systems |
| Numbers in the report differ from the result files | K4 |

## 5. Who and when

Melisa Subasi writes the test plan (#34), the tests (#39) and the test report (#47).

| Date | Milestone | Test work |
|---|---|---|
| 3 Nov | P2: core code and baseline | This plan. The existing tests pass in CI. |
| 24 Nov | P3: features and testing | The planned tests are added: D1, D2, D3, D7, C2, C3, C7, S2, S3, S4 (#39) |
| 8 Dec | P4: full evaluation | Test report on the final model: K1 to K5 (#47) |
| 22 Dec | P5: packaging | All tests pass on a clean install |
| 12 Jan | Final | Test plan and evaluation in the final report (#44) |
