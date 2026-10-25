# Test plan

Team 1 · Binary classification of AOI alarms into true defects and false calls · issue #34 · author: Melisa Subasi · status: draft for P2 (3 Nov 2026)

## 1. Purpose and scope

This plan says what we test, how, and when a test passes. It covers four things:

1. the data (Siemens AOI data, Pfab et al. 2024),
2. the classifier (logistic regression, random forest, neural network) and its threshold,
3. the day-by-day simulation (walk-forward evaluation),
4. the KPIs of the project (README section 7).

Out of scope: real hardware, connection to a real AOI machine, camera images (see README, sections 8 and "Future work").

Terms follow the README: positive class = true defect; recall = share of true defects classified as true defect; specificity = share of false calls classified as false call (manual inspections saved).

## 2. Test levels

| Level | What it checks | Data | When it runs | Where |
|---|---|---|---|---|
| Unit test | One function or rule | Small synthetic data set (the real data are not in the repository) | On every push, on Linux, Windows and macOS (CI) | `tests/test_classification.py` |
| Data check | The real data meet our assumptions | Siemens AOI data after `scripts/download_data.py` | After each download, and before each full run | planned, issue #39 |
| Simulation check | The simulation has no access to future data | Synthetic data (unit test) and real data (assertion in the run) | On every push, and in each run of `scripts/run_replay.py` | `tests/` and `scripts/` |
| KPI check | The project targets are met | Result files in `results/` | At each milestone (P2 baseline, P4 final model) | `results/` and the test report (#47) |

## 3. Tests

Status: **exists** = already in `tests/test_classification.py`; **planned** = to be added in issue #39 (due 24 Nov).

### 3.1 Data checks

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| D1 | Labels are only 0 or 1 | Check the unique values of column `class` | Set of values equals {0, 1} | planned |
| D2 | No missing values in the model inputs | Count missing values in the categorical columns and the `inspection_feat*` columns | 0 missing values, or each missing column is listed and handled in the code | planned |
| D3 | Records are in time order | Sort by timestamp, compare with the loaded order | Timestamps never decrease | planned (the loader sorts by time; the test checks it) |
| D4 | A board is true defect if any of its joints is a true defect | Group synthetic joints by board and compare | Board label equals the maximum joint label | exists (`test_board_is_defective_if_any_joint_is`) |
| D5 | Categorical columns are present | Look for each name in `CATEGORICAL` | All columns found | exists (`test_categorical_columns_exist`) |
| D6 | The downloaded file is the published one | `scripts/download_data.py` compares the checksum | Checksum matches | exists in the script; not a unit test |
| D7 | Data facts match the report | Re-run `scripts/data_facts.py`, compare with `results/data_facts.json` | Joints = 440,274; boards = 39,742; same numbers as in the report | planned |

### 3.2 Classifier and threshold

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| C1 | The threshold releases at most the allowed share of true defects | Small example with known scores; `threshold_for` with `max_miss = 0` | No true defect is released; the number of released false calls is as expected | exists (`test_threshold_releases_at_most_the_allowed_defects`) |
| C2 | Same rule for the project setting `max_miss = 0.01` | Synthetic scores with 100 or more defective boards; release at most 1 % of them | At most 1 % of defective boards released | planned |
| C3 | Recall ≥ 99 % on the calibration days | After `fit`, classify the held-back calibration days and compute recall | Recall ≥ 99 % | planned |
| C4 | Classifier finds false calls and keeps true defects | Synthetic data with an easy signal; train, then classify the later days | More than 30 % of false calls classified as false call; fewer than 20 % of true defects classified as false call | exists (`test_classifier_finds_some_false_calls_and_keeps_defects`) |
| C5 | Each of the three models trains and returns a decision | Run all three model names on synthetic data | Output has the columns `score`, `defective`, `predicted_false_call`; more than 10 % of false calls classified as false call | exists (`test_each_model_trains_and_decides`) |
| C6 | Training without true defects fails clearly | Set all labels to 0 and call `fit` | `ValueError` is raised | exists (`test_classifier_needs_defects_in_both_parts`) |
| C7 | Results can be repeated | Run the same model twice with the same seed | Same scores | planned |

### 3.3 Day-by-day simulation

| ID | What | How | Pass criterion | Status |
|---|---|---|---|---|
| S1 | No future data in the simulation | In `replay`, compare the day each model was trained on with the day it classifies | `trained_on_day <= day` in every row | exists (`test_replay_uses_only_past_days`) |
| S2 | The simulation starts on the requested day and counts all boards | Check the first day and the board counts of `summary` | First day = `start_day`; boards in the summary equal the sum over days | exists (same test) |
| S3 | No future data in the calibration | The threshold is set on held-back days that come after the training days | Calibration days are later than all training days | planned |
| S4 | The simulation works on the real data | Run `scripts/run_replay.py` and read `results/replay.json` | Run ends without error; board count equals the number of boards in the test period (17,366 for the second half of the data) | planned (assertion in the script) |

### 3.4 KPIs

| ID | KPI | How | Pass criterion | Status |
|---|---|---|---|---|
| K1 | Recall of true defects | `summary` of the simulation, on the final model | **≥ 99 %** (at most 1 % of true defects classified as false call) | open: baseline 96.3–97.8 % |
| K2 | Specificity | Same run | **≥ 20 %** (baseline 6.4 %: best model at 99 % recall, time order) | open: simulation 6.5–15.0 % at lower recall |
| K3 | Time order, not random split | Report both values for each model | Every reported KPI comes from the time-ordered evaluation; random split only shown as comparison | exists in `results/baseline.json` |
| K4 | Every number in the report has a source | Compare each number with a file in `results/` | No number without a result file | manual check before each milestone |
| K5 | The limits of the project are stated | Compare the limitations section of the README and the report with the risk list of the proposal | The text names: one production line, 132 days (101 with records), operator labels that may contain errors, anonymised data, non-commercial licence, and the line changing over time | manual check before each milestone |

If K1 or K2 is not met at the end, the project reports what works, what does not, and why (fallback in the proposal). A missed KPI is a result, not a failed test; the test is that the number is measured correctly.

## 4. Risks and the test or check for each

| Risk (proposal, README section 8) | Test or check |
|---|---|
| The line changes over time, so 99 % recall may not hold on later days | S1 and S3 (no future data); K1 on later days; drift AUROC 0.974 in `results/data_facts.json` (D7); retraining every week compared with training once (issue #38) |
| Random split overstates results | K3 |
| Labels from operators may contain errors | D1 (only 0 or 1); error analysis in the final report (#40); K5 (stated as a limit); the error rate itself cannot be tested because it is unknown (Pfab et al. 2024) |
| Only about 1 % of flagged joints are true defects (class imbalance) | C2 and C3 use recall and specificity, not accuracy; C6 |
| Only one production line and 132 days (101 days with records) | K5 (stated as a limit); D7 checks that we use all the data |
| Personal data | The data are anonymised by Siemens (Pfab et al. 2024) and contain no personal data; K5; the data are never committed (D6) |
| Data licence is non-commercial | D6; the data are never committed (README rule; check in each pull request) |
| Results cannot be repeated | C7; seeds fixed; `requirements.txt`; CI on three systems |
| Report numbers differ from result files | K4 |

## 5. Environment and responsibilities

- Python 3.11 or newer, pandas, scikit-learn, pytest. CI runs on Ubuntu, Windows and macOS with Python 3.11 and 3.12 (`.github/workflows/ci.yml`).
- Unit tests use synthetic data. Full runs use the Siemens data and take about 10 minutes (`scripts/run_replay.py`).
- Test plan and test report: Melisa Subasi. Core code and fixes: Yu Han. Error analysis: Meron Hagos. Experiments: Alejandro Rivas.
- A pull request is merged only when CI is green.

## 6. Schedule

| Date | Milestone | Test work |
|---|---|---|
| 3 Nov (P2) | Core code and baseline simulation | This plan; existing unit tests pass |
| 24 Nov (P3) | Features and testing | Planned tests D1–D3, D7, C2, C3, C7, S3, S4 added; CI status shown (#39) |
| 8 Dec (P4) | Full evaluation | Test report on the final model: K1–K4 (#47) |
| 22 Dec (P5) | Packaging | All tests pass on a clean install; run instructions tested |
| 12 Jan | Final | Test plan and evaluation section in the final report (#44) |

## 7. Notes

- Tests marked "exists" were taken from `tests/test_classification.py`; they were not re-run for this draft.
- Test IDs are new and can be used as names or comments in the code of issue #39.
