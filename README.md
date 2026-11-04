# Binary Classification of AOI Alarms into True Defects and False Calls

Second check of AOI results with supervised machine learning (Siemens production data)
Case Study *Intelligent Systems in Production* · THD · Team 1 · Prof. Dr.-Ing. Hamidreza Heidari
Yu Han · Melisa Subasi · Meron Hagos · Alejandro Rivas

## Results by milestone

| Milestone | Course requirement | Results |
|---|---|---|
| **P1** · 20 Oct (revised 4 Nov) | proposal, tech stack, MVP | [one-page proposal](docs/proposal.pdf) ([Word](docs/proposal.docx)) · [slides](docs/slides_p1.pdf) |
| **P2** · 3 Nov | core code + baseline simulation; test plan + report | [report](docs/literature_report.pdf) · [slides](docs/slides_p2.pdf) · [test plan](docs/test_plan.md) · [core code](scripts/replicate_pfab2025.py) · [baseline result](results/replication_pfab2025.json) · [data understanding](notebooks/data_understanding.ipynb) |
| **P3** · 24 Nov | feature progress; unit tests and CI; issues board | planned: threshold methods and retraining against the baseline · [issues](../../issues) |
| **P4** · 8 Dec | feature progress; unit tests and CI; issues board | planned: full design of experiments, evaluation, simulation tool |
| **P5** · 22 Dec | demo video, run instructions, packaging | planned |
| **Final** · 12 Jan | final presentation and report | planned |

Numbers in brackets refer to the report. [7] = Pfab & Rothering (2025), our core paper.

## The problem

![What a second check has reached so far](docs/img/benefit_pfab2025.png)

About 99 % of AOI alarms are false calls, and people check every one. A second check with machine learning releases enough false calls on almost all periods (≥ 40 %), but on later production periods it misses 5–15 % of the true defects instead of at most 1 % (D, E). Only the test set from the training period is inside the target zone (B, C).

## Objective

> On every later production period: **missed true defects ≤ 1 %** (hard quality limit) and **false calls released ≥ 40 %**, as many as possible. Expected saving: between 40 % and the 80–90 % reached on the test set. (Report, Section 2.4)

## System

![System: classifier with two adaptation strategies](docs/img/system_diagram.png)

Supervised binary classifier (random forest, balanced random forest, as in [7]) with two adaptation strategies: **decision threshold selection** (Neyman–Pearson threshold [14], prior shift adjustment [15]) and **concept drift adaptation** by retraining (periodic [12], [17], [18], drift-triggered [16]). (Report, Section 2.3)

## Baseline: our code shows the same problem as [7]

![Baseline](docs/img/reproduction_pfab2025.png)

| Model (10 seeds) | Missed true defects, test set: [7] / ours | Later periods: [7] / ours |
|---|---|---|
| Balanced random forest | 0.7 % / 1.5 % | 7.5 % / 7.5 % |
| Random forest | 0.7 % / 1.9 % | 11.3 % / 9.1 % |

## Test plan (P3)

Threshold method 3 × retraining schedule 6 × training window 2 × model 2 = 72 settings, each with 10 seeds. Details: [`docs/test_plan.md`](docs/test_plan.md).

## Run

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # Siemens data (CC BY-NC 3.0, not stored here)
python scripts/replicate_pfab2025.py runs.csv   # runs.csv: run log of [7], src/fcr_case_study/notebooks/runs/runs.csv
                                                # in gitlab.com/research9841542/FCR_1
```

Code: MIT licence.
