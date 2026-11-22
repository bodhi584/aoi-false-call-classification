# P3 results on the development data

Days 1-63 only: train on days 1-22, evaluate on days 23-63 in 5 periods (time order). Missed = released true defects / all true defects; released = released false calls / all false calls; mean over the 5 periods and the seeds; worst period = highest period mean. Made by `python scripts/p3_pipeline.py report` from `results/p3_results.json`.

| Setting (BRF) | Seeds | Missed, mean | Missed, worst period | Released | Seeds with every period ≤ 1 % |
|---|---|---|---|---|---|
| Method of [7]: trained once, CV threshold | 10 | 10.72 % | 17.38 % | 33.9 % | 0 / 10 |
| Retraining per period | 10 | 3.23 % | 5.93 % | 39.7 % | 0 / 10 |
| Retraining weekly | 10 | 2.12 % | 3.96 % | 29.5 % | 0 / 10 |
| Retraining every 3 days | 10 | 3.36 % | 7.27 % | 37.5 % | 0 / 10 |
| Retraining daily | 10 | 2.92 % | 3.60 % | 46.7 % | 0 / 10 |
| Weekly + threshold q = 0 | 10 | 0.24 % | 0.72 % | 3.9 % | 4 / 10 |
| Weekly + threshold q = 0.1 % | 10 | 0.62 % | 1.74 % | 5.7 % | 0 / 10 |
| Weekly + threshold q = 0.2 % | 10 | 0.90 % | 2.39 % | 7.1 % | 0 / 10 |
| Weekly + threshold q = 0.5 % | 10 | 1.18 % | 2.68 % | 13.7 % | 0 / 10 |
| Weekly + threshold q = 1 % | 10 | 1.42 % | 3.19 % | 19.7 % | 0 / 10 |
| Weekly + threshold q = 2 % | 10 | 3.03 % | 7.78 % | 27.7 % | 0 / 10 |
