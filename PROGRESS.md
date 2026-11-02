# Progress

Last update: 2 Nov 2026

## Milestones

Each milestone follows the course requirements for the Software & Simulation track.

| Milestone | Date | Course requirement | What we deliver | Status |
|---|---|---|---|---|
| **P1** | 20 Oct · revised 23 Oct | Proposal + tech stack + MVP | Revised one-page proposal ([`docs/proposal.pdf`](docs/proposal.pdf)), slides ([`docs/slides_revised_topic.pdf`](docs/slides_revised_topic.pdf)), [README](README.md), data set, baseline and first day-by-day simulation running (three models) | ✅ done |
| **P2** | 3 Nov | Core code + baseline sim; test plan + report | Core code: our reimplementation of Pfab & Rothering (2025), threshold corrected, retraining added; baseline reproduces their problem; test plan as a design of experiments; bibliography report (18 sources); slides ([`docs/slides_p2.pdf`](docs/slides_p2.pdf)) | ✅ done |
| **P3** | 24 Nov | Feature progress; unit/CI status; issues board | Run the design of experiments: threshold first, then retraining; unit tests on the core code; tasks as issues | ⏳ next |
| **P4** | 8 Dec | Feature progress; unit/CI status; issues board | Full evaluation; simulation software with threshold and retraining as settings | ⏳ planned |
| **P5** | 22 Dec | Demo video + run instructions; packaging status | Packaging, run instructions, demo video | ⏳ planned |
| **Final** | 12 Jan | Final presentation, report | Final presentation and report | ⏳ planned |

## Timeline

![Project schedule](docs/img/gantt.png)

Tasks and owners are tracked as [issues](../../issues), grouped by [milestone](../../milestones).

## Tech stack

Python · pandas · scikit-learn (random forest) · imbalanced-learn (balanced random forest) · matplotlib and Jupyter (figures, notebook) · pytest and GitHub Actions (tests on every push) — all from the lecture's data science tools (p. 22)

## Log

- **2 Nov 2026** · P2 complete: slides, bibliography report (18 sources), test plan as a design of experiments, data understanding notebook. Core code reimplements Pfab & Rothering (2025); the baseline shows the same problem (5–14 % missed true defects on later periods). The literature review found this study on the same data; its two named next steps, threshold and retraining, are now the focus of our project.
- **23 Oct 2026** · Revised topic submitted: binary classification of AOI alarms (true defect or false call) to reduce false calls (Siemens data). Repository renamed and rewritten for the new topic. Three standard models (scikit-learn) and a day-by-day simulation of the line running; unit tests and CI set up.
- **22 Oct 2026** · Siemens AOI data set (Pfab et al. 2024) downloaded and checked: 440,274 flagged joints, 39,742 boards, strong drift over time. First baseline: time-ordered evaluation is much harder than a random split.
- **21 Oct 2026** · Siemens and NVIDIA announce an AI second check of AOI results in Erlangen (about 40 % fewer false calls targeted), confirming the problem is current.
- **20 Oct 2026** · P1: the first topic (Planner Copilot, an AI assistant for production-planning simulations) was judged too broad and not focused on one production problem. Decision: narrow to one concrete problem from the lecture examples (p. 18, Siemens quality control).
