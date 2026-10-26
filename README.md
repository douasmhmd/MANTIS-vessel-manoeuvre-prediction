# MANTIS — Vessel Manoeuvre Duration Prediction

**M**anoeuvre **AN**alytics and **T**ime **I**nference **S**ystem — an end-to-end MLOps system
predicting arrival and departure manoeuvre durations for container ships at **Port Tanger Med**,
the largest container port in Africa and the Mediterranean.

Built during a 2-month engineering internship at **Tanger Med Port Authority (TMPA)**.
Academic project — ESI Rabat, ISITD program.

---

## TL;DR

| | |
|---|---|
| **Problem** | Predict how long a container ship takes to enter or leave the port, to enable Just-In-Time port call planning (IMO recommendation) |
| **Data** | 210,474 raw records from the Port Management System, Jan 2021 – Jul 2025 → **30,557 unique manoeuvres** after deduplication |
| **Key finding** | A manoeuvre is **not one process**. It splits into two statistically independent phases: one physical and predictable, one that is a *resource-assignment delay the information system never records* |
| **Delivered** | Point predictions where the process is regular (R² 0.53, MAE 8.7 min), **calibrated 80 % intervals** where it is not (coverage 80.5 %) |
| **Stack** | Kedro · MLflow · scikit-learn · FastAPI · Evidently · React · Docker |

> The headline result is not a score. It is that **the predictability ceiling is a data-design
> limit, not an algorithmic one** — and the report quantifies exactly how much a single extra
> column in the port's information system would move it.

---

## 1. The analytical result that drives the architecture

Most of this repository's design decisions follow from four findings. They are worth reading
before the code.

### 1.1 The data granularity is not what it appears

The Port Management System writes **one row per resource assigned** (pilot, tug), because that
is the billing unit. One manoeuvre with a pilot and two tugs appears as three identical rows.

- Duplication factor: **×2.53**
- It is *not uniform*: larger ships use more resources, so they are systematically overweighted
- A random train/test split places copies of the same manoeuvre on both sides

Consequence: the statistical unit is the **manoeuvre** (`MovementID`), never the row.

### 1.2 A manoeuvre decomposes into four phases

| Phase | Boundaries (PMS timestamps) | Median | CV | Share of variance |
|---|---|---|---|---|
| **A** — approach | `PortLimitArrival → JettyArrival` | 49 min | 0.89 | **98.7 %** |
| **B** — berthing | `JettyArrival → ActualFinish` | 42 min | 0.37 | 6.3 % |
| **C** — unberthing | `ActualStart → JettyDeparture` | 18 min | 0.47 | **78.4 %** |
| **D** — transit | `JettyDeparture → PortLimitDeparture` | 15 min | 0.28 | 15.7 % |

`corr(A, B) = −0.10` · `corr(C, D) = +0.09` — the phases of a manoeuvre are **independent
processes**. Modelling the total duration averages two unrelated regularities and loses the
predictable one.

### 1.3 Phase A is a waiting time, not a manoeuvre

The PMS records `Mouvement_ActualStartDate = PortLimitArrival` for **100 %** of arrivals. The
recorded "manoeuvre duration" therefore starts when the ship crosses the port limit — pilot
assigned or not.

- Spearman correlation between phase A and the **pilot mobilisation delay**: **0.94**
- Residual navigation time once the delay is removed: median **26 min**, p5–p95 = 14–49 min
- Among ships served immediately (pilot ≤ 15 min), **2 out of 6,237** have an extreme approach —
  versus **8.6 %** among ships put on hold
- Same ship, same berth, 169 calls: entry ranges from **38 to 375 min**; exit stays at 19–56 min

Nothing in the ship explains it. Nothing in the recorded context explains more than ρ = 0.26.
**The cause is a decision the information system does not store.**

### 1.4 The process drifts

From Q2 2024, median unberthing time jumps 17 → 22 min and berthing 40 → 45 min, while the
median vessel length grows 229 m → 260 m. Phase A does **not** drift. Drift monitoring must
therefore be differentiated per model, not global.

---

## 2. Honest evaluation — and what a careless protocol would have reported

Same model, same hyperparameters, same target (phase B). Only the data preparation and the
split change:

| Protocol | n test | R² | MAE |
|---|---:|---:|---:|
| A. Duplicated rows, global encodings, random split | 8,353 | **0.704** | 7.27 |
| B. Deduplicated, leak-free encodings, random split | 3,013 | 0.575 | 8.01 |
| C. Deduplicated, leak-free encodings, **temporal split** | 3,202 | **0.505** | 8.91 |

None of the three raises a warning at runtime. Only the comparison reveals the gap.
**This is why leak prevention is architectural here, not procedural** — see §3.

### Final results (rolling-origin, 4 folds, 6-month horizon)

| Target | R² (mean ± sd) | MAE | Naive baseline | Deliverable |
|---|---:|---:|---:|---|
| **B** — berthing | 0.539 ± 0.042 | **8.7 min** | 13.2 min | point estimate |
| **A** — approach | 0.142 ± 0.075 | 42.2 min | 42.9 min | **80 % interval** |
| **C** — unberthing | 0.084 ± 0.041 | 7.5 min | 8.2 min | point estimate |
| **D** — transit | 0.130 ± 0.023 | 2.7 min | 2.9 min | point estimate |

**Conformalized Quantile Regression on phase A** (holdout Jul 2024 → Jul 2025, n = 3,202):

- Empirical coverage **80.5 %** for an 80 % target
- Mean width **126 min** vs 150 min for an unconditional interval (**−16 %**)
- Width adapts to context: 75 min on an empty terminal → 147 min on a busy one, coverage stable
  at 78–81 % across all bands
- Quantile loss beats squared loss on the conditional median: MAE 36.6 vs 42.2 min

---

## 3. Architecture — the FTI pattern

Each layer answers a measured finding, not a convention.

```
           ┌─────────────── F · FEATURE ───────────────┐
 PMS ────► │  clean → deduplicate → 4 phases →         │
 (DWH)     │  context@t → historical encodings →       │ ──► validated
           │  Pandera schema validation                │     feature tables
           └───────────────────────────────────────────┘
                                │
           ┌─────────────── T · TRAINING ──────────────┐
           │  rolling-origin validation (4 origins)    │
           │  HistGradientBoosting  →  B, C, D         │ ──► MLflow registry
           │  CQR (q10/q50/q90 + conformal)  →  A      │     (versioned)
           └───────────────────────────────────────────┘
                                │
           ┌─────────────── I · INFERENCE ─────────────┐
           │  event-driven FastAPI service             │
           │  ETA → PORT_LIMIT → PILOT_ONBOARD →       │ ──► prediction
           │  JETTY · DEPARTURE_REQUEST                │  +  SQLite journal
           └───────────────────────────────────────────┘
                                │
           ┌───────────── MONITORING ──────────────────┐
           │  Evidently (data drift)                   │
           │  rolling bias / coverage (performance)    │ ──► retrain decision
           └───────────────────────────────────────────┘
                                │
           ┌───────────── PRESENTATION ────────────────┐
           │  React operator UI · Streamlit backoffice │
           │  LLM narration behind a strict allow-list │
           └───────────────────────────────────────────┘
```

### Why FTI, specifically

| Requirement | Finding that imposes it | Architectural answer |
|---|---|---|
| Single feature implementation | A leaked protocol inflates R² by 0.20 and raises **no runtime signal** | The inference service calls the *same* feature functions as training — a skew cannot appear by accident |
| Differentiated monitoring | Drift hits physical phases, spares the waiting phase | Independently observable pipelines; bias for point estimates, coverage for intervals |
| Reproducibility | ISO/IEC 27001 + 20000 certified IS at TMPA | Versioned features and models; every prediction reconstructible |
| Non-intrusiveness | Production system must not be queried by analytics | Consumes a data warehouse, never the PMS directly |

### Event-driven inference

Precision is not a property of the model — it is a property of **when you ask**.

| Event | What is known | Output |
|---|---|---|
| `ETA` / `PORT_LIMIT` | Ship + port context; pilot **not yet assigned** | Interval, e.g. **67 – 222 min** + wait-risk level |
| `PILOT_ONBOARD` | The wait has resolved | Point estimate, **62 – 78 min** |
| `JETTY` | Only berthing remains | ± 9 min |
| `DEPARTURE_REQUEST` | — | Unberthing + transit |

A single port call, four API calls, an interval that narrows from **156 min to 16 min** — not
because the model improved, but because the uncertainty resolved.

### Data minimisation for the LLM layer

The narration layer *verbalises* numbers; it never computes them. It receives a schema-enforced
allow-list only:

```jsonc
// transmitted
{ "manoeuvre": "entree", "mode": "intervalle",
  "phase_A_lo": 22, "phase_A": 49, "phase_A_hi": 177, "phase_B": 45,
  "total_lo": 67, "total": 94, "total_hi": 222,
  "charge_terminal": "faible", "file_attente": 0, "releve": true,
  "classe_taille": "> 300 m (post-Panamax)", "risque_attente": "eleve",
  "facteurs_risque": ["creneau de releve des pilotes"],
  "facteurs_favorables": ["aucun navire en attente", "charge du terminal faible"] }

// never transmitted: vessel name, IMO, voyage number, agent, charterer,
// pilot/tug identities, raw timestamps, exact berth
```

A deterministic template takes over on any failure — the operator never waits on an external
service. The prototype calls an external API for proof of concept; **the target is a local
model on TMPA infrastructure**, and the swap costs one client change since the interface is a
plain structured document.

---

## 4. Repository layout

```
conf/base/
  catalog.yml              dataset declarations (logical names ↔ storage)
  parameters*.yml          every threshold, window and hyperparameter — no hard-coded values
src/mantis/
  pipelines/feature/       6 nodes: clean → manoeuvres → split → context → history → validate
  pipelines/training/      5 nodes: rolling eval → B / A-CQR / C+D → MLflow registry
  pipelines/inference/     2 nodes: batch scoring of the holdout (feeds monitoring)
  pipelines/monitoring/    3 nodes: Evidently report → rolling metrics → retrain decision
  schemas.py               Pandera contracts between F, T and I
  models.py                HGB factory + CQRModel (conformalized quantile regression)
  inference.py             shared prediction service — used by both the API and the pipeline
  narrate.py               minimisation allow-list + deterministic fallback
api/main.py                FastAPI: /predict · /narrate · /journal · /health
ui-react/                  operator interface (Vite + React)
ui/admin.py                backoffice: ops supervision / ML engineering tabs
docker-compose.yml         mlflow · api · operator · admin
```

---

## 5. Running it

```bash
pip install -r requirements.txt && pip install -e .
kedro run                     # full chain: feature → training → inference → monitoring (~146 s, 16 nodes)
kedro viz run                 # pipeline DAG
uvicorn api.main:app --port 8000
cd ui-react && npm install && npm run dev
```

| Service | URL |
|---|---|
| API docs | http://127.0.0.1:8000/docs |
| Operator UI | http://localhost:5173 |
| Backoffice | http://localhost:8502 |
| MLflow | http://localhost:5000 |

Runtime on an ordinary laptop, no GPU:

| Pipeline | Nodes | Duration | Artefacts |
|---|---:|---:|---|
| `feature` | 6 | 23 s | validated feature tables |
| `training` | 5 | 93 s | 4 models + metrics + registry entries |
| `inference` | 2 | 16 s | scored holdout joined to ground truth |
| `monitoring` | 3 | 11 s | drift report, rolling metrics, retrain decision |

> **Data is not included.** The PMS export is TMPA operational data and is not distributable.
> The pipelines expect a CSV at `data/01_raw/`; the schema is documented in the report appendix.

---

## 6. What I would not claim

Stated plainly, because a reviewer will ask:

- **Phase A is not well predicted** — R² 0.14, and the point estimate barely beats the median.
  That is the honest result; the interval is the usable deliverable.
- **No embargo** between train and test folds. Historical encodings create a weak dependency
  between temporally adjacent manoeuvres that this does not cancel.
- **Interval propagation is partial** — the total interval sums A's bounds with B's point
  estimate. Calibrating B by CQR would make it exact; the machinery is already there.
- **Not validated in production.** Evaluated on historical data; operator adoption and
  year-long behaviour remain to be observed.
- **Not implemented:** enterprise feature store, centralised authentication, local LLM. These
  are integration work with TMPA's IS, not design gaps — each swaps behind an unchanged interface.

---

## 7. Recommendations delivered to TMPA

Ordered by effort, which is the inverse of how they are usually prioritised:

1. **Keep the *planned* resource-assignment timestamp** (currently overwritten by the actual).
   One non-overwritten field, correlated 0.94 with the target. This single change would make
   phase A nearly trivial to predict — `A ≈ planned_delay + 26 min` — with no model change.
2. **Date the real start of the arrival manoeuvre**, as is already done for departures.
   Turns one unusable sum into two usable indicators: a manoeuvre duration and a pilotage KPI.
3. **Attach hourly weather data.** Atypical manoeuvres carry the signature of degraded
   conditions the PMS does not record.
4. **Build a data layer designed for models**, not a by-product of transactions.

> An information system that records *events* makes dashboards possible.
> One that records *decisions* makes prediction possible.

---

## Tech stack

`Python 3.12` · `Kedro 1.5` · `scikit-learn` · `Pandera` · `MLflow 3.16` · `Evidently` ·
`SHAP` · `FastAPI` · `React 19 (Vite)` · `Streamlit` · `Docker Compose`

## Author

**Mohamed Douas** — SI engineering and digital transformation  , 
[LinkedIn](https://www.linkedin.com/in/douas-mohamed-146250285) · mohamed.douas@esi.ac.ma

