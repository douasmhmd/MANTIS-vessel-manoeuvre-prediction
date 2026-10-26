"""API MANTIS — inférence événementielle.

    uvicorn api.main:app --reload --port 8000
    POST /predict   {imo, berth, event, timestamp, length, ...}
    GET  /journal   dernières prédictions journalisées
    GET  /health
"""
from __future__ import annotations
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mantis.narrate import narrate


from mantis.inference import EVENTS_ENTREE, EVENTS_SORTIE, Event, MantisPredictor

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {}
for f in sorted((ROOT / "conf" / "base").glob("parameters*.yml")):
    PARAMS.update(yaml.safe_load(open(f, encoding="utf-8")) or {})
P_INF, P_TR = PARAMS["inference"], PARAMS["training"]
app = FastAPI(title="MANTIS — prédiction des manœuvres", version="0.1")
predictor: MantisPredictor | None = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class EventIn(BaseModel):
    imo: str = Field(..., examples=["9776171"])
    berth: str = Field(..., examples=["TC21"])
    event: str = Field(..., description="ETA | PORT_LIMIT | PILOT_ONBOARD | JETTY | DEPARTURE_REQUEST")
    timestamp: str = Field(..., examples=["2025-03-14T08:20:00"])
    length: float = Field(..., examples=[334.0])
    after_draft: float | None = None
    forward_draft: float | None = None
    nb_tugs: float | None = None
    nb_pilots: float = 1.0
    eta: str | None = None
    jetty_arrival: str | None = None
    vessel_name: str | None = None


@app.on_event("startup")
def _load():
    global predictor
    predictor = MantisPredictor(
        model_dir=str(ROOT / P_INF["model_dir"]), ref_manoeuvres=str(ROOT / P_INF["ref_manoeuvres"]),
        ref_entree=str(ROOT / P_INF["ref_entree"]), ref_sortie=str(ROOT / P_INF["ref_sortie"]),
        navigation_min=P_INF["navigation_min"], navigation_tol=P_INF["navigation_tol"],
        features_entree=P_TR["features_entree"], features_sortie=P_TR["features_sortie"])


@app.get("/health")
def health():
    return {"status": "ok", "model_version": predictor.version if predictor else None}


@app.post("/predict")
def predict(ev: EventIn):
    if ev.event not in EVENTS_ENTREE | EVENTS_SORTIE:
        raise HTTPException(400, f"événement inconnu : {ev.event}")
    event = Event(**ev.model_dump())
    try:
        pred = predictor.predict(event)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
    MantisPredictor.journal_write(str(ROOT / P_INF["journal_path"]), event, pred)
    return pred


@app.get("/journal")
def journal(limit: int = 50):
    df = MantisPredictor.journal_read(str(ROOT / P_INF["journal_path"]), limit)
    return df.drop(columns=["payload"], errors="ignore").to_dict(orient="records")

class NarrateIn(BaseModel):
    pred: dict
    length: float | None = None
    nb_tugs: float | None = None


@app.post("/narrate")
def narrate_endpoint(body: NarrateIn):
    return narrate(body.pred, body.length, body.nb_tugs)


@app.get("/report/evidently", response_class=HTMLResponse)
def report_evidently():
    p = ROOT / "data" / "08_reporting" / "evidently_drift.html"
    if not p.exists():
        raise HTTPException(404, "rapport non généré")
    return p.read_text(encoding="utf-8")