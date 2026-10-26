"""Service d'inférence MANTIS — utilisé par l'API (en ligne) et par le pipeline Kedro (batch).

Le même code construit les features « à l'instant t » qu'à l'entraînement :
    - contexte : en_cours, occup_term, attente_term calculés sur la table de référence
    - encodages historiques : moyennes des manœuvres passées du même navire / quai
Événements pris en charge (précision croissante) :
    ETA → PORT_LIMIT → PILOT_ONBOARD → JETTY   (entrée)
    DEPARTURE_REQUEST                          (sortie)
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TERMS = ["TC1", "TC2", "TC3", "TC4"]
EVENTS_ENTREE = {"ETA", "PORT_LIMIT", "PILOT_ONBOARD", "JETTY"}
EVENTS_SORTIE = {"DEPARTURE_REQUEST"}


@dataclass
class Event:
    imo: str
    berth: str
    event: str
    timestamp: str
    length: float
    after_draft: float | None = None
    forward_draft: float | None = None
    nb_tugs: float | None = None
    nb_pilots: float = 1.0
    eta: str | None = None
    jetty_arrival: str | None = None        # pour DEPARTURE_REQUEST (séjour à quai)
    vessel_name: str | None = None          # jamais transmis au LLM

    @property
    def t(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamp)

    @property
    def term(self) -> str:
        return str(self.berth)[:3]


def _count_active(t: np.datetime64, starts: np.ndarray, ends: np.ndarray) -> int:
    starts, ends = np.sort(starts[~np.isnat(starts)]), np.sort(ends[~np.isnat(ends)])
    return int(np.searchsorted(starts, t, "left") - np.searchsorted(ends, t, "right"))


@dataclass
class MantisPredictor:
    model_dir: str
    ref_manoeuvres: str
    ref_entree: str
    ref_sortie: str
    navigation_min: float = 26.0
    navigation_tol: float = 8.0
    features_entree: list[str] = field(default_factory=list)
    features_sortie: list[str] = field(default_factory=list)

    def __post_init__(self):
        d = Path(self.model_dir)
        self.models = {n: pickle.load(open(d / f"model_{n}.pkl", "rb")) for n in ["B", "A_cqr", "C", "D"]}
        self.M = pd.read_parquet(self.ref_manoeuvres)
        self.E = pd.read_parquet(self.ref_entree).sort_values("t_ref")
        self.S = pd.read_parquet(self.ref_sortie).sort_values("t_ref")
        self.occ = self.M[self.M["MovementTypeID"] == 1][["term", "PortLimitArrival", "JettyArrival", "JettyDeparture"]].dropna()
        self.version = json.load(open(Path(self.model_dir).parent / "08_reporting" / "metrics_holdout.json")).get("run_id", "local")

    # ------------------------------------------------------------------ features
    def _context(self, ev: Event) -> dict:
        t = np.datetime64(ev.t)
        sub = self.occ[self.occ["term"] == ev.term]
        return {
            "en_cours": _count_active(t, self.M["Mouvement_ActualStartDate"].values, self.M["Mouvement_ActualFinishDate"].values),
            "occup_term": _count_active(t, sub["JettyArrival"].values, sub["JettyDeparture"].values),
            "attente_term": _count_active(t, sub["PortLimitArrival"].values, sub["JettyArrival"].values),
        }

    def _history(self, ev: Event, ref: pd.DataFrame, targets: list[str], min_quai=5, min_nav=3) -> dict:
        past = ref[ref["t_ref"] < ev.t]
        nav, quai = past[past["IMONumber"] == str(ev.imo)], past[past["BerthName"] == ev.berth]
        out = {"nb_escales_prev": int(len(nav))}
        for tg in targets:
            out[f"navire_hist_{tg}"] = float(nav[tg].mean()) if len(nav) >= min_nav else np.nan
            out[f"quai_hist_{tg}"] = float(quai[tg].mean()) if len(quai) >= min_quai else np.nan
        return out

    def _physical(self, ev: Event) -> dict:
        ad = ev.after_draft if ev.after_draft and 3 <= ev.after_draft <= 25 else np.nan
        fd = ev.forward_draft if ev.forward_draft and 3 <= ev.forward_draft <= 25 else np.nan
        regle = float(np.digitize(ev.length, [200, 350]) + 1)
        tugs = ev.nb_tugs if ev.nb_tugs is not None else regle
        return {"Length": ev.length, "AfterDraft": ad, "assiette": ad - fd if not np.isnan(ad) and not np.isnan(fd) else np.nan,
                "nb_tugs": min(tugs, 4), "nb_tugs_regle": regle, "ecart_tugs": min(tugs, 4) - regle,
                "nb_pilots": min(ev.nb_pilots, 2), "term_c": TERMS.index(ev.term) if ev.term in TERMS else -1}

    def _calendar(self, ev: Event) -> dict:
        h = ev.t.hour
        return {"releve": int(h in (8, 9)), "heure_sin": np.sin(2 * np.pi * h / 24), "heure_cos": np.cos(2 * np.pi * h / 24)}

    def features_entree_at(self, ev: Event) -> pd.DataFrame:
        f = {**self._physical(ev), **self._calendar(ev), **self._context(ev),
             **self._history(ev, self.E, ["phase_A", "phase_B"])}
        f["retard_ETA_w"] = float(np.clip((ev.t - pd.Timestamp(ev.eta)).total_seconds() / 60, -1440, 2880)) if ev.eta else 0.0
        return pd.DataFrame([f])[self.features_entree]

    def features_sortie_at(self, ev: Event) -> pd.DataFrame:
        f = {**self._physical(ev), **self._calendar(ev), **self._context(ev),
             **self._history(ev, self.S, ["phase_C", "phase_D"])}
        f["sejour_quai_h"] = (ev.t - pd.Timestamp(ev.jetty_arrival)).total_seconds() / 3600 if ev.jetty_arrival else np.nan
        return pd.DataFrame([f])[self.features_sortie]

    # ------------------------------------------------------------------ prédiction
# ------------------------------------------------------------------ prédiction
    def predict(self, ev: Event) -> dict:
        """Les durées de phase sont arrondies une seule fois, à la source ; toutes les
        bornes agrégées en découlent, de sorte que lo == phase_A_lo + phase_B."""
        if ev.event in EVENTS_SORTIE:
            X = self.features_sortie_at(ev)
            C = round(float(self.models["C"].predict(X)[0]))
            D = round(float(self.models["D"].predict(X)[0]))
            tot = C + D
            out = {"phase_C": C, "phase_D": D, "total": tot, "lo": tot - 10, "hi": tot + 10,
                   "mode": "ponctuel",
                   "fin_estimee": (ev.t + pd.Timedelta(minutes=tot)).isoformat(timespec="minutes")}
        elif ev.event in EVENTS_ENTREE:
            X = self.features_entree_at(ev)
            B = round(float(self.models["B"].predict(X)[0]))
            if ev.event in ("ETA", "PORT_LIMIT"):
                a = self.models["A_cqr"].predict(X).iloc[0]
                A_lo, A_md, A_hi = round(float(a["lo"])), round(float(a["md"])), round(float(a["hi"]))
                mode = "intervalle"
            elif ev.event == "PILOT_ONBOARD":
                A_lo = round(self.navigation_min - self.navigation_tol)
                A_md = round(self.navigation_min)
                A_hi = round(self.navigation_min + self.navigation_tol)
                mode = "ponctuel"
            else:  # JETTY : la phase A est terminée
                A_lo = A_md = A_hi = 0
                mode = "ponctuel"
            out = {"phase_A_lo": A_lo, "phase_A": A_md, "phase_A_hi": A_hi, "phase_B": B,
                   "total": A_md + B, "lo": A_lo + B, "hi": A_hi + B, "mode": mode,
                   "accostage_estime": (ev.t + pd.Timedelta(minutes=A_md)).isoformat(timespec="minutes"),
                   "fin_estimee": (ev.t + pd.Timedelta(minutes=A_md + B)).isoformat(timespec="minutes"),
                   "risque_attente": "élevé" if (A_hi - A_lo) > 120
                                     else "modéré" if (A_hi - A_lo) > 80 else "faible"}
        else:
            raise ValueError(f"événement inconnu : {ev.event}")
        ctx = {k: v for k, v in X.iloc[0].to_dict().items()
               if k in ("occup_term", "attente_term", "en_cours", "releve")}
        return {"event": ev.event, "timestamp": ev.timestamp, "berth": ev.berth, "term": ev.term,
                "model_version": self.version, **out, "contexte": ctx}

    # ------------------------------------------------------------------ journal
    @staticmethod
    def journal_write(db_path: str, ev: Event, pred: dict) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, imo TEXT, berth TEXT, event TEXT, event_ts TEXT,
            model_version TEXT, mode TEXT, lo REAL, md REAL, hi REAL, payload TEXT)""")
        con.execute("INSERT INTO predictions (created_at, imo, berth, event, event_ts, model_version, mode, lo, md, hi, payload) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (pd.Timestamp.now().isoformat(), str(ev.imo), ev.berth, ev.event, ev.timestamp, pred["model_version"],
                     pred["mode"], pred["lo"], pred["total"], pred["hi"], json.dumps(pred, default=str)))
        con.commit(); con.close()

    @staticmethod
    def journal_read(db_path: str, limit: int = 200) -> pd.DataFrame:
        if not Path(db_path).exists():
            return pd.DataFrame()
        con = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {int(limit)}", con)
        con.close()
        return df