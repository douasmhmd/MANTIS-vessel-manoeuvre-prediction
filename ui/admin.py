"""Backoffice MANTIS — deux rôles, deux onglets.

    streamlit run ui/admin.py --server.port 8502

Onglet 1 · Supervision opérationnelle : le service fonctionne-t-il, que produit-il ?
Onglet 2 · Ingénierie ML : dérive, validation, registre, réentraînement.
(En production, le RBAC du gateway sépare ces deux vues ; ici, deux onglets.)
"""
from __future__ import annotations
import altair as alt

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "data" / "08_reporting"
JOURNAL = ROOT / "data" / "07_model_output" / "journal.db"

st.set_page_config(page_title="MANTIS — Administration", layout="wide")
st.title("MANTIS — Supervision du système")


# --------------------------------------------------------------------------- chargements
def _json(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _csv(p: Path, **kw):
    return pd.read_csv(p, **kw) if p.exists() else None


hold = _json(REP / "metrics_holdout.json")
dec = _json(REP / "monitoring_decision.json")
roll = _csv(REP / "monitoring_rolling.csv", parse_dates=["fenetre_debut", "fenetre_fin"])
rolling_eval = _csv(REP / "metrics_rolling.csv")
evid = REP / "evidently_drift.html"


def _run(pipeline: str):
    with st.spinner(f"kedro run --pipeline {pipeline} …"):
        r = subprocess.run([sys.executable, "-m", "kedro", "run", "--pipeline", pipeline],
                           cwd=ROOT, capture_output=True, text=True)
    (st.success if r.returncode == 0 else st.error)(f"{pipeline} : code retour {r.returncode}")
    with st.expander("journal d'exécution"):
        st.code((r.stdout or "")[-4000:] + (r.stderr or "")[-2000:])


tab_ops, tab_ml = st.tabs(["Supervision opérationnelle", "Ingénierie ML"])

# =========================================================================== ONGLET 1
with tab_ops:
    st.header("État du service")
    if dec:
        if dec["retrain"]:
            st.warning("Une mise à jour du modèle est recommandée par l'équipe ML — le service reste disponible.")
        else:
            st.success("Service nominal.")
    if hold:
        c = st.columns(4)
        c[0].metric("Fiabilité accostage", f"± {hold['B']['MAE']:.0f} min", "erreur moyenne")
        c[1].metric("Intervalle d'approche tenu", f"{hold['A_cqr']['coverage']:.0%}", "cible 80 %")
        c[2].metric("Fiabilité appareillage", f"± {hold['C']['MAE']:.0f} min", "erreur moyenne")
        c[3].metric("Fiabilité transit", f"± {hold['D']['MAE']:.0f} min", "erreur moyenne")

    st.header("Historique des prédictions")
    if JOURNAL.exists():
        con = sqlite3.connect(JOURNAL)
        j = pd.read_sql("SELECT created_at, event, berth, mode, lo, md, hi FROM predictions ORDER BY id DESC LIMIT 200", con)
        con.close()
        k = st.columns(3)
        k[0].metric("Prédictions journalisées", len(j))
        k[1].metric("En intervalle", int((j["mode"] == "intervalle").sum()))
        k[2].metric("Ponctuelles", int((j["mode"] == "ponctuel").sum()))
        st.dataframe(j.round(0).rename(columns={"lo": "min", "md": "attendu", "hi": "max"}),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("Aucune prédiction journalisée pour l'instant.")

# =========================================================================== ONGLET 2
with tab_ml:
    st.header("1 · Décision de monitoring")
    if dec:
        if dec["retrain"]:
            st.error("**Réentraînement recommandé** — " + " · ".join(dec["raisons"]))
        else:
            st.success("Aucune dérive au-delà des seuils.")
    if hold:
        c = st.columns(4)
        c[0].metric("B accostage — R²", f"{hold['B']['R2']:.2f}", f"MAE {hold['B']['MAE']:.1f} min")
        c[1].metric("A approche — couverture", f"{hold['A_cqr']['coverage']:.0%}", f"largeur {hold['A_cqr']['width']:.0f} min")
        c[2].metric("C appareillage — MAE", f"{hold['C']['MAE']:.1f} min", f"biais {hold['C']['biais']:+.1f}")
        c[3].metric("D transit — MAE", f"{hold['D']['MAE']:.1f} min", f"biais {hold['D']['biais']:+.1f}")
        st.caption(f"version {hold['run_id'][:8]} · holdout depuis {hold['holdout_start']}")

    st.header("2 · Dérive de performance (fenêtres glissantes)")
    if roll is not None and "flux" in roll.columns:
        e = roll[roll["flux"] == "entree"].set_index("fenetre_fin")
        s = roll[roll["flux"] == "sortie"].set_index("fenetre_fin")
        c1, c2 = st.columns(2)
        cb = [c for c in ["biais_B"] if c in e.columns]
        cs = [c for c in ["biais_C", "biais_D"] if c in s.columns]
        c1.markdown("**Biais des résidus (min)** — seuil ±1")
        c1.line_chart(e[cb].rename(columns={"biais_B": "B accostage"}))
        c1.line_chart(s[cs].rename(columns={"biais_C": "C appareillage", "biais_D": "D transit"}))
        c2.markdown("**Couverture de l'intervalle A** — cible 0,80")
        if "coverage_A" in e.columns:
            c2.line_chart(e[["coverage_A"]])

    st.header("3 · Dérive des données (Evidently)")
    if evid.exists():
        st.link_button("Ouvrir le rapport Evidently en pleine page", "http://127.0.0.1:8000/report/evidently")
        st.components.v1.html(open(evid, encoding="utf-8").read(), width=1600, height=700, scrolling=True)
    else:
        st.caption("Rapport non généré — lancer le pipeline monitoring.")

    st.header("4 · Validation rolling-origin (protocole)")
    if rolling_eval is not None:
        st.dataframe(rolling_eval.round(3), use_container_width=True, hide_index=True)

    st.header("5 · Actions")
    a1, a2, a3 = st.columns(3)
    if a1.button("Réentraîner (pipeline training)"):
        _run("training")
    if a2.button("Rescorer + monitoring"):
        _run("inference")
        _run("monitoring")
    if a3.button("Recalculer les features"):
        _run("feature")

    st.header("6 · Registre MLflow")
    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        rows = []
        for rm in client.search_registered_models():
            for v in client.search_model_versions(f"name='{rm.name}'"):
                rows.append({"modèle": rm.name, "version": int(v.version),
                             "alias": ", ".join(v.aliases) if v.aliases else "—",
                             "créé": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
                             "run": v.run_id[:8]})
        st.dataframe(pd.DataFrame(rows).sort_values(["modèle", "version"]), use_container_width=True, hide_index=True)
        st.caption("Rollback = réassigner l'alias « production » à une version antérieure.")
    except Exception as e:  # noqa: BLE001
        st.warning(f"MLflow indisponible : {e}")