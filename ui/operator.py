"""UI opérateur MANTIS — une prédiction par événement, qui se précise au fil de l'escale.

    streamlit run ui/operator.py
"""
from __future__ import annotations

import datetime as dt
import altair as alt

import pandas as pd
import requests
import streamlit as st

import os
API = st.sidebar.text_input("API", os.getenv("MANTIS_API", "http://127.0.0.1:8000"))
st.set_page_config(page_title="MANTIS — Opérateur", layout="wide")
st.title("MANTIS — Prédiction des manœuvres")
st.caption("Porte-conteneurs · Tanger Med · la prédiction se précise à chaque événement")

if "timeline" not in st.session_state:
    st.session_state.timeline = []

# --------------------------------------------------------------------------- formulaire
with st.form("event"):
    c1, c2, c3 = st.columns(3)
    imo = c1.text_input("IMO", "9776171")
    berth = c1.selectbox("Quai", ["TC11", "TC12", "TC13", "TC14", "TC21", "TC22", "TC23", "TC24", "TC31", "TC32",
                                  "TC41", "TC42", "TC43", "TC44"], index=4)
    event = c1.selectbox("Événement", ["ETA", "PORT_LIMIT", "PILOT_ONBOARD", "JETTY", "DEPARTURE_REQUEST"], index=1)
    d = c2.date_input("Date", dt.date(2025, 3, 14)); t = c2.time_input("Heure", dt.time(8, 20))
    eta = c2.text_input("ETA annoncée (entrée)", "2025-03-14T06:00:00")
    jetty = c2.text_input("Arrivée à quai (sortie)", "")
    length = c3.number_input("Longueur (m)", 80.0, 450.0, 334.0)
    ad = c3.number_input("Tirant d'eau arrière (m)", 0.0, 25.0, 12.5)
    fd = c3.number_input("Tirant d'eau avant (m)", 0.0, 25.0, 11.8)
    tugs = c3.number_input("Remorqueurs planifiés (0 = règle)", 0, 4, 0)
    go = st.form_submit_button("Prédire")

if go:
    body = {"imo": imo, "berth": berth, "event": event, "timestamp": f"{d}T{t}", "length": length,
            "after_draft": ad or None, "forward_draft": fd or None, "nb_tugs": tugs or None,
            "eta": eta or None, "jetty_arrival": jetty or None}
    try:
        pred = requests.post(f"{API}/predict", json=body, timeout=30).json()
        narr = requests.post(f"{API}/narrate", json={"pred": pred, "length": length, "nb_tugs": tugs or None}, timeout=30).json()
        st.session_state.timeline.append({"body": body, "pred": pred, "narr": narr})
    except Exception as e:  # noqa: BLE001
        st.error(f"API injoignable : {e}")

# --------------------------------------------------------------------------- affichage
if st.session_state.timeline:
    last = st.session_state.timeline[-1]
    p = last["pred"]
    st.subheader(f"{p['event']} · {p['berth']} · {p['timestamp'][:16].replace('T', ' ')}")
    st.info(last["narr"]["text"])
    st.caption(f"source du texte : {last['narr']['source']} · modèle {p['model_version'][:8]}")

    m1, m2, m3, m4 = st.columns(4)
    if p["event"] != "DEPARTURE_REQUEST":
        m1.metric("Approche (A)", f"{p['phase_A']:.0f} min", f"{p['phase_A_lo']:.0f}–{p['phase_A_hi']:.0f}" if p["mode"] == "intervalle" else "±8")
        m2.metric("Accostage (B)", f"{p['phase_B']:.0f} min")
        m3.metric("Accostage estimé", p["accostage_estime"][11:16])
        m4.metric("Fin de manœuvre", p["fin_estimee"][11:16], f"risque {p['risque_attente']}")
    else:
        m1.metric("Appareillage (C)", f"{p['phase_C']:.0f} min"); m2.metric("Transit (D)", f"{p['phase_D']:.0f} min")
        m3.metric("Sortie limite portuaire", p["fin_estimee"][11:16])

    st.markdown("**Contexte à l'instant de l'événement**")
    st.json(p["contexte"], expanded=False)

    st.markdown("---"); st.markdown("**Chronologie de l'escale — l'intervalle se resserre**")
    rows = [{"événement": x["pred"]["event"], "heure": x["pred"]["timestamp"][11:16],
             "fin estimée": x["pred"]["fin_estimee"][11:16], "min": round(x["pred"]["lo"]),
             "attendu": round(x["pred"]["total"]), "max": round(x["pred"]["hi"]), "mode": x["pred"]["mode"]}
            for x in st.session_state.timeline]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    base = alt.Chart(df).encode(x=alt.X("événement:N", sort=None))
    st.altair_chart(
        base.mark_rule(size=8, color="#4C72B0").encode(y="min:Q", y2="max:Q")
        + base.mark_point(size=120, color="white", filled=True).encode(y="attendu:Q"),
        use_container_width=True)

    with st.expander("JSON transmis au LLM (minimisé — aucun identifiant)"):
        st.json(last["narr"]["payload"])

    if st.button("Nouvelle escale"):
        st.session_state.timeline = []
        st.rerun()