import { useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const QUAIS = ["TC11","TC12","TC13","TC14","TC21","TC22","TC23","TC24",
               "TC31","TC32","TC41","TC42","TC43","TC44"];

const EVENTS = [
  { id: "ETA",               label: "Annonce ETA" },
  { id: "PORT_LIMIT",        label: "Arrivée limite portuaire" },
  { id: "PILOT_ONBOARD",     label: "Pilote à bord" },
  { id: "JETTY",             label: "Navire à quai" },
  { id: "DEPARTURE_REQUEST", label: "Demande d'appareillage" },
];

const hhmm = (iso) => (iso ? iso.slice(11, 16) : "—");
const min  = (v) => (v == null ? "—" : `${Math.round(v)} min`);

export default function App() {
  const [form, setForm] = useState({
    imo: "9776171", berth: "TC21", event: "PORT_LIMIT",
    date: "2025-03-14", heure: "08:20", length: 334,
    after_draft: 12.5, forward_draft: 11.8, nb_tugs: "",
    eta: "2025-03-14T06:00:00", jetty_arrival: "",
  });
  const [timeline, setTimeline] = useState([]);
  const [narration, setNarration] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [erreur, setErreur]       = useState(null);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  async function predire() {
    setLoading(true); setErreur(null);
    const body = {
      imo: form.imo, berth: form.berth, event: form.event,
      timestamp: `${form.date}T${form.heure}:00`,
      length: Number(form.length),
      after_draft: form.after_draft ? Number(form.after_draft) : null,
      forward_draft: form.forward_draft ? Number(form.forward_draft) : null,
      nb_tugs: form.nb_tugs ? Number(form.nb_tugs) : null,
      eta: form.eta || null,
      jetty_arrival: form.jetty_arrival || null,
    };
    try {
      const r = await fetch(`${API}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      const pred = await r.json();
      setTimeline((t) => [...t, pred]);

      const n = await fetch(`${API}/narrate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pred, length: Number(form.length),
                               nb_tugs: form.nb_tugs ? Number(form.nb_tugs) : null }),
      });
      setNarration(n.ok ? await n.json() : null);
    } catch (e) {
      setErreur(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  const last    = timeline[timeline.length - 1];
  const entree  = last && last.event !== "DEPARTURE_REQUEST";
  const maxHi   = Math.max(...timeline.map((p) => p.hi), 1);
  const risque  = { "faible": "ok", "modéré": "warn", "élevé": "bad" };

  return (
    <div className="app">
      <header>
        <h1>MANTIS</h1>
        <p>Prédiction des durées de manœuvre · Tanger Med</p>
      </header>

      <section className="card form">
        <div className="grid">
          <label>IMO<input value={form.imo} onChange={set("imo")} /></label>
          <label>Quai
            <select value={form.berth} onChange={set("berth")}>
              {QUAIS.map((q) => <option key={q}>{q}</option>)}
            </select>
          </label>
          <label>Événement
            <select value={form.event} onChange={set("event")}>
              {EVENTS.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
            </select>
          </label>
          <label>Date<input type="date" value={form.date} onChange={set("date")} /></label>
          <label>Heure<input type="time" value={form.heure} onChange={set("heure")} /></label>
          <label>Longueur (m)<input type="number" value={form.length} onChange={set("length")} /></label>
          <label>Tirant arrière<input type="number" step="0.1" value={form.after_draft} onChange={set("after_draft")} /></label>
          <label>Tirant avant<input type="number" step="0.1" value={form.forward_draft} onChange={set("forward_draft")} /></label>
          <label>Remorqueurs<input type="number" placeholder="règle" value={form.nb_tugs} onChange={set("nb_tugs")} /></label>
          <label>ETA annoncée<input value={form.eta} onChange={set("eta")} placeholder="2025-03-14T06:00:00" /></label>
          <label>Arrivée à quai<input value={form.jetty_arrival} onChange={set("jetty_arrival")} placeholder="sortie uniquement" /></label>
        </div>
        <div className="actions">
          <button onClick={predire} disabled={loading}>
            {loading ? "Calcul…" : "Prédire"}
          </button>
          <button className="ghost" onClick={() => { setTimeline([]); setNarration(null); }}>
            Nouvelle escale
          </button>
        </div>
        {erreur && <p className="err">Erreur : {erreur}</p>}
      </section>

      {last && (
        <>
          <section className="card">
            <div className="head">
              <span className="tag">{EVENTS.find((e) => e.id === last.event)?.label}</span>
              <span className="muted">{last.berth} · {hhmm(last.timestamp)}</span>
              {last.risque_attente && (
                <span className={`risk ${risque[last.risque_attente] || ""}`}>
                  risque {last.risque_attente}
                </span>
              )}
            </div>

            {narration && <p className="narr">{narration.text}</p>}

            <div className="metrics">
              {entree ? (
                <>
                  <Metric titre="Approche (A)" valeur={min(last.phase_A)}
                          sous={last.mode === "intervalle"
                            ? `${Math.round(last.phase_A_lo)} – ${Math.round(last.phase_A_hi)} min` : "± 8 min"} />
                  <Metric titre="Accostage (B)" valeur={min(last.phase_B)} />
                  <Metric titre="Accostage estimé" valeur={hhmm(last.accostage_estime)} />
                  <Metric titre="Fin de manœuvre" valeur={hhmm(last.fin_estimee)}
                          sous={`${Math.round(last.lo)} – ${Math.round(last.hi)} min`} />
                </>
              ) : (
                <>
                  <Metric titre="Appareillage (C)" valeur={min(last.phase_C)} />
                  <Metric titre="Transit (D)" valeur={min(last.phase_D)} />
                  <Metric titre="Sortie limite portuaire" valeur={hhmm(last.fin_estimee)} />
                  <Metric titre="Total" valeur={min(last.total)}
                          sous={`${Math.round(last.lo)} – ${Math.round(last.hi)} min`} />
                </>
              )}
            </div>

            <div className="ctx">
              {Object.entries(last.contexte).map(([k, v]) => (
                <span key={k}><b>{k}</b> {String(v)}</span>
              ))}
              <span className="muted">modèle {last.model_version?.slice(0, 8)}</span>
            </div>
          </section>

          <section className="card">
            <h2>Chronologie de l'escale</h2>
            <p className="muted small">L'intervalle se resserre à mesure que l'incertitude se lève.</p>
            {timeline.map((p, i) => (
              <div className="row" key={i}>
                <span className="ev">{EVENTS.find((e) => e.id === p.event)?.label}</span>
                <span className="h">{hhmm(p.timestamp)}</span>
                <div className="bar">
                  <div className="range"
                       style={{ left: `${(p.lo / maxHi) * 100}%`,
                                width: `${((p.hi - p.lo) / maxHi) * 100}%` }} />
                  <div className="dot" style={{ left: `${(p.total / maxHi) * 100}%` }} />
                </div>
                <span className="val">{Math.round(p.lo)}–{Math.round(p.hi)} min</span>
              </div>
            ))}
          </section>

          {narration?.payload && (
            <details className="card">
              <summary>Document transmis à la couche de reformulation (minimisé)</summary>
              <pre>{JSON.stringify(narration.payload, null, 2)}</pre>
              <p className="muted small">Source : {narration.source}</p>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function Metric({ titre, valeur, sous }) {
  return (
    <div className="metric">
      <span className="t">{titre}</span>
      <span className="v">{valeur}</span>
      {sous && <span className="s">{sous}</span>}
    </div>
  );
}
