import { useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const QUAIS = ["TC11","TC12","TC13","TC14","TC21","TC22","TC23","TC24",
               "TC31","TC32","TC41","TC42","TC43","TC44"];

const EVENTS = [
  { id: "ETA",               label: "Annonce ETA",              sens: "entree" },
  { id: "PORT_LIMIT",        label: "Arrivée limite portuaire", sens: "entree" },
  { id: "PILOT_ONBOARD",     label: "Pilote à bord",            sens: "entree" },
  { id: "JETTY",             label: "Navire à quai",            sens: "entree" },
  { id: "DEPARTURE_REQUEST", label: "Demande d'appareillage",   sens: "sortie" },
];

const CTX_LABEL = {
  occup_term:   "Navires au terminal",
  attente_term: "En attente",
  en_cours:     "Manœuvres en cours",
  releve:       "Relève pilotes",
};

const MAE_B = 9;   // erreur moyenne du modele d'accostage (chapitre 3)
const MAE_D = 3;

const pad   = (n) => String(n).padStart(2, "0");
const hhmm  = (iso) => (iso ? iso.slice(11, 16) : "—");
const rnd   = (v) => (v == null ? null : Math.round(v));
// decale un horodatage ISO de n minutes et renvoie HH:MM
const shift = (iso, mn) => {
  if (!iso) return "—";
  const d = new Date(iso.slice(0, 19));
  d.setMinutes(d.getMinutes() + mn);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
};
// nombre de remorqueurs deduit de la longueur (regle observee, chapitre 2)
const tugsRegle = (L) => (L < 200 ? 1 : L < 350 ? 2 : 3);

export default function App() {
  const [form, setForm] = useState({
    imo: "9776171", berth: "TC21", event: "PORT_LIMIT",
    date: "2025-03-14", heure: "08:20", length: 334,
    after_draft: 12.5, forward_draft: 11.8, nb_tugs: "",
    eta_date: "2025-03-14", eta_heure: "06:00",
    jetty_date: "", jetty_heure: "",
  });
  const [timeline, setTimeline]   = useState([]);
  const [narration, setNarration] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [erreur, setErreur]       = useState(null);

  const set  = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const sens = EVENTS.find((e) => e.id === form.event)?.sens;

  // controles de saisie (bornes du chapitre 2)
  const L = Number(form.length), AD = Number(form.after_draft), FD = Number(form.forward_draft);
  const errL  = form.length && (L < 50 || L > 500);
  const errAD = form.after_draft && (AD < 3 || AD > 25);
  const errFD = form.forward_draft && (FD < 3 || FD > 25);
  const saisieInvalide = errL || errAD || errFD;

  // retard derive, affiche a l'operateur
  const retard = (() => {
    if (sens !== "entree" || !form.eta_date || !form.eta_heure) return null;
    const t0 = new Date(`${form.eta_date}T${form.eta_heure}:00`);
    const t1 = new Date(`${form.date}T${form.heure}:00`);
    return Math.round((t1 - t0) / 60000);
  })();

  async function predire() {
    setLoading(true); setErreur(null);
    const body = {
      imo: form.imo, berth: form.berth, event: form.event,
      timestamp: `${form.date}T${form.heure}:00`,
      length: L,
      after_draft: form.after_draft ? AD : null,
      forward_draft: form.forward_draft ? FD : null,
      nb_tugs: form.nb_tugs ? Number(form.nb_tugs) : null,
      eta: sens === "entree" && form.eta_date ? `${form.eta_date}T${form.eta_heure}:00` : null,
      jetty_arrival: sens === "sortie" && form.jetty_date
        ? `${form.jetty_date}T${form.jetty_heure}:00` : null,
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
        body: JSON.stringify({ pred, length: L,
                               nb_tugs: form.nb_tugs ? Number(form.nb_tugs) : tugsRegle(L) }),
      });
      setNarration(n.ok ? await n.json() : null);
    } catch (e) {
      setErreur(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  const last   = timeline[timeline.length - 1];
  const entree = last && last.event !== "DEPARTURE_REQUEST";
  const maxHi  = Math.max(...timeline.map((p) => p.hi), 1);
  const risque = { "faible": "ok", "modéré": "warn", "élevé": "bad" };
  const interv = last?.mode === "intervalle";

  return (
    <div className="app">
      <header>
        <h1>MANTIS</h1>
        <p>Prédiction des durées de manœuvre · Tanger Med</p>
      </header>

      <section className="card">
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

          <label className={errL ? "bad" : ""}>Longueur (m)
            <input type="number" min="50" max="500" value={form.length} onChange={set("length")} />
            {errL && <em>50 – 500 m</em>}
          </label>

          <label className={errAD ? "bad" : ""}>Tirant arrière (m)
            <input type="number" step="0.1" min="3" max="25"
                   value={form.after_draft} onChange={set("after_draft")} />
            {errAD && <em>3 – 25 m</em>}
          </label>

          <label className={errFD ? "bad" : ""}>Tirant avant (m)
            <input type="number" step="0.1" min="3" max="25"
                   value={form.forward_draft} onChange={set("forward_draft")} />
            {errFD && <em>3 – 25 m</em>}
          </label>

          <label>Remorqueurs
            <input type="number" min="0" max="4" value={form.nb_tugs} onChange={set("nb_tugs")}
                   placeholder={`${tugsRegle(L)} (règle)`} />
            <em>règle : {tugsRegle(L)} pour {L || "—"} m</em>
          </label>

          {sens === "entree" && (
            <>
              <label>ETA — date
                <input type="date" value={form.eta_date} onChange={set("eta_date")} /></label>
              <label>ETA — heure
                <input type="time" value={form.eta_heure} onChange={set("eta_heure")} />
                {retard != null && (
                  <em>{retard >= 0 ? `retard ${retard} min` : `avance ${-retard} min`}</em>
                )}
              </label>
            </>
          )}

          {sens === "sortie" && (
            <>
              <label>Arrivée à quai — date
                <input type="date" value={form.jetty_date} onChange={set("jetty_date")} /></label>
              <label>Arrivée à quai — heure
                <input type="time" value={form.jetty_heure} onChange={set("jetty_heure")} /></label>
            </>
          )}
        </div>

        <div className="actions">
          <button onClick={predire} disabled={loading || saisieInvalide}>
            {loading ? "Calcul…" : "Prédire"}
          </button>
          <button className="ghost" onClick={() => { setTimeline([]); setNarration(null); }}>
            Nouvelle escale
          </button>
        </div>
        {saisieInvalide && <p className="err">Valeurs hors des plages admises.</p>}
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
                  risque d'attente {last.risque_attente}
                </span>
              )}
            </div>

            {narration && <p className="narr">{narration.text}</p>}

            <div className="metrics">
              {entree ? (
                <>
                  <Metric
                    titre="Approche (A)"
                    valeur={interv
                      ? `${rnd(last.phase_A_lo)} – ${rnd(last.phase_A_hi)} min`
                      : `${rnd(last.phase_A)} min`}
                    sous={interv
                      ? `médiane ${rnd(last.phase_A)} min · intervalle 80 %`
                      : `± 8 min`} />
                  <Metric
                    titre="Accostage (B)"
                    valeur={`${rnd(last.phase_B)} min`}
                    sous={`± ${MAE_B} min`} />
                  <Metric
                    titre="Accostage estimé"
                    valeur={interv
                      ? `${shift(last.timestamp, rnd(last.phase_A_lo))} – ${shift(last.timestamp, rnd(last.phase_A_hi))}`
                      : `${shift(last.timestamp, rnd(last.phase_A) - 8)} – ${shift(last.timestamp, rnd(last.phase_A) + 8)}`}
                    sous={`estimation ${hhmm(last.accostage_estime)}`} />
                  <Metric
                    titre="Fin de manœuvre"
                    valeur={`${shift(last.timestamp, rnd(last.lo))} – ${shift(last.timestamp, rnd(last.hi))}`}
                    sous={`${rnd(last.lo)} – ${rnd(last.hi)} min`} />
                </>
              ) : (
                <>
                  <Metric titre="Appareillage (C)" valeur={`${rnd(last.phase_C)} min`} sous={`± ${MAE_B} min`} />
                  <Metric titre="Transit (D)" valeur={`${rnd(last.phase_D)} min`} sous={`± ${MAE_D} min`} />
                  <Metric
                    titre="Sortie limite portuaire"
                    valeur={`${shift(last.timestamp, rnd(last.lo))} – ${shift(last.timestamp, rnd(last.hi))}`}
                    sous={`estimation ${hhmm(last.fin_estimee)}`} />
                  <Metric titre="Durée totale" valeur={`${rnd(last.total)} min`}
                          sous={`${rnd(last.lo)} – ${rnd(last.hi)} min`} />
                </>
              )}
            </div>

            <div className="ctx">
              {Object.entries(last.contexte).map(([k, v]) => (
                <span key={k}>
                  <b>{CTX_LABEL[k] || k}</b>{" "}
                  {typeof v === "boolean" || k === "releve"
                    ? (v ? "oui" : "non") : String(v)}
                </span>
              ))}
              <span className="muted">
                modèle {last.model_version?.slice(0, 8)} · entraîné jusqu'au 30/06/2024
              </span>
            </div>
          </section>

          <section className="card">
            <h2>Chronologie de l'escale</h2>
            <p className="muted small">
              L'intervalle se resserre à mesure que l'incertitude se lève.
            </p>
            {timeline.map((p, i) => (
              <div className="row" key={i}>
                <span className="ev">{EVENTS.find((e) => e.id === p.event)?.label}</span>
                <span className="h">{hhmm(p.timestamp)}</span>
                <div className="bar">
                  <div className="range"
                       style={{ left: `${(p.lo / maxHi) * 100}%`,
                                width: `${Math.max((p.hi - p.lo) / maxHi * 100, 0.8)}%` }} />
                  <div className="dot" style={{ left: `${(p.total / maxHi) * 100}%` }} />
                </div>
                <span className="val">{rnd(p.lo)} – {rnd(p.hi)} min</span>
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
