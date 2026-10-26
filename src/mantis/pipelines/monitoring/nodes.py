"""Pipeline Monitoring — dérive de performance et de données, décision de réentraînement.

Nodes :
    rolling_performance : biais / MAE / couverture par fenêtre glissante (prédictions vs réalisé)
    data_drift_report   : rapport Evidently (référence = période d'entraînement, courant = holdout)
    decide_retrain      : règles de seuil → {retrain: bool, raisons: [...]}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Performance sur fenêtre glissante
# --------------------------------------------------------------------------- #
def rolling_performance(pe: pd.DataFrame, ps: pd.DataFrame, params: dict) -> pd.DataFrame:
    w = params["window"]

    def _windows(df: pd.DataFrame, cols: dict, flux: str, extra=None) -> pd.DataFrame:
        df = df.sort_values("t_ref").reset_index(drop=True)
        rows = []
        for start in range(0, max(len(df) - w + 1, 1), w // 2):
            win = df.iloc[start:start + w]
            if len(win) < w // 2:
                break
            r = {"flux": flux, "fenetre_debut": win["t_ref"].iloc[0], "fenetre_fin": win["t_ref"].iloc[-1], "n": len(win)}
            for phase, resid in cols.items():
                r[f"biais_{phase}"] = float(win[resid].mean())
                r[f"MAE_{phase}"] = float(win[resid].abs().mean())
            if extra:
                r.update(extra(win))
            rows.append(r)
        return pd.DataFrame(rows)

    ent = _windows(pe, {"A": "resid_A", "B": "resid_B"}, "entree", lambda win: {"coverage_A": float(win["couvert_A"].mean())})
    sor = _windows(ps, {"C": "resid_C", "D": "resid_D"}, "sortie")
    return pd.concat([ent, sor], ignore_index=True).sort_values(["flux", "fenetre_debut"]).reset_index(drop=True)

# --------------------------------------------------------------------------- #
# 2. Dérive des données (Evidently)
# --------------------------------------------------------------------------- #
def data_drift_report(E: pd.DataFrame, S: pd.DataFrame, params: dict) -> dict:
    ref_end = pd.Timestamp(params["reference_end"])
    feats = [c for c in params["drift_features"] if c in E.columns]
    ref = pd.concat([E[E["t_ref"] < ref_end][feats + ["phase_A", "phase_B"]],
                     S[S["t_ref"] < ref_end][["phase_C", "phase_D"]].reset_index(drop=True)], axis=1)
    cur = pd.concat([E[E["t_ref"] >= ref_end][feats + ["phase_A", "phase_B"]],
                     S[S["t_ref"] >= ref_end][["phase_C", "phase_D"]].reset_index(drop=True)], axis=1)
    ref, cur = ref.dropna(how="all"), cur.dropna(how="all")

    path = Path(params["report_path"]); path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"reference_n": int(len(ref)), "current_n": int(len(cur)), "report": str(path)}
    try:                                                   # Evidently ≥ 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snap = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
        snap.save_html(str(path))
        summary["engine"] = "evidently>=0.7"
    except ImportError:                                    # API historique
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
        rep = Report(metrics=[DataDriftPreset()]); rep.run(current_data=cur, reference_data=ref); rep.save_html(str(path))
        summary["engine"] = "evidently.legacy"

    # Dérive des cibles, indépendante d'Evidently (pour la décision et le rapport LaTeX)
    for tg in ["phase_A", "phase_B", "phase_C", "phase_D"]:
        a, b = ref[tg].dropna(), cur[tg].dropna()
        summary[f"{tg}_median_ref"], summary[f"{tg}_median_cur"] = float(a.median()), float(b.median())
        summary[f"{tg}_shift_min"] = float(b.median() - a.median())
    return summary


# --------------------------------------------------------------------------- #
# 3. Décision de réentraînement
# --------------------------------------------------------------------------- #
def decide_retrain(rolling: pd.DataFrame, drift: dict, holdout: dict, params: dict) -> dict:
    s = params["seuils"]
    last_e = rolling[rolling["flux"] == "entree"].iloc[-1] if (rolling["flux"] == "entree").any() else None
    last_s = rolling[rolling["flux"] == "sortie"].iloc[-1] if (rolling["flux"] == "sortie").any() else None
    last = pd.concat([last_e.dropna(), last_s.dropna()]) if last_e is not None and last_s is not None else (last_e if last_s is None else last_s)
    last = last[~last.index.duplicated()]
    raisons = []
    if last is not None:
        for phase in ["B", "C", "D"]:
            b = last.get(f"biais_{phase}")
            if b is not None and not np.isnan(b) and abs(b) > s["biais_abs_max"]:
                raisons.append(f"biais {phase} = {b:+.1f} min sur la dernière fenêtre (seuil ±{s['biais_abs_max']})")
        cov = last.get("coverage_A")
        if cov is not None and not np.isnan(cov) and cov < s["coverage_min"]:
            raisons.append(f"couverture A = {cov:.2f} < {s['coverage_min']}")
        ref_mae = {"B": holdout["B"]["MAE"], "C": holdout["C"]["MAE"], "D": holdout["D"]["MAE"]}
        for phase, m in ref_mae.items():
            v = last.get(f"MAE_{phase}")
            if v is not None and not np.isnan(v) and v > s["mae_ratio_max"] * m:
                raisons.append(f"MAE {phase} = {v:.1f} > {s['mae_ratio_max']}× MAE holdout ({m:.1f})")
    for tg in ["phase_C", "phase_B"]:
        if abs(drift.get(f"{tg}_shift_min", 0)) > 2:
            raisons.append(f"médiane {tg} décalée de {drift[f'{tg}_shift_min']:+.1f} min vs référence")
    return {"retrain": bool(raisons), "raisons": raisons,
            "derniere_fenetre": None if last is None else {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                                                             for k, v in last.items()},
            "model_version": holdout.get("run_id")}