"""Pipeline I (Inference, mode batch) — score le holdout avec les modèles enregistrés.

Simule la production sur la période de test : une prédiction par manœuvre,
jointe à la valeur réalisée → base du pipeline monitoring.
"""
from __future__ import annotations

import pandas as pd


def score_entree(E: pd.DataFrame, model_B, model_A_cqr, params_tr: dict) -> pd.DataFrame:
    te = E[E["t_ref"] >= pd.Timestamp(params_tr["holdout_start"])].copy()
    X = te[params_tr["features_entree"]]
    a = model_A_cqr.predict(X)
    out = te[["MovementID", "IMONumber", "BerthName", "term", "t_ref", "phase_A", "phase_B", "duree", "delai_pilote"]].copy()
    out["pred_A_lo"], out["pred_A"], out["pred_A_hi"] = a["lo"].values, a["md"].values, a["hi"].values
    out["pred_B"] = model_B.predict(X)
    out["pred_total"] = out["pred_A"] + out["pred_B"]
    out["couvert_A"] = te["phase_A"].between(out["pred_A_lo"], out["pred_A_hi"]).astype(int)
    out["resid_B"] = out["pred_B"] - te["phase_B"]
    out["resid_A"] = out["pred_A"] - te["phase_A"]
    return out.reset_index(drop=True)


def score_sortie(S: pd.DataFrame, model_C, model_D, params_tr: dict) -> pd.DataFrame:
    te = S[S["t_ref"] >= pd.Timestamp(params_tr["holdout_start"])].copy()
    X = te[params_tr["features_sortie"]]
    out = te[["MovementID", "IMONumber", "BerthName", "term", "t_ref", "phase_C", "phase_D", "duree"]].copy()
    out["pred_C"], out["pred_D"] = model_C.predict(X), model_D.predict(X)
    out["pred_total"] = out["pred_C"] + out["pred_D"]
    out["resid_C"], out["resid_D"] = out["pred_C"] - te["phase_C"], out["pred_D"] - te["phase_D"]
    return out.reset_index(drop=True)