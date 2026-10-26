"""Pipeline T (Training) — validation rolling-origin, 4 modèles, traçabilité MLflow.

Nodes :
    rolling_eval  → tableau des scores par origine (pour le rapport)
    train_B       → modèle ponctuel accostage
    train_A_cqr   → intervalles conformaux approche
    train_sortie  → modèles C et D
    register_mlflow → runs MLflow + registre + metrics_holdout.json
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from mantis.models import CQRModel, CQRPyfunc, make_hgb


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _split(df: pd.DataFrame, cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(cut)
    return df[df["t_ref"] < cut], df[df["t_ref"] >= cut]


def _scores(y_true: pd.Series, y_pred: np.ndarray, y_base: float) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAE_base": float(mean_absolute_error(y_true, np.full(len(y_true), y_base))),
        "biais": float(np.mean(y_pred - y_true.values)),
        "n_test": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# 1. Validation rolling-origin (protocole du rapport)
# --------------------------------------------------------------------------- #
def rolling_eval(E: pd.DataFrame, S: pd.DataFrame, params: dict) -> pd.DataFrame:
    rows = []
    targets = [("entree", E, params["features_entree"], ["phase_A", "phase_B", "duree"]),
               ("sortie", S, params["features_sortie"], ["phase_C", "phase_D", "duree"])]
    for lib, df, feats, tgts in targets:
        for tg in tgts:
            for o in params["origins"]:
                o = pd.Timestamp(o)
                tr = df[df["t_ref"] < o]
                te = df[(df["t_ref"] >= o) & (df["t_ref"] < o + pd.DateOffset(months=params["horizon_months"]))]
                if len(te) < 50:
                    continue
                m = make_hgb(params["hgb"]).fit(tr[feats], tr[tg])
                rows.append({"manoeuvre": lib, "cible": tg, "origine": o.date().isoformat(),
                             **_scores(te[tg], m.predict(te[feats]), tr[tg].median())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Modèles finaux (train < holdout_start, test = holdout)
# --------------------------------------------------------------------------- #
def train_B(E: pd.DataFrame, params: dict):
    feats = params["features_entree"]
    tr, te = _split(E, params["holdout_start"])
    model = make_hgb(params["hgb"]).fit(tr[feats], tr["phase_B"])
    metrics = _scores(te["phase_B"], model.predict(te[feats]), tr["phase_B"].median())
    return model, metrics


def train_A_cqr(E: pd.DataFrame, params: dict):
    feats = params["features_entree"]
    tr, rest = _split(E, params["calib_start"])
    cal, te = _split(rest, params["holdout_start"])
    model = CQRModel(params["hgb"], params["quantiles"], params["alpha"]).fit(
        tr[feats], tr["phase_A"], cal[feats], cal["phase_A"])
    metrics = model.coverage(te[feats], te["phase_A"])
    # Comparaison à l'intervalle naïf (quantiles globaux du train)
    lo_n, hi_n = tr["phase_A"].quantile(params["quantiles"][0]), tr["phase_A"].quantile(params["quantiles"][-1])
    metrics["coverage_naive"] = float(te["phase_A"].between(lo_n, hi_n).mean())
    metrics["width_naive"] = float(hi_n - lo_n)
    metrics["mae_base"] = float(mean_absolute_error(te["phase_A"], np.full(len(te), tr["phase_A"].median())))
    metrics["n_test"] = int(len(te))
    return model, metrics


def train_sortie(S: pd.DataFrame, params: dict):
    feats = params["features_sortie"]
    tr, te = _split(S, params["holdout_start"])
    out = {}
    models = {}
    for tg in ["phase_C", "phase_D"]:
        m = make_hgb(params["hgb"]).fit(tr[feats], tr[tg])
        models[tg] = m
        out[tg] = _scores(te[tg], m.predict(te[feats]), tr[tg].median())
    return models["phase_C"], models["phase_D"], out


# --------------------------------------------------------------------------- #
# 3. Traçabilité MLflow (runs, métriques, registre)
# --------------------------------------------------------------------------- #
def register_mlflow(model_B, model_A, model_C, model_D, metrics_B: dict, metrics_A: dict, metrics_CD: dict,
                    metrics_rolling: pd.DataFrame, params: dict) -> dict:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    def _log_common(run_name: str, target: str, metrics: dict, feats: list[str]):
        mlflow.log_params({"target": target, "holdout_start": params["holdout_start"],
                           "n_features": len(feats), **{f"hgb.{k}": v for k, v in params["hgb"].items()}})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        sub = metrics_rolling[metrics_rolling["cible"] == target]
        if len(sub):
            mlflow.log_metrics({"rolling_R2_mean": sub["R2"].mean(), "rolling_R2_std": sub["R2"].std(),
                                "rolling_MAE_mean": sub["MAE"].mean()})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics_rolling.csv"; metrics_rolling.to_csv(p, index=False); mlflow.log_artifact(str(p))

    fe, fs = params["features_entree"], params["features_sortie"]
    with mlflow.start_run(run_name="mantis-training") as parent:
        mlflow.set_tag("pattern", "FTI")
        for name, model, target, metrics, feats in [
            ("mantis_B", model_B, "phase_B", metrics_B, fe),
            ("mantis_C", model_C, "phase_C", metrics_CD["phase_C"], fs),
            ("mantis_D", model_D, "phase_D", metrics_CD["phase_D"], fs),
        ]:
            with mlflow.start_run(run_name=name, nested=True):
                _log_common(name, target, metrics, feats)
                mlflow.sklearn.log_model(model, name="model", registered_model_name=name)
        with mlflow.start_run(run_name="mantis_A_cqr", nested=True):
            _log_common("mantis_A_cqr", "phase_A", metrics_A, fe)
            mlflow.log_param("alpha", params["alpha"])
            mlflow.pyfunc.log_model(name="model", python_model=CQRPyfunc(model_A), registered_model_name="mantis_A_cqr")
        parent_id = parent.info.run_id

    summary = {"run_id": parent_id, "holdout_start": params["holdout_start"],
               "B": metrics_B, "A_cqr": metrics_A, "C": metrics_CD["phase_C"], "D": metrics_CD["phase_D"]}
    return summary