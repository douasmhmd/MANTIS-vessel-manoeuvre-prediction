"""Modèles MANTIS.

- ``make_hgb``      : régression ponctuelle (phases B, C, D)
- ``CQRModel``      : quantiles conditionnels + calibration conformale (phase A)
- ``CQRPyfunc``     : enveloppe MLflow pyfunc pour enregistrer CQRModel dans le registre
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def make_hgb(hgb_params: dict, loss: str = "squared_error", quantile: float | None = None):
    kw = dict(hgb_params)
    if loss == "quantile":
        kw.update(loss="quantile", quantile=quantile)
    return HistGradientBoostingRegressor(**kw)


class CQRModel:
    """Conformalized Quantile Regression (Romano et al., 2019).

    fit(X, y, X_cal, y_cal) entraîne trois régressions quantiles puis calibre
    la correction q̂ sur une fenêtre de calibration disjointe. predict(X)
    renvoie un DataFrame [lo, md, hi] à couverture (1-alpha) garantie.
    """

    def __init__(self, hgb_params: dict, quantiles=(0.10, 0.50, 0.90), alpha: float = 0.2, floor: float = 10.0):
        self.hgb_params, self.quantiles, self.alpha, self.floor = hgb_params, tuple(quantiles), alpha, floor
        self.models_: dict[float, HistGradientBoostingRegressor] = {}
        self.qhat_: float = 0.0
        self.features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, X_cal: pd.DataFrame, y_cal: pd.Series) -> "CQRModel":
        self.features_ = list(X.columns)
        for q in self.quantiles:
            self.models_[q] = make_hgb(self.hgb_params, "quantile", q).fit(X, y)
        lo, hi = self.models_[self.quantiles[0]].predict(X_cal), self.models_[self.quantiles[-1]].predict(X_cal)
        scores = np.maximum(lo - y_cal.values, y_cal.values - hi)
        n = len(scores)
        self.qhat_ = float(np.quantile(scores, min(1.0, np.ceil((1 - self.alpha) * (n + 1)) / n)))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features_]
        q0, q1, q2 = self.quantiles
        lo = np.maximum(self.models_[q0].predict(X) - self.qhat_, self.floor)
        md = self.models_[q1].predict(X)
        hi = self.models_[q2].predict(X) + self.qhat_
        return pd.DataFrame({"lo": lo, "md": md, "hi": np.maximum(hi, lo + 1)}, index=X.index)

    def coverage(self, X: pd.DataFrame, y: pd.Series) -> dict:
        p = self.predict(X)
        ok = (y.values >= p["lo"].values) & (y.values <= p["hi"].values)
        return {"coverage": float(ok.mean()), "width": float((p["hi"] - p["lo"]).mean()),
                "mae_median": float(np.abs(p["md"].values - y.values).mean()), "qhat": self.qhat_}


try:
    import mlflow.pyfunc

    class CQRPyfunc(mlflow.pyfunc.PythonModel):
        def __init__(self, model: CQRModel):
            self.model = model

        def predict(self, context, model_input: pd.DataFrame, params=None):
            return self.model.predict(model_input)
except ImportError:  # mlflow absent (ex. conteneur API minimal)
    CQRPyfunc = None