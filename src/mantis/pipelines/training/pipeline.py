"""Pipeline T — Training : features → 4 modèles + métriques + registre MLflow."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import register_mlflow, rolling_eval, train_A_cqr, train_B, train_sortie


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(rolling_eval, ["features_entree", "features_sortie", "params:training"],
                 "metrics_rolling", name="rolling_eval"),
            node(train_B, ["features_entree", "params:training"], ["model_B", "metrics_B"], name="train_B"),
            node(train_A_cqr, ["features_entree", "params:training"], ["model_A_cqr", "metrics_A"], name="train_A_cqr"),
            node(train_sortie, ["features_sortie", "params:training"], ["model_C", "model_D", "metrics_CD"],
                 name="train_sortie"),
            node(register_mlflow,
                 ["model_B", "model_A_cqr", "model_C", "model_D", "metrics_B", "metrics_A", "metrics_CD",
                  "metrics_rolling", "params:training"],
                 "metrics_holdout", name="register_mlflow"),
        ]
    )