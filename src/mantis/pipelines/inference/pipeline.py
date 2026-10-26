"""Pipeline I — Inference batch : modèles + features holdout → prédictions jointes au réalisé."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import score_entree, score_sortie


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(score_entree, ["features_entree", "model_B", "model_A_cqr", "params:training"],
                 "predictions_entree", name="score_entree"),
            node(score_sortie, ["features_sortie", "model_C", "model_D", "params:training"],
                 "predictions_sortie", name="score_sortie"),
        ]
    )