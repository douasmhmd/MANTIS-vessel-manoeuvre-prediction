"""Pipeline Monitoring — prédictions + features → métriques glissantes, rapport Evidently, décision."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_drift_report, decide_retrain, rolling_performance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(rolling_performance, ["predictions_entree", "predictions_sortie", "params:monitoring"],
                 "monitoring_rolling", name="rolling_performance"),
            node(data_drift_report, ["features_entree", "features_sortie", "params:monitoring"],
                 "drift_summary", name="data_drift_report"),
            node(decide_retrain, ["monitoring_rolling", "drift_summary", "metrics_holdout", "params:monitoring"],
                 "monitoring_decision", name="decide_retrain"),
        ]
    )