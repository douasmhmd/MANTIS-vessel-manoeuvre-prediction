"""Pipeline F — Feature : CSV brut → features_entree / features_sortie."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_context, add_history, build_manoeuvres, clean_raw, split_filter, validate_select


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(clean_raw, ["raw_vessels", "params:feature"], "mvt_clean", name="clean_raw"),
            node(build_manoeuvres, ["mvt_clean", "params:feature"], "manoeuvres", name="build_manoeuvres"),
            node(split_filter, ["manoeuvres", "params:feature"], ["ent_filtered", "sor_filtered"], name="split_filter"),
            node(add_context, ["ent_filtered", "sor_filtered", "manoeuvres", "params:feature"],
                 ["ent_ctx", "sor_ctx"], name="add_context"),
            node(add_history, ["ent_ctx", "sor_ctx", "params:feature"], ["ent_hist", "sor_hist"], name="add_history"),
            node(validate_select, ["ent_hist", "sor_hist"], ["features_entree", "features_sortie"], name="validate_select"),
        ],
        namespace=None,
    )