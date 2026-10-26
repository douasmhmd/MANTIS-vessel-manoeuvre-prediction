"""Schémas Pandera — contrat de données entre F, T et I.

Un schéma qui échoue arrête le pipeline : c'est volontaire.
"""
import pandera.pandas as pa
from pandera.pandas import Column, Check

_COMMON = {
    "MovementID": Column(int, unique=True, coerce=True),
    "IMONumber": Column(str, nullable=True, coerce=True),
    "term": Column(str, Check.isin(["TC1", "TC2", "TC3", "TC4"])),
    "t_ref": Column(pa.DateTime),
    "Length": Column(float, Check.in_range(50, 500), nullable=True),
    "AfterDraft": Column(float, Check.in_range(3, 25), nullable=True),
    "nb_tugs": Column(float, Check.in_range(0, 4), nullable=True),
    "nb_pilots": Column(float, Check.in_range(0, 2), nullable=True),
    "occup_term": Column(int, Check.ge(0)),
    "attente_term": Column(int, Check.ge(0)),
    "en_cours": Column(int, Check.ge(0)),
    "releve": Column(int, Check.isin([0, 1])),
    "term_c": Column(int, Check.in_range(0, 3)),
    "duree": Column(float, Check.gt(0)),
}

SCHEMA_ENTREE = pa.DataFrameSchema(
    {**_COMMON,
     "phase_A": Column(float, Check.in_range(0, 480)),
     "phase_B": Column(float, Check.in_range(0, 480)),
     "retard_ETA_w": Column(float, Check.in_range(-1440, 2880)),
     "delai_pilote": Column(float, nullable=True),
     },
    strict=False, coerce=True, name="features_entree",
)

SCHEMA_SORTIE = pa.DataFrameSchema(
    {**_COMMON,
     "phase_C": Column(float, Check.in_range(0, 120)),
     "phase_D": Column(float, Check.in_range(0, 120)),
     "sejour_quai_h": Column(float, Check.gt(0), nullable=True),
     },
    strict=False, coerce=True, name="features_sortie",
)


def validate_entree(df):
    return SCHEMA_ENTREE.validate(df, lazy=True)


def validate_sortie(df):
    return SCHEMA_SORTIE.validate(df, lazy=True)