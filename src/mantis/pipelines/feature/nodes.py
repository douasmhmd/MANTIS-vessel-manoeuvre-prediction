"""Pipeline F (Feature) — du CSV brut TMPA aux tables de features par manœuvre.

Six nodes, dans l'ordre :
    clean_raw → build_manoeuvres → split_filter → add_context → add_history → validate_select

Toutes les règles (seuils, bornes, colonnes) viennent de conf/base/parameters.yml
et ont été justifiées dans l'analyse exploratoire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.schemas import validate_entree, validate_sortie

# --------------------------------------------------------------------------- #
# 1. Nettoyage du CSV brut
# --------------------------------------------------------------------------- #
def clean_raw(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Nettoie guillemets/suffixes, filtre porte-conteneurs + types 1/5, parse les dates."""
    df = raw.copy()
    df.columns = df.columns.str.strip().str.replace('"', "", regex=False).str.replace(";;;", "", regex=False)

    for c in df.select_dtypes(include="object").columns:
        s = df[c].astype(str).str.strip().str.strip('"').str.replace(";;;", "", regex=False)
        df[c] = s.mask(s.isin(["nan", "NULL", "None", ""]))

    df["MovementTypeID"] = pd.to_numeric(df["MovementTypeID"], errors="coerce")
    for c in ["Length", "Beam", "AfterDraft", "ForwardDraft"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask_pc = df["typeNavire"].str.contains(params["type_navire_pattern"], case=False, na=False)
    mask_mv = df["MovementTypeID"].isin(list(params["movement_types"].values()))
    df = df[mask_pc & mask_mv].copy()

    for c in params["date_cols"]:
        df[c] = pd.to_datetime(df[c], format="ISO8601", errors="coerce")

    df["ResourceType"] = df["ResourceType"].str.strip()
    return df


# --------------------------------------------------------------------------- #
# 2. Une ligne par manœuvre + 4 phases
# --------------------------------------------------------------------------- #
def _minutes(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds() / 60


def build_manoeuvres(mvt: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Agrège les lignes ressource (2,5 par manœuvre) en une ligne par MovementID."""
    res = mvt.pivot_table(index="MovementID", columns="ResourceType", values="ResourceUtilizationID",
                          aggfunc="nunique", fill_value=0)
    res.columns = [f"nb_{str(c).lower()}" for c in res.columns]
    for c in ["nb_tugs", "nb_pilots"]:
        if c not in res.columns:
            res[c] = 0

    first = mvt.sort_values("Resource_ActualStartDate").groupby("MovementID").first()
    m = first.join(res[["nb_tugs", "nb_pilots"]])

    # Délai de mobilisation du pilote (POST-HOC : jamais une feature, utile au monitoring/EDA)
    pil = mvt[mvt["ResourceType"] == "Pilots"]
    lag = (pil["Resource_ActualStartDate"] - pil["PortLimitArrival"]).dt.total_seconds() / 60
    m["delai_pilote"] = lag.groupby(pil["MovementID"]).min()

    te, ts = params["movement_types"]["entree"], params["movement_types"]["sortie"]
    is_e, is_s = m["MovementTypeID"] == te, m["MovementTypeID"] == ts

    m["duree"] = _minutes(m["Mouvement_ActualFinishDate"], m["Mouvement_ActualStartDate"])
    m.loc[is_e, "phase_A"] = _minutes(m["JettyArrival"], m["PortLimitArrival"])[is_e]
    m.loc[is_e, "phase_B"] = _minutes(m["Mouvement_ActualFinishDate"], m["JettyArrival"])[is_e]
    m.loc[is_s, "phase_C"] = _minutes(m["JettyDeparture"], m["Mouvement_ActualStartDate"])[is_s]
    m.loc[is_s, "phase_D"] = _minutes(m["PortLimitDeparture"], m["JettyDeparture"])[is_s]
    m.loc[is_s, "sejour_quai_h"] = (_minutes(m["JettyDeparture"], m["JettyArrival"]) / 60)[is_s]

    m["term"] = m["BerthName"].astype(str).str[:3]
    return m.reset_index()


# --------------------------------------------------------------------------- #
# 3. Séparation entrée / sortie + nettoyage physique
# --------------------------------------------------------------------------- #
def split_filter(m: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique les bornes physiques, met les drafts aberrants à NaN, plafonne les ressources."""
    lo_d, hi_d = params["draft_bounds"]
    te, ts = params["movement_types"]["entree"], params["movement_types"]["sortie"]

    def _common(d: pd.DataFrame) -> pd.DataFrame:
        d = d[d["term"].isin(params["terminaux"])].copy()
        for c in ["AfterDraft", "ForwardDraft"]:
            d.loc[~d[c].between(lo_d, hi_d), c] = np.nan
        d["nb_tugs"] = d["nb_tugs"].clip(upper=params["clip_tugs"])
        d["nb_pilots"] = d["nb_pilots"].clip(upper=params["clip_pilots"])
        return d

    lo, hi = params["seuils_min"]["entree"]
    ent = m[(m["MovementTypeID"] == te) & m["duree"].between(lo, hi) & (m["phase_A"] > 0) & (m["phase_B"] > 0)]
    lo, hi = params["seuils_min"]["sortie"]
    sor = m[(m["MovementTypeID"] == ts) & m["duree"].between(lo, hi) & (m["phase_C"] > 0) & (m["phase_D"] > 0)]

    ent, sor = _common(ent), _common(sor)
    ent["t_ref"] = ent["PortLimitArrival"]
    sor["t_ref"] = sor["Mouvement_ActualStartDate"]
    return ent, sor


# --------------------------------------------------------------------------- #
# 4. Contexte opérationnel à l'instant t_ref (sans fuite)
# --------------------------------------------------------------------------- #
def _count_active(t: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """#(starts < t) - #(ends <= t) : nombre d'intervalles [start, end) contenant t."""
    starts, ends = np.sort(starts[~np.isnat(starts)]), np.sort(ends[~np.isnat(ends)])
    return np.searchsorted(starts, t, "left") - np.searchsorted(ends, t, "right")


def _count_active_by_group(d: pd.DataFrame, key: str, ref: pd.DataFrame, start_col: str, end_col: str) -> np.ndarray:
    out = np.zeros(len(d), dtype=int)
    t_all = d["t_ref"].values
    for k, idx in d.groupby(key).indices.items():
        sub = ref[ref[key] == k]
        out[idx] = _count_active(t_all[idx], sub[start_col].values, sub[end_col].values)
    return out


def add_context(ent: pd.DataFrame, sor: pd.DataFrame, m: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Congestion globale/locale, calendrier, retard ETA, règle des remorqueurs."""
    te = params["movement_types"]["entree"]
    all_starts = m["Mouvement_ActualStartDate"].values
    all_ends = m["Mouvement_ActualFinishDate"].values
    occ = m[m["MovementTypeID"] == te][["term", "BerthName", "PortLimitArrival", "JettyArrival", "JettyDeparture"]].dropna()

    lo_r, hi_r = params["retard_eta_clip"]
    bins = params["tugs_regle_bins"]

    def _ctx(d: pd.DataFrame, with_eta: bool) -> pd.DataFrame:
        d = d.copy()
        t = d["t_ref"].values
        d["heure"] = d["t_ref"].dt.hour
        d["jour_sem"] = d["t_ref"].dt.dayofweek
        d["releve"] = d["heure"].isin(params["releve_heures"]).astype(int)
        d["heure_sin"] = np.sin(2 * np.pi * d["heure"] / 24)
        d["heure_cos"] = np.cos(2 * np.pi * d["heure"] / 24)

        d["en_cours"] = _count_active(t, all_starts, all_ends)
        d["occup_term"] = _count_active_by_group(d, "term", occ, "JettyArrival", "JettyDeparture")
        d["attente_term"] = _count_active_by_group(d, "term", occ, "PortLimitArrival", "JettyArrival")

        d["retard_ETA_w"] = (_minutes(d["PortLimitArrival"], d["Mouvement_ETA"]).clip(lo_r, hi_r) if with_eta else 0.0)

        d["assiette"] = d["AfterDraft"] - d["ForwardDraft"]
        d["nb_tugs_regle"] = pd.cut(d["Length"], bins, labels=range(1, len(bins))).astype(float)
        d["ecart_tugs"] = d["nb_tugs"] - d["nb_tugs_regle"]
        return d

    return _ctx(ent, True), _ctx(sor, False)


# --------------------------------------------------------------------------- #
# 5. Encodages historiques anti-fuite (shift → expanding)
# --------------------------------------------------------------------------- #
def _hist(df: pd.DataFrame, key: str, target: str, min_n: int) -> tuple[pd.Series, pd.Series]:
    g = df.groupby(key)[target]
    mean_prev = g.transform(lambda s: s.shift(1).expanding().mean())
    n_prev = g.cumcount()
    return mean_prev.where(n_prev >= min_n), n_prev


def add_history(ent: pd.DataFrame, sor: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moyennes passées par quai et par navire (n'utilise que les manœuvres antérieures à t_ref)."""
    mn = params["hist_min_n"]

    def _enrich(d: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        d = d.sort_values("t_ref").copy()
        for c in ["AfterDraft", "assiette"]:                       # imputation par navire
            d[c] = d[c].fillna(d.groupby("IMONumber")[c].transform("median"))
        for tg in targets:
            d[f"quai_hist_{tg}"], _ = _hist(d, "BerthName", tg, mn["quai"])
            d[f"navire_hist_{tg}"], d["nb_escales_prev"] = _hist(d, "IMONumber", tg, mn["navire"])
        d["term_c"] = pd.Categorical(d["term"], categories=params["terminaux"]).codes
        return d

    return _enrich(ent, ["phase_A", "phase_B"]), _enrich(sor, ["phase_C", "phase_D"])


# --------------------------------------------------------------------------- #
# 6. Sélection des colonnes + validation Pandera
# --------------------------------------------------------------------------- #
META = ["MovementID", "VisitID", "VesselName", "IMONumber", "BerthName", "term", "t_ref"]
FEAT_PHYS = ["Length", "AfterDraft", "assiette", "nb_tugs", "nb_tugs_regle", "ecart_tugs", "nb_pilots"]
FEAT_CTX = ["occup_term", "attente_term", "en_cours", "retard_ETA_w", "releve", "heure_sin", "heure_cos", "term_c"]
POST_HOC = ["delai_pilote"]   # conservé pour le monitoring, JAMAIS passé au modèle


def validate_select(ent: pd.DataFrame, sor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_e = ["quai_hist_phase_A", "quai_hist_phase_B", "navire_hist_phase_A", "navire_hist_phase_B", "nb_escales_prev"]
    hist_s = ["quai_hist_phase_C", "quai_hist_phase_D", "navire_hist_phase_C", "navire_hist_phase_D", "nb_escales_prev"]
    E = ent[META + FEAT_PHYS + FEAT_CTX + hist_e + POST_HOC + ["phase_A", "phase_B", "duree"]].reset_index(drop=True)
    S = sor[META + FEAT_PHYS + FEAT_CTX + hist_s + ["sejour_quai_h", "phase_C", "phase_D", "duree"]].reset_index(drop=True)
    return validate_entree(E), validate_sortie(S)