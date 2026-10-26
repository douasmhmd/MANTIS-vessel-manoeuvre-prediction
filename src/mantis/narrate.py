"""Reformulation en langage naturel de la prédiction pour l'opérateur.

Principe : le LLM verbalise, il ne calcule rien. Il ne reçoit qu'un JSON minimisé
(liste blanche stricte : durées, catégories, indicateurs). Aucun identifiant de navire,
d'agent, de pilote, aucun horodatage, aucun quai précis ne traverse la frontière.
En cas d'échec (clé absente, réseau, réponse non conforme), un gabarit déterministe
prend le relais : l'opérateur n'attend jamais une prédiction à cause du texte.
"""
from __future__ import annotations

import os
import re

# --------------------------------------------------------------------------- #
# 1. Minimisation — liste blanche validée
# --------------------------------------------------------------------------- #
ALLOWED = {"manoeuvre", "etape", "mode", "phase_A_lo", "phase_A", "phase_A_hi", "phase_B", "phase_C", "phase_D",
           "total_lo", "total", "total_hi", "risque_attente", "charge_terminal", "file_attente", "releve",
           "classe_taille", "nb_remorqueurs", "facteurs_risque", "facteurs_favorables"}

_ETAPE = {"ETA": "annonce ETA", "PORT_LIMIT": "arrivée à la limite portuaire", "PILOT_ONBOARD": "pilote à bord",
          "JETTY": "navire à quai", "DEPARTURE_REQUEST": "demande d'appareillage"}


def _charge(occup: int | None) -> str:
    if occup is None:
        return "inconnue"
    return "faible" if occup <= 1 else "modérée" if occup <= 2 else "élevée"


def _classe(length: float | None) -> str:
    if length is None:
        return "inconnue"
    return "< 200 m" if length < 200 else "200–300 m" if length < 300 else "> 300 m (post-Panamax)"


def _facteurs(charge: str, file_attente: int, releve: bool) -> tuple[list[str], list[str]]:
    """Sépare ce qui aggrave l'attente de ce qui la limite, pour éviter un texte contradictoire."""
    risque, favorables = [], []
    (risque if releve else favorables).append(
        "créneau de relève des pilotes" if releve else "hors créneau de relève")
    if file_attente > 0:
        risque.append(f"{file_attente} navire(s) déjà en attente")
    else:
        favorables.append("aucun navire en attente")
    if charge in ("modérée", "élevée"):
        risque.append(f"charge du terminal {charge}")
    elif charge == "faible":
        favorables.append("charge du terminal faible")
    return risque, favorables


def minimize(pred: dict, length: float | None = None, nb_tugs: float | None = None) -> dict:
    """Construit le JSON transmissible. Tout champ hors ALLOWED est refusé.

    Les bornes agrégées sont arrondies après sommation, jamais avant, pour que
    total_lo == phase_A_lo + phase_B à l'affichage.
    """
    ctx = pred.get("contexte", {})
    entree = pred["event"] != "DEPARTURE_REQUEST"
    charge = _charge(ctx.get("occup_term"))
    file_attente = int(ctx.get("attente_term", 0) or 0)
    releve = bool(ctx.get("releve", 0))

    out = {
        "manoeuvre": "entrée" if entree else "sortie",
        "etape": _ETAPE.get(pred["event"], pred["event"]),
        "mode": pred["mode"],
        "total_lo": round(pred["lo"]), "total": round(pred["total"]), "total_hi": round(pred["hi"]),
        "charge_terminal": charge,
        "file_attente": file_attente,
        "releve": releve,
        "classe_taille": _classe(length),
        "nb_remorqueurs": None if nb_tugs is None else int(nb_tugs),
    }
    if entree:
        a_lo, a_md, a_hi = round(pred["phase_A_lo"]), round(pred["phase_A"]), round(pred["phase_A_hi"])
        b = round(pred["phase_B"])
        risque, favorables = _facteurs(charge, file_attente, releve)
        out.update(phase_A_lo=a_lo, phase_A=a_md, phase_A_hi=a_hi, phase_B=b,
                   risque_attente=pred.get("risque_attente"),
                   facteurs_risque=risque, facteurs_favorables=favorables,
                   # cohérence arithmétique : les bornes agrégées sont les sommes des bornes affichées
                   total_lo=a_lo + b, total=a_md + b, total_hi=a_hi + b)
    else:
        c, d = round(pred["phase_C"]), round(pred["phase_D"])
        out.update(phase_C=c, phase_D=d)
    illegal = set(out) - ALLOWED
    if illegal:
        raise ValueError(f"champs non autorisés vers le LLM : {illegal}")
    return out


# --------------------------------------------------------------------------- #
# 2. Gabarit déterministe (fallback et référence)
# --------------------------------------------------------------------------- #
def _joindre(items: list[str], liaison: str = "et") -> str:
    """Joint une liste en français : « a », « a et b », « a, b et c »."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {liaison} {items[-1]}"


def _phrase_risque(m: dict) -> str:
    """Formule le risque en distinguant ce qui l'explique de ce qui l'atténue."""
    risque = m.get("facteurs_risque", [])
    fav = m.get("facteurs_favorables", [])
    base = f"Risque d'attente {m['risque_attente']}"

    if len(risque) == 1:
        cause = f" en raison du {risque[0]}" if risque[0].startswith("créneau") else f" en raison de la {risque[0]}"
    elif risque:
        cause = f" en raison de {_joindre(risque)}"
    else:
        cause = ""

    if risque and fav:
        subs = [_negatif(f) for f in fav]
        nuance = f", bien que {_joindre(subs, liaison='et que')}"
        return f"{base}{cause}{nuance}."
    if risque:
        return f"{base}{cause}."
    if fav:
        return f"{base} ({_joindre(fav)})."
    return f"{base}."


def _negatif(f: str) -> str:
    """Reformule un facteur favorable en proposition subordonnée grammaticale."""
    table = {
        "aucun navire en attente": "aucun navire ne soit en attente",
        "hors créneau de relève": "l'arrivée soit hors créneau de relève",
        "charge du terminal faible": "la charge du terminal soit faible",
    }
    return table.get(f, f)


def template(m: dict) -> str:
    if m["manoeuvre"] == "entrée":
        if m["mode"] == "intervalle":
            txt = (f"Étape : {m['etape']}. Approche estimée entre {m['phase_A_lo']} et "
                   f"{m['phase_A_hi']} min. {_phrase_risque(m)} "
                   f"Accostage ensuite ≈ {m['phase_B']} min, soit une fin de manœuvre entre "
                   f"{m['total_lo']} et {m['total_hi']} min.")
        else:
            txt = (f"Étape : {m['etape']}. Approche ≈ {m['phase_A']} min, accostage ≈ {m['phase_B']} min. "
                   f"Fin de manœuvre dans {m['total']} min (± {m['total_hi'] - m['total']}).")
    else:
        txt = (f"Étape : {m['etape']}. Appareillage ≈ {m['phase_C']} min puis transit ≈ {m['phase_D']} min. "
               f"Sortie de la limite portuaire dans {m['total']} min (± 10).")
    return txt


# --------------------------------------------------------------------------- #
# 3. Reformulation LLM
# --------------------------------------------------------------------------- #
_SYSTEM = """
Tu es un assistant intelligent du poste de commandement d'un terminal
portuaire à conteneurs.

Ta mission est de reformuler les résultats d'un modèle de Machine Learning
en un message opérationnel court, clair et professionnel destiné à un
opérateur portuaire.

Le modèle ML fournit les prédictions sous forme de JSON.

RÈGLES ABSOLUES :
1. Utilise exclusivement les informations présentes dans le JSON.
2. N'invente aucune information, aucune cause et aucun chiffre.
3. Ne modifie jamais les valeurs numériques.
4. Ne recalcule aucune durée et ne fais aucun calcul.
5. Si une plage de prédiction est fournie, conserve exactement cette plage.
6. Le champ facteurs_risque contient ce qui explique le risque d'attente ;
   le champ facteurs_favorables contient ce qui l'atténue. Ne présente jamais
   un facteur favorable comme une cause du risque.
7. Si facteurs_risque est vide, n'invente aucune cause.
8. Distingue clairement les différentes phases de la manœuvre :
   A = approche, B = accostage, C = désaccostage, D = transit.
9. Le message doit être en français.
10. Maximum 2 phrases.
11. Ton professionnel, factuel et neutre.
12. Pas de salutation.
13. Pas de Markdown.
14. Le LLM ne doit jamais remplacer ou corriger la prédiction du modèle ML.

FORMAT ATTENDU :
Produis uniquement le message destiné à l'opérateur.
"""


def _numbers_ok(text: str, m: dict) -> bool:
    """Garde-fou : tout nombre du texte doit exister dans le JSON (tolérance ±1 pour les arrondis)."""
    allowed = {v for v in m.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    allowed |= {v + 1 for v in allowed} | {v - 1 for v in allowed} | {10}
    return all(int(n) in allowed for n in re.findall(r"\d+", text))


def narrate(pred: dict, length: float | None = None, nb_tugs: float | None = None,
            model: str = "gpt-4o-mini") -> dict:
    m = minimize(pred, length, nb_tugs)
    fallback = template(m)
    if not os.getenv("OPENAI_API_KEY"):
        return {"text": fallback, "source": "texte standard", "detail": "cle absente", "payload": m}
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=180, timeout=8,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": str(m)}])
        text = r.choices[0].message.content.strip()
        if not text or not _numbers_ok(text, m):
            return {"text": fallback, "source": "texte standard",
                    "detail": "reponse non conforme", "payload": m}
        return {"text": text, "source": f"llm:{model}", "payload": m}
    except Exception as e:  # noqa: BLE001
        return {"text": fallback, "source": "reformulation indisponible — texte standard",
                "detail": type(e).__name__, "payload": m}