"""
CVSS v3.1 base score from the eight vector components.

The base score and severity are deterministic functions of the components
(FIRST CVSS v3.1 specification, section 7). Predicting the components and
computing the score guarantees that score, severity and vector agree.
"""
import math

AV = {"NETWORK": 0.85, "ADJACENT_NETWORK": 0.62, "ADJACENT": 0.62, "LOCAL": 0.55, "PHYSICAL": 0.2}
AC = {"LOW": 0.77, "HIGH": 0.44}
PR_UNCHANGED = {"NONE": 0.85, "LOW": 0.62, "HIGH": 0.27}
PR_CHANGED = {"NONE": 0.85, "LOW": 0.68, "HIGH": 0.5}
UI = {"NONE": 0.85, "REQUIRED": 0.62}
CIA = {"HIGH": 0.56, "LOW": 0.22, "NONE": 0.0}

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def roundup(x: float) -> float:
    """Spec roundup: smallest number, to one decimal place, >= x (float-safe)."""
    i = round(x * 100_000)
    if i % 10_000 == 0:
        return i / 100_000.0
    return (math.floor(i / 10_000) + 1) / 10.0


def base_score(c: dict) -> float | None:
    """Return the CVSS v3.1 base score, or None if any component is missing/invalid."""
    try:
        changed = c["scope"].upper() == "CHANGED"
        av = AV[c["attack_vector"].upper()]
        ac = AC[c["attack_complexity"].upper()]
        pr = (PR_CHANGED if changed else PR_UNCHANGED)[c["privileges_required"].upper()]
        ui = UI[c["user_interaction"].upper()]
        conf = CIA[c["confidentiality"].upper()]
        integ = CIA[c["integrity"].upper()]
        avail = CIA[c["availability"].upper()]
    except (KeyError, AttributeError):
        return None

    iss = 1 - (1 - conf) * (1 - integ) * (1 - avail)
    if changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    exploitability = 8.22 * av * ac * pr * ui

    if impact <= 0:
        return 0.0
    if changed:
        return roundup(min(1.08 * (impact + exploitability), 10))
    return roundup(min(impact + exploitability, 10))


def severity(score: float | None) -> str:
    """Qualitative severity rating for a CVSS v3.x score."""
    if score is None:
        return ""
    if score == 0:
        return "NONE"
    if score < 4.0:
        return "LOW"
    if score < 7.0:
        return "MEDIUM"
    if score < 9.0:
        return "HIGH"
    return "CRITICAL"
