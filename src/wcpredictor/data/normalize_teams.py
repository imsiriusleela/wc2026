"""Canonical team name normalisation.

Only stores known alias → canonical mappings.
Unknown names pass through unchanged (idempotent).
"""

_ALIASES: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "USA": "United States",
    "United States of America": "United States",
    "Chinese Taipei": "Taiwan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Northern Ireland": "Northern Ireland",
    "Faroe Islands": "Faroe Islands",
}


def canonical(name: str) -> str:
    return _ALIASES.get(name, name)
