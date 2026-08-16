"""Scoring Rules based on Coverages
Simple Weighted Sum, acts as one half of the hiring decision
"""

from agenticresume.domain.models import Coverage, Requirement

_STATUS_WEIGHT = {"covered": 1.0, "partial": 0.5, "none": 0.0}
_NECESSITY_WEIGHT = {"must_have": 1.0, "nice_to_have": 0.25}


def weighted_coverage(coverage: tuple[Coverage, ...], requirements: tuple[Requirement, ...]) -> float:
    """"0.0–1.0 coverage score, weighting must-haves far above nice-to-haves."""

    necessity_map = {r.id: r.necessity for r in requirements}

    total = 0.0
    earned = 0.0

    for c in coverage:
        weight = _NECESSITY_WEIGHT[necessity_map[c.requirement_id]]
        total += weight
        earned += _STATUS_WEIGHT[c.status] * weight

    return earned / total if total > 0 else 0.0


def unmet_must_haves(coverages: tuple[Coverage, ...], requirements: tuple[Requirement, ...]) -> list[Requirement]:
    """Required requirements the candidate does not meet, the dealbreakers."""
    by_id = {r.id: r for r in requirements}
    return [
        by_id[c.requirement_id]
        for c in coverages
        if c.status == "none" and by_id[c.requirement_id].necessity == "must_have"
    ]