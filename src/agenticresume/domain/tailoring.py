"""Deterministic rules over a finished tailoring plan — LLM-free, applied once
the reflection loop stops. The critic can only *flag* an overreaching move; turning
that flag into enforced honesty is a rule, not a judgement, so it lives here."""

from agenticresume.domain.models import Critique, Reframe


def redact_unfounded(reframes: list[Reframe], critique: Critique) -> list[Reframe]:
    """Downgrade every move the critic deemed 'unfounded' to an honest gap.

    The loop may accept a plan at its pass cap while a move still claims more than
    its cited fact supports. Shipping that is the one outcome this system exists to
    prevent, so the guard rewrites each flagged move into a gap — a truthful
    'nothing here supports this' — instead of letting a possible fabrication out.
    A no-op when nothing was flagged, so it is safe on the early-stop path too."""

    unfounded = {n.requirement_id for n in critique.notes if n.verdict == "unfounded"}
    if not unfounded:
        return reframes

    guarded: list[Reframe] = []
    for r in reframes:
        if r.requirement_id in unfounded and r.move != "gap":
            guarded.append(
                Reframe(
                    requirement_id=r.requirement_id,
                    move="gap",                 # gap invariant: no fact, no proposed text
                    source_fact_id=None,
                    proposed_text="",
                    rationale="redacted: the critic flagged this as unsupported by the cited fact",
                )
            )
        else:
            guarded.append(r)

    return guarded
