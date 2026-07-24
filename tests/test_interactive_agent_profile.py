from __future__ import annotations

from fugitive.agents.belief_informed_random import (
    BeliefInformedRandomMarshalAgent,
)
from fugitive.agents.registry import (
    INTERACTIVE_BIR_MAX_GUESS_CANDIDATES,
    INTERACTIVE_BIR_PARTICLE_COUNT,
    MARSHAL_AGENT_REGISTRY,
)
from fugitive.particle_belief import DEFAULT_PARTICLE_COUNT


def test_bir_registry_keeps_offline_and_interactive_profiles_separate() -> None:
    registration = MARSHAL_AGENT_REGISTRY["belief-informed-random"]

    offline = registration.create(7)
    interactive = registration.create(7, interactive=True)

    assert isinstance(offline, BeliefInformedRandomMarshalAgent)
    assert isinstance(interactive, BeliefInformedRandomMarshalAgent)
    assert offline.particle_count == DEFAULT_PARTICLE_COUNT
    assert interactive.particle_count == INTERACTIVE_BIR_PARTICLE_COUNT
    assert (
        interactive.max_guess_candidates
        == INTERACTIVE_BIR_MAX_GUESS_CANDIDATES
    )
