from __future__ import annotations

from types import MethodType

import pytest

from fugitive.agents.bir_fugitive import BeliefInformedRandomFugitiveAgent
from fugitive.agents.bootstrap_bir import BeliefInformedRandomMarshalAgent
from fugitive.engine import GameEngine
from fugitive.inference_diagnostics import BootstrapInferenceWorkDiagnostics
from fugitive.model import FugitiveAction, Observation, Phase, Role, RouteView


def fugitive_manhunt_observation() -> Observation:
    return Observation(
        role=Role.FUGITIVE,
        hand=(42,),
        pile_sizes=(0, 0, 0),
        route=(
            RouteView(0, 0, 0, (), True),
            RouteView(1, 5, 0, (), False),
            RouteView(2, 8, 0, (), False),
        ),
        guess_history=(),
        draw_history=(),
        round_number=4,
        phase=Phase.FUGITIVE_ACTION,
        legal_draw_piles=(),
    )


def test_fugitive_manhunt_rollout_reveals_and_recomputes_after_each_hit() -> None:
    observation = fugitive_manhunt_observation()
    agent = BeliefInformedRandomFugitiveAgent(3, manhunt_rollouts=4)

    def sequential_distribution(self, shadow):
        revealed = {slot.hideout for slot in shadow.route if slot.revealed}
        return {8: 1.0} if 5 in revealed else {5: 1.0}

    agent._shadow_manhunt_number_distribution = MethodType(  # type: ignore[method-assign]
        sequential_distribution,
        agent,
    )
    before = agent.rng.getstate()

    survival = agent._manhunt_survival_probability(
        observation,
        FugitiveAction(42),
    )

    assert survival == 0.0
    assert agent.rng.getstate() == before


def test_fugitive_manhunt_rollout_stops_at_the_first_miss() -> None:
    observation = fugitive_manhunt_observation()
    agent = BeliefInformedRandomFugitiveAgent(3, manhunt_rollouts=4)

    def misses_after_first_hit(self, shadow):
        revealed = {slot.hideout for slot in shadow.route if slot.revealed}
        return {7: 1.0} if 5 in revealed else {5: 1.0}

    agent._shadow_manhunt_number_distribution = MethodType(  # type: ignore[method-assign]
        misses_after_first_hit,
        agent,
    )

    assert agent._manhunt_survival_probability(
        observation,
        FugitiveAction(42),
    ) == 1.0


class _FakeBelief:
    is_empty = False
    marginals = {4: 0.25, 7: 0.75}


def test_marshal_manhunt_policy_parameters_are_configurable() -> None:
    observation = GameEngine(seed=1).observation(Role.MARSHAL)
    observation = Observation(
        role=observation.role,
        hand=observation.hand,
        pile_sizes=observation.pile_sizes,
        route=observation.route,
        guess_history=observation.guess_history,
        draw_history=observation.draw_history,
        round_number=observation.round_number,
        phase=Phase.MANHUNT,
        legal_draw_piles=(),
        play_history=observation.play_history,
    )
    agent = BeliefInformedRandomMarshalAgent(
        5,
        particle_count=8,
        manhunt_epsilon=0.0,
        manhunt_alpha=1.0,
    )

    distribution = agent.action_policy._manhunt_distribution(  # type: ignore[arg-type]
        observation,
        _FakeBelief(),
    )

    assert distribution == pytest.approx({(4,): 0.25, (7,): 0.75})


@pytest.mark.parametrize(
    ("agent", "kwargs"),
    [
        (BeliefInformedRandomFugitiveAgent, {"manhunt_rollouts": 0}),
        (BeliefInformedRandomFugitiveAgent, {"manhunt_epsilon": -0.1}),
        (BeliefInformedRandomFugitiveAgent, {"manhunt_alpha": 0.0}),
        (BeliefInformedRandomMarshalAgent, {"manhunt_epsilon": 1.1}),
        (BeliefInformedRandomMarshalAgent, {"manhunt_alpha": float("nan")}),
    ],
)
def test_manhunt_configuration_is_strict(agent, kwargs) -> None:
    with pytest.raises(ValueError):
        agent(**kwargs)


def test_inference_diagnostics_expose_particle_quality() -> None:
    engine = GameEngine(seed=5)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    agent = BeliefInformedRandomMarshalAgent(7, particle_count=32)
    agent.belief(engine.observation(Role.MARSHAL))

    diagnostics = agent.inference_diagnostics()
    assert diagnostics is not None
    assert isinstance(diagnostics.work, BootstrapInferenceWorkDiagnostics)
    assert diagnostics.quality.requested_particles == 32
    assert diagnostics.quality.particle_entries == 32
    assert 1 <= diagnostics.quality.unique_worlds <= 32
    assert 0.0 < diagnostics.quality.entry_ess <= 32
    assert (
        diagnostics.work.origin_fresh_sampling_proposals
        >= diagnostics.quality.particle_entries
    )
    assert (
        0.0
        < diagnostics.work.origin_fresh_sampling_acceptance_rate
        <= 1.0
    )
