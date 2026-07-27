from __future__ import annotations

from types import MethodType

from fugitive.agents.fugitive.bir import BeliefInformedRandomFugitiveAgent
from fugitive.game.model import FugitiveAction, Observation, Phase, Role, RouteView


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
