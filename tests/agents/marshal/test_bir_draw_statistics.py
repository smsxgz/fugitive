from __future__ import annotations

from collections import defaultdict

import pytest

from fugitive.agents.marshal.bir.bootstrap import (
    BeliefInformedRandomMarshalAgent,
)
from fugitive.agents.common.baseline_utils import epsilon_softmax
from fugitive.agents.fugitive.hierarchical_random import HierarchicalRandomFugitiveAgent
from fugitive.agents.marshal.hierarchical_random import HierarchicalRandomMarshalAgent
from fugitive.game.engine import GameEngine
from fugitive.game.model import Observation, Phase, Role
from fugitive.agents.marshal.inference.particle_belief import MarshalParticleBelief


def _reference_best_guess_score(
    agent: BeliefInformedRandomMarshalAgent,
    observation: Observation,
    belief: MarshalParticleBelief,
) -> float:
    """Score a draw outcome through an explicitly conditioned belief."""

    candidates = agent.candidate_guess_sets(observation, belief)
    if not candidates or belief.is_empty:
        return 0.0
    failure_cost = agent.action_policy._escape_risk_after_draw(belief)
    hidden_count = agent.action_policy._hidden_count(observation)
    return max(
        agent.action_policy._guess_score(
            observation,
            belief,
            guess,
            failure_cost=failure_cost,
            hidden_count=hidden_count,
        )
        for guess in candidates
    )


def _reference_draw_pile_distribution(
    agent: BeliefInformedRandomMarshalAgent,
    observation: Observation,
) -> dict[int, float]:
    """Direct conditioned-belief implementation used only as a test oracle."""

    belief = agent.belief(observation)
    if belief.is_empty:
        if not observation.legal_draw_piles:
            return {}
        probability = 1.0 / len(observation.legal_draw_piles)
        return {pile: probability for pile in observation.legal_draw_piles}

    scores: dict[int, float] = {}
    for pile in observation.legal_draw_piles:
        posterior = belief.draw_card_posterior(pile)
        if not posterior:
            continue
        scores[pile] = sum(
            probability
            * _reference_best_guess_score(
                agent,
                observation,
                belief.conditioned_on_marshal_draw(pile, card),
            )
            for card, probability in posterior.items()
        )
    if not scores:
        return {}
    return epsilon_softmax(
        scores,
        epsilon=agent.epsilon,
        temperature=agent.temperature,
    )


def _marshal_draw_observation() -> Observation:
    engine = GameEngine(seed=0)
    fugitive = HierarchicalRandomFugitiveAgent(100)
    marshal = HierarchicalRandomMarshalAgent(200)

    while True:
        phase = engine.phase
        role = (
            Role.FUGITIVE
            if phase
            in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_DRAW, Phase.FUGITIVE_ACTION)
            else Role.MARSHAL
        )
        observation = engine.observation(role)
        if phase is Phase.MARSHAL_DRAW and observation.round_number == 2:
            return observation

        if phase in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_ACTION):
            engine.apply_fugitive_action(
                fugitive.choose_fugitive_action(observation)
            )
        elif phase is Phase.FUGITIVE_DRAW:
            engine.draw(fugitive.choose_draw_pile(observation))
        elif phase is Phase.MARSHAL_DRAW:
            engine.draw(marshal.choose_draw_pile(observation))
        elif phase in (Phase.MARSHAL_GUESS, Phase.MANHUNT):
            engine.apply_guess(marshal.choose_guess(observation))
        else:
            raise AssertionError(f"unexpected phase before round two: {phase}")


def test_optimized_draw_distribution_matches_conditioned_reference() -> None:
    observation = _marshal_draw_observation()
    agent = BeliefInformedRandomMarshalAgent(
        712,
        particle_count=128,
        max_guess_candidates=64,
    )

    reference = _reference_draw_pile_distribution(agent, observation)
    optimized = agent.draw_pile_distribution(observation)

    assert optimized.keys() == reference.keys()
    assert optimized == pytest.approx(reference, abs=1e-12)


def test_draw_outcome_statistics_match_conditioned_beliefs() -> None:
    observation = _marshal_draw_observation()
    agent = BeliefInformedRandomMarshalAgent(
        913,
        particle_count=128,
        max_guess_candidates=64,
    )
    belief = agent.belief(observation)
    statistics = belief.draw_outcome_statistics()

    for pile in observation.legal_draw_piles:
        posterior = belief.draw_card_posterior(pile)
        assert set(statistics[pile]) == set(posterior)
        assert {
            card: outcome.probability
            for card, outcome in statistics[pile].items()
        } == pytest.approx(posterior, abs=1e-12)

        cards = tuple(statistics[pile])
        sample_cards = {cards[0], cards[len(cards) // 2], cards[-1]}
        for card in sample_cards:
            outcome = statistics[pile][card]
            conditioned = belief.conditioned_on_marshal_draw(pile, card)

            assert dict(outcome.marginals) == pytest.approx(
                dict(conditioned.marginals),
                abs=1e-12,
            )

            route_masses: dict[tuple[int, ...], float] = defaultdict(float)
            for particle in conditioned.particles:
                hidden = tuple(
                    sorted(conditioned.current_hidden_hideouts(particle))
                )
                if hidden:
                    route_masses[hidden] += particle.weight
            assert dict(outcome.hidden_route_masses) == pytest.approx(
                route_masses,
                abs=1e-12,
            )
            assert agent.action_policy._candidate_guess_sets_from_masses(
                outcome.marginals,
                outcome.hidden_route_masses,
            ) == agent.candidate_guess_sets(observation, conditioned)

            candidates = agent.candidate_guess_sets(observation, conditioned)
            for guess in candidates[:24]:
                assert outcome.joint_success(guess) == pytest.approx(
                    conditioned.joint_success(guess),
                    abs=1e-12,
                )
            assert outcome.escape_risk == pytest.approx(
                agent.action_policy._escape_risk_after_draw(conditioned),
                abs=1e-12,
            )
