from __future__ import annotations

from collections import defaultdict

import pytest

from fugitive.agents.belief_informed_random import (
    BeliefInformedRandomMarshalAgent,
)
from fugitive.agents.baseline_utils import epsilon_softmax
from fugitive.agents.hierarchical_random import (
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
from fugitive.engine import GameEngine
from fugitive.model import Observation, Phase, Role
from fugitive.particle_belief import MarshalParticleBelief


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
            * agent._best_guess_score(
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


def _marshal_draw_observations() -> tuple[Observation, Observation, Observation]:
    engine = GameEngine(seed=0)
    fugitive = HierarchicalRandomFugitiveAgent(100)
    marshal = HierarchicalRandomMarshalAgent(200)
    observations: list[Observation] = []

    while len(observations) < 3:
        phase = engine.phase
        role = (
            Role.FUGITIVE
            if phase
            in (Phase.FUGITIVE_OPENING, Phase.FUGITIVE_DRAW, Phase.FUGITIVE_ACTION)
            else Role.MARSHAL
        )
        observation = engine.observation(role)
        if phase is Phase.MARSHAL_DRAW:
            if not observations or observation != observations[-1]:
                observations.append(observation)
                if len(observations) == 3:
                    break

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

    first, second, round_two = observations
    assert first.round_number == 1 and len(first.hand) == 0
    assert second.round_number == 1 and len(second.hand) == 1
    assert round_two.round_number == 2 and len(round_two.hand) == 2
    return first, second, round_two


@pytest.mark.parametrize("observation_index", (0, 1, 2))
def test_optimized_draw_distribution_matches_conditioned_reference(
    observation_index: int,
) -> None:
    observation = _marshal_draw_observations()[observation_index]
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
    observation = _marshal_draw_observations()[2]
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
            assert agent._candidate_guess_sets_from_masses(
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
                agent._escape_risk_after_draw(conditioned),
                abs=1e-12,
            )


def test_optimized_draw_scans_each_hidden_route_once_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _marshal_draw_observations()[0]
    agent = BeliefInformedRandomMarshalAgent(
        111,
        particle_count=128,
        max_guess_candidates=64,
    )
    belief = agent.belief(observation)
    original = MarshalParticleBelief.current_hidden_hideouts
    calls = 0

    def counted(
        self: MarshalParticleBelief,
        particle: object,
    ) -> frozenset[int]:
        nonlocal calls
        calls += 1
        return original(self, particle)  # type: ignore[arg-type]

    monkeypatch.setattr(
        MarshalParticleBelief,
        "current_hidden_hideouts",
        counted,
    )

    first = belief.draw_outcome_statistics()
    second = belief.draw_outcome_statistics()

    assert first is second
    assert calls == len(belief.particles)


def test_optimized_draw_does_not_construct_conditioned_beliefs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _marshal_draw_observations()[0]
    agent = BeliefInformedRandomMarshalAgent(
        112,
        particle_count=128,
        max_guess_candidates=64,
    )
    agent.belief(observation)

    def forbidden(*_args: object, **_kwargs: object) -> MarshalParticleBelief:
        raise AssertionError("optimized draw constructed a conditioned belief")

    monkeypatch.setattr(
        MarshalParticleBelief,
        "conditioned_on_marshal_draw",
        forbidden,
    )

    distribution = agent.draw_pile_distribution(observation)

    assert distribution
    assert sum(distribution.values()) == pytest.approx(1.0)
