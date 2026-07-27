from __future__ import annotations

import json
import math
import random

import pytest

from fugitive.agents.planning.policy_mixture import WeightedPolicySpec, default_policy_spec
from fugitive.agents.marshal.rollout_bir2u import RolloutBIR2UMarshalAgent
from fugitive.game.driver import GuessDecision
from fugitive.game.engine import GameEngine
from fugitive.game.model import FugitiveAction, Observation, Phase, Role
from fugitive.game.rules import is_legal_guess


def _hierarchical_fugitive_spec(*, weight: float = 1.0) -> WeightedPolicySpec:
    resolved = default_policy_spec("hierarchical-random", Role.FUGITIVE)
    return WeightedPolicySpec(resolved.spec, weight)


def _tiny_agent(seed: int) -> RolloutBIR2UMarshalAgent:
    return RolloutBIR2UMarshalAgent(
        seed,
        particle_count=4,
        max_guess_candidates=4,
        rollout_candidate_count=2,
        max_terminal_simulations=2,
        opponent_policies=(_hierarchical_fugitive_spec(),),
    )


def _marshal_guess_observation(seed: int = 7) -> Observation:
    engine = GameEngine(seed=seed)
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))
    while engine.phase is Phase.MARSHAL_DRAW:
        observation = engine.observation(Role.MARSHAL)
        engine.draw(observation.legal_draw_piles[0])
    assert engine.phase is Phase.MARSHAL_GUESS
    return engine.observation(Role.MARSHAL)


def test_real_terminal_rollout_distribution_is_valid_and_uses_base_shortlist():
    observation = _marshal_guess_observation()
    agent = _tiny_agent(101)

    base_distribution = agent.base_agent.guess_distribution(observation)
    shortlist = {
        guess
        for guess, _probability in sorted(
            base_distribution.items(),
            key=lambda item: (-item[1], repr(item[0])),
        )[: agent.rollout_candidate_count]
    }
    distribution = agent.guess_distribution(observation)
    repeated = _tiny_agent(101).guess_distribution(observation)

    assert distribution == repeated
    assert distribution
    assert set(distribution) == shortlist
    assert all(is_legal_guess(guess, manhunt=False) for guess in distribution)
    assert all(
        math.isfinite(probability) and probability > 0.0
        for probability in distribution.values()
    )
    assert math.fsum(distribution.values()) == pytest.approx(1.0)
def test_constructor_hashes_rng_state_without_consuming_it():
    rng = random.Random(99)
    state = rng.getstate()

    first = RolloutBIR2UMarshalAgent(
        rng=rng,
        particle_count=1,
        rollout_candidate_count=1,
        max_terminal_simulations=1,
        opponent_policies=(_hierarchical_fugitive_spec(),),
    )
    second = RolloutBIR2UMarshalAgent(
        rng=random.Random(99),
        particle_count=1,
        rollout_candidate_count=1,
        max_terminal_simulations=1,
        opponent_policies=(_hierarchical_fugitive_spec(),),
    )

    assert rng.getstate() == state
    assert first._rollout_salt == second._rollout_salt
    assert first.algorithm_id.endswith("v2")
    assert first.rollout_model_id.endswith("v2")
    assert first.max_terminal_simulations > 0


def test_single_candidate_shortcut_precedes_root_belief_resampling(
    monkeypatch,
):
    observation = _marshal_guess_observation()
    agent = _tiny_agent(71)
    action = GuessDecision((41,))

    def fail_belief(*_args, **_kwargs):
        raise AssertionError("single-candidate decisions must not resample belief")

    monkeypatch.setattr(agent.base_agent, "belief", fail_belief)
    distribution = agent._rollout_distribution(observation, (action,))
    diagnostics = agent.rollout_diagnostics()

    assert distribution == {action: 1.0}
    assert diagnostics is not None
    assert diagnostics.plan.single_candidate_shortcut
    assert diagnostics.plan.terminal_simulations == 0
    assert diagnostics.simulated_decisions == 0


def test_budgeted_marshal_rollout_reports_paired_work() -> None:
    observation = _marshal_guess_observation(11)
    agent = RolloutBIR2UMarshalAgent(
        72,
        particle_count=4,
        max_guess_candidates=4,
        rollout_candidate_count=3,
        max_terminal_simulations=5,
        opponent_policies=(_hierarchical_fugitive_spec(),),
    )

    distribution = agent.guess_distribution(observation)
    diagnostics = agent.rollout_diagnostics()

    assert diagnostics is not None
    assert diagnostics.plan.offered_candidates == 3
    assert diagnostics.plan.evaluated_candidates == 3
    assert diagnostics.plan.paired_scenarios == 1
    assert diagnostics.plan.terminal_simulations == 3
    assert diagnostics.plan.unused_terminal_simulations == 2
    assert len(distribution) == 3
    assert len(diagnostics.pairs) == 3
    json.dumps(diagnostics.to_dict(), allow_nan=False)


def test_rollout_marshal_relabels_existing_inference_snapshot() -> None:
    observation = _marshal_guess_observation(12)
    agent = _tiny_agent(73)

    agent.base_agent.guess_distribution(observation)
    snapshot = agent.inference_diagnostics()

    assert snapshot is not None
    assert snapshot.algorithm_id == agent.algorithm_id
    assert snapshot.backend_id == agent.belief_backend_id


@pytest.mark.parametrize("value", (True, 0))
def test_invalid_terminal_simulation_budget_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        RolloutBIR2UMarshalAgent(
            max_terminal_simulations=value,
            opponent_policies=(_hierarchical_fugitive_spec(),),
        )
def test_opponent_policy_configuration_round_trips_through_json():
    original = RolloutBIR2UMarshalAgent(
        31,
        particle_count=1,
        rollout_candidate_count=1,
        max_terminal_simulations=1,
        opponent_policies=(_hierarchical_fugitive_spec(weight=2.5),),
    )
    payload = json.loads(json.dumps(original.opponent_policies))

    restored = RolloutBIR2UMarshalAgent(
        31,
        particle_count=1,
        rollout_candidate_count=1,
        max_terminal_simulations=1,
        opponent_policies=payload,
    )

    assert restored.opponent_policies == original.opponent_policies
