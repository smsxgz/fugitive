from __future__ import annotations

import pytest

from fugitive.game.model import Role
from fugitive.psro.ledger import PayoffLedger, PayoffSample
from fugitive.psro.payoff_matrix import PayoffPair
from fugitive.psro.population import PolicyIdentifier


def sample(sample_key: str, payoff: float) -> PayoffSample:
    pair = PayoffPair(
        PolicyIdentifier(Role.MARSHAL, "marshal"),
        PolicyIdentifier(Role.FUGITIVE, "fugitive"),
    )
    return PayoffSample(
        experiment_id="lesson-run",
        pair=pair,
        sample_key=sample_key,
        marshal_payoff=payoff,
        outcome={"winner": "Marshal" if payoff else "Fugitive"},
        metadata={"master_seed": sample_key},
    )


def test_ledger_is_resumable_idempotent_and_aggregates_payoffs(tmp_path) -> None:
    first = sample("seed:1", 1.0)
    second = sample("seed:2", 0.0)
    ledger = PayoffLedger(tmp_path)

    assert ledger.append(first)
    assert not ledger.append(first)
    assert ledger.append(second)

    resumed = PayoffLedger(tmp_path)
    assert resumed.get(first.sample_id) == first
    assert PayoffSample.from_json(first.to_json()) == first
    estimate = resumed.estimate(first.pair, experiment_id="lesson-run")
    assert estimate is not None
    assert estimate.marshal_payoff == 0.5
    assert estimate.sample_count == 2
    assert estimate.standard_error == pytest.approx(0.5)
    assert resumed.samples(experiment_id="lesson-run") == (first, second)
