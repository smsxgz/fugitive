from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pytest

from fugitive.model import Role
from fugitive.payoff_ledger import (
    PayoffLedger,
    PayoffLedgerConflictError,
    PayoffLedgerCorruptionError,
    PayoffSample,
    deterministic_sample_id,
)
from fugitive.psro import PayoffPair, PolicyIdentifier


def _pair(marshal: str = "marshal", fugitive: str = "fugitive") -> PayoffPair:
    return PayoffPair(
        PolicyIdentifier(Role.MARSHAL, marshal),
        PolicyIdentifier(Role.FUGITIVE, fugitive),
    )


def _sample(
    key: str,
    payoff: float,
    *,
    experiment_id: str = "lesson-run-v1",
    pair: PayoffPair | None = None,
) -> PayoffSample:
    return PayoffSample(
        experiment_id=experiment_id,
        pair=pair or _pair(),
        sample_key=key,
        marshal_payoff=payoff,
        outcome={"winner": "Marshal" if payoff else "Fugitive", "rounds": 9},
        metadata={"master_seed": key, "tags": ["train", 1]},
    )


def _append_in_subprocess(
    directory: str,
    key: str,
    payoff: float,
) -> bool:
    return PayoffLedger(directory).append(_sample(key, payoff))


def test_sample_id_is_stable_and_domain_separated():
    pair = _pair()
    first = deterministic_sample_id("run", pair, "seed:7")

    assert first == deterministic_sample_id("run", pair, "seed:7")
    assert len(first) == 64
    assert first != deterministic_sample_id("other-run", pair, "seed:7")
    assert first != deterministic_sample_id("run", pair, "seed:8")
    assert first != deterministic_sample_id(
        "run", _pair(marshal="other-marshal"), "seed:7"
    )


def test_sample_round_trip_freezes_metadata_and_checks_derived_id():
    sample = _sample("seed:1", 1.0)
    restored = PayoffSample.from_json(sample.to_json())

    assert restored == sample
    assert restored.outcome["winner"] == "Marshal"
    with pytest.raises(TypeError):
        restored.metadata["new"] = "value"  # type: ignore[index]

    payload = sample.to_dict()
    payload["sample_id"] = "0" * 64
    with pytest.raises(ValueError, match="inconsistent sample_id"):
        PayoffSample.from_dict(payload)


def test_append_is_resumable_idempotent_and_rejects_conflicts(tmp_path):
    sample = _sample("seed:2", 1.0)
    first_process = PayoffLedger(tmp_path)

    assert first_process.append(sample)
    assert not first_process.append(sample)

    resumed = PayoffLedger(tmp_path)
    assert resumed.contains(sample.sample_id)
    assert resumed.get(sample.sample_id) == sample
    assert resumed.samples() == (sample,)

    with pytest.raises(PayoffLedgerConflictError):
        resumed.append(replace(sample, marshal_payoff=0.0))


def test_aggregate_returns_mean_standard_error_and_raw_samples(tmp_path):
    ledger = PayoffLedger(tmp_path)
    for index, payoff in enumerate((1.0, 0.0, 1.0)):
        ledger.append(_sample(f"seed:{index}", payoff))
    ledger.append(_sample("seed:other", 0.25, experiment_id="other-run"))

    estimate = ledger.estimate(_pair(), experiment_id="lesson-run-v1")
    assert estimate is not None
    assert estimate.marshal_payoff == pytest.approx(2 / 3)
    assert estimate.sample_count == 3
    assert estimate.standard_error == pytest.approx(1 / 3)
    assert len(ledger.samples(experiment_id="lesson-run-v1")) == 3
    assert len(ledger.payoff_entries(experiment_id="lesson-run-v1")) == 1

    single = ledger.estimate(_pair(), experiment_id="other-run")
    assert single is not None
    assert single.standard_error is None


def test_loader_rejects_corrupt_or_misnamed_published_records(tmp_path):
    ledger = PayoffLedger(tmp_path)
    sample = _sample("seed:3", 1.0)
    wrong_name = tmp_path / ("f" * 64 + ".json")
    wrong_name.write_text(sample.to_json(), encoding="utf-8")

    with pytest.raises(PayoffLedgerCorruptionError, match="filename"):
        ledger.samples()

    wrong_name.unlink()
    corrupt = tmp_path / ("e" * 64 + ".json")
    corrupt.write_text("{not JSON", encoding="utf-8")
    with pytest.raises(PayoffLedgerCorruptionError, match="invalid"):
        ledger.samples()


def test_processes_can_publish_distinct_samples_and_one_shared_sample(tmp_path):
    directory = str(tmp_path)
    jobs = [
        (directory, "shared", 1.0),
        (directory, "shared", 1.0),
        (directory, "seed:1", 0.0),
        (directory, "seed:2", 1.0),
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        published = tuple(executor.map(_run_append_job, jobs))

    assert sum(published) == 3
    samples = PayoffLedger(tmp_path).samples()
    assert len(samples) == 3
    assert {sample.sample_key for sample in samples} == {
        "shared",
        "seed:1",
        "seed:2",
    }


def _run_append_job(job: tuple[str, str, float]) -> bool:
    return _append_in_subprocess(*job)
