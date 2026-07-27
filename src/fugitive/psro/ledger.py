"""Append-only storage for individual empirical payoff games.

PSRO payoff cells can be expensive: one cell is usually the average of many
complete games.  Keeping only that average means an interrupted run loses all
games played since the last checkpoint.  :class:`PayoffLedger` instead stores
each completed game as one immutable JSON file.

The file-per-sample layout is intentionally simple.  A process first writes a
complete temporary file, then atomically publishes it with a hard link.  The
publication fails when another process has already published the same sample,
so no file lock or long-lived writer process is needed.  Equivalent duplicate
writes are idempotent; conflicting duplicates are reported as errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, cast
import uuid

from .payoff_matrix import PayoffEntry, PayoffEstimate, PayoffPair
from ..shared.reproducibility import (
    FrozenJSONValue,
    JSONValue,
    freeze_parameters,
    thaw_parameters,
)


PAYOFF_SAMPLE_SCHEMA = "fugitive.psro-payoff-sample"
PAYOFF_SAMPLE_VERSION = 1
PAYOFF_SAMPLE_ID_VERSION = "fugitive.psro-payoff-sample-id-v1"
_SAMPLE_ID_LENGTH = hashlib.sha256().digest_size * 2


class PayoffLedgerError(RuntimeError):
    """Base class for persistent payoff-ledger failures."""


class PayoffLedgerConflictError(PayoffLedgerError):
    """The same deterministic sample ID was used for different content."""


class PayoffLedgerCorruptionError(PayoffLedgerError):
    """A published ledger record is malformed or has the wrong filename."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty text without outer whitespace")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _is_sample_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SAMPLE_ID_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def deterministic_sample_id(
    experiment_id: str,
    pair: PayoffPair,
    sample_key: str,
) -> str:
    """Return the stable ID for one logical game.

    ``experiment_id`` should identify the full experiment configuration, while
    ``sample_key`` identifies a game within that experiment (for example,
    ``"payoff-seed:17"``).  Re-running the same logical game therefore targets
    the same file.  Policy parameters need not appear here when they are
    already covered by the experiment ID.
    """

    experiment_id = _require_non_empty_text(experiment_id, "experiment_id")
    sample_key = _require_non_empty_text(sample_key, "sample_key")
    if not isinstance(pair, PayoffPair):
        raise ValueError("pair must be a PayoffPair")
    identity = {
        "version": PAYOFF_SAMPLE_ID_VERSION,
        "experiment_id": experiment_id,
        "pair": pair.to_dict(),
        "sample_key": sample_key,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PayoffSample:
    """One complete game's raw payoff and replay-oriented metadata."""

    experiment_id: str
    pair: PayoffPair
    sample_key: str
    marshal_payoff: float
    outcome: Mapping[str, FrozenJSONValue] = field(default_factory=dict)
    metadata: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            _require_non_empty_text(self.experiment_id, "experiment_id"),
        )
        if not isinstance(self.pair, PayoffPair):
            raise ValueError("payoff sample requires a PayoffPair")
        object.__setattr__(
            self,
            "sample_key",
            _require_non_empty_text(self.sample_key, "sample_key"),
        )
        if isinstance(self.marshal_payoff, bool) or not isinstance(
            self.marshal_payoff, (int, float)
        ):
            raise ValueError("marshal_payoff must be a finite number")
        payoff = float(self.marshal_payoff)
        if not math.isfinite(payoff):
            raise ValueError("marshal_payoff must be a finite number")
        object.__setattr__(self, "marshal_payoff", payoff)
        object.__setattr__(self, "outcome", freeze_parameters(self.outcome))
        object.__setattr__(self, "metadata", freeze_parameters(self.metadata))

    @property
    def sample_id(self) -> str:
        return deterministic_sample_id(
            self.experiment_id,
            self.pair,
            self.sample_key,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": PAYOFF_SAMPLE_SCHEMA,
            "version": PAYOFF_SAMPLE_VERSION,
            "sample_id": self.sample_id,
            "experiment_id": self.experiment_id,
            "pair": self.pair.to_dict(),
            "sample_key": self.sample_key,
            "marshal_payoff": self.marshal_payoff,
            "outcome": thaw_parameters(self.outcome),
            "metadata": thaw_parameters(self.metadata),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PayoffSample":
        if data.get("schema") != PAYOFF_SAMPLE_SCHEMA:
            raise ValueError("unsupported payoff sample schema")
        if data.get("version") != PAYOFF_SAMPLE_VERSION:
            raise ValueError("unsupported payoff sample version")
        pair = PayoffPair.from_dict(_require_mapping(data.get("pair"), "pair"))
        sample = cls(
            experiment_id=_require_non_empty_text(
                data.get("experiment_id"), "experiment_id"
            ),
            pair=pair,
            sample_key=_require_non_empty_text(data.get("sample_key"), "sample_key"),
            marshal_payoff=cast(float, data.get("marshal_payoff")),
            outcome=_require_mapping(data.get("outcome", {}), "outcome"),
            metadata=_require_mapping(data.get("metadata", {}), "metadata"),
        )
        stored_id = data.get("sample_id")
        if not _is_sample_id(stored_id) or stored_id != sample.sample_id:
            raise ValueError("payoff sample has an inconsistent sample_id")
        return sample

    @classmethod
    def from_json(cls, payload: str) -> "PayoffSample":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid payoff sample JSON") from exc
        return cls.from_dict(_require_mapping(value, "payoff sample"))


class PayoffLedger:
    """A directory of immutable, independently publishable payoff samples."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.directory.is_dir():
            raise ValueError("payoff ledger path must be a directory")

    def _path(self, sample_id: str) -> Path:
        if not _is_sample_id(sample_id):
            raise ValueError("sample_id must be a lowercase SHA-256 digest")
        return self.directory / f"{sample_id}.json"

    def get(self, sample_id: str) -> PayoffSample | None:
        """Load one sample, returning ``None`` when it has not been run."""

        path = self._path(sample_id)
        if not path.exists():
            return None
        return self._read(path)

    def contains(self, sample_id: str) -> bool:
        return self.get(sample_id) is not None

    def append(self, sample: PayoffSample) -> bool:
        """Persist ``sample`` once.

        Returns ``True`` for the process that publishes a new record and
        ``False`` when an identical record already exists.  A different record
        with the same deterministic ID raises :class:`PayoffLedgerConflictError`.
        """

        if not isinstance(sample, PayoffSample):
            raise ValueError("ledger entries must be PayoffSample values")
        destination = self._path(sample.sample_id)
        existing = self.get(sample.sample_id)
        if existing is not None:
            self._require_equivalent(sample, existing)
            return False

        temporary = self.directory / (
            f".{sample.sample_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        payload = (sample.to_json() + "\n").encode("utf-8")
        try:
            with temporary.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            try:
                # A hard link publishes the already-complete bytes and, unlike
                # os.replace(), never overwrites another process's winner.
                os.link(temporary, destination)
                return True
            except FileExistsError:
                existing = self._read(destination)
                self._require_equivalent(sample, existing)
                return False
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def samples(
        self,
        *,
        experiment_id: str | None = None,
        pair: PayoffPair | None = None,
    ) -> tuple[PayoffSample, ...]:
        """Load a deterministic snapshot of all matching records."""

        if experiment_id is not None:
            experiment_id = _require_non_empty_text(experiment_id, "experiment_id")
        if pair is not None and not isinstance(pair, PayoffPair):
            raise ValueError("pair must be a PayoffPair")
        loaded: list[PayoffSample] = []
        for path in sorted(self.directory.glob("*.json")):
            sample = self._read(path)
            if experiment_id is not None and sample.experiment_id != experiment_id:
                continue
            if pair is not None and sample.pair != pair:
                continue
            loaded.append(sample)
        return tuple(loaded)

    def estimate(
        self,
        pair: PayoffPair,
        *,
        experiment_id: str | None = None,
    ) -> PayoffEstimate | None:
        """Aggregate one cell using the sample mean and its standard error."""

        values = tuple(
            sample.marshal_payoff
            for sample in self.samples(experiment_id=experiment_id, pair=pair)
        )
        if not values:
            return None
        count = len(values)
        mean = math.fsum(values) / count
        standard_error = None
        if count > 1:
            sample_variance = math.fsum(
                (value - mean) ** 2 for value in values
            ) / (count - 1)
            standard_error = math.sqrt(sample_variance / count)
        return PayoffEstimate(mean, count, standard_error)

    def payoff_entries(
        self,
        *,
        experiment_id: str | None = None,
    ) -> tuple[PayoffEntry, ...]:
        """Aggregate every policy pair represented in the selected samples."""

        selected = self.samples(experiment_id=experiment_id)
        pairs = sorted(
            {sample.pair for sample in selected},
            key=lambda item: (item.marshal.name, item.fugitive.name),
        )
        entries: list[PayoffEntry] = []
        for pair in pairs:
            estimate = self.estimate(pair, experiment_id=experiment_id)
            assert estimate is not None
            entries.append(PayoffEntry(pair, estimate))
        return tuple(entries)

    def _read(self, path: Path) -> PayoffSample:
        try:
            sample = PayoffSample.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PayoffLedgerCorruptionError(
                f"invalid payoff ledger record: {path.name}"
            ) from exc
        if path.name != f"{sample.sample_id}.json":
            raise PayoffLedgerCorruptionError(
                f"payoff ledger filename does not match its sample ID: {path.name}"
            )
        return sample

    @staticmethod
    def _require_equivalent(
        requested: PayoffSample,
        existing: PayoffSample,
    ) -> None:
        if requested != existing:
            raise PayoffLedgerConflictError(
                "a different payoff sample already uses deterministic ID "
                f"{requested.sample_id}"
            )


__all__ = [
    "PAYOFF_SAMPLE_ID_VERSION",
    "PAYOFF_SAMPLE_SCHEMA",
    "PAYOFF_SAMPLE_VERSION",
    "PayoffLedger",
    "PayoffLedgerConflictError",
    "PayoffLedgerCorruptionError",
    "PayoffLedgerError",
    "PayoffSample",
    "deterministic_sample_id",
]
