"""Observation-only SIR and independent-MH rejuvenation.

The kernel in this module has a deliberately narrow target:
:class:`~fugitive.constructive_belief.ConstraintUniformTarget`, the uniform
unnormalised density over complete worlds compatible with one Marshal
observation.  It is not a calibrated posterior over Fugitive behaviour.

An input importance batch is first converted to an equally weighted
population with systematic resampling.  Each resulting chain then receives
``steps_per_chain`` independent Metropolis-Hastings proposals from one fresh
batch produced by the same ``ConstructiveWorldSampler`` proposal density that
created the input batch.  The latter is a caller precondition: the current
constructive batch format records each world's ``log_q``, but does not yet
carry a machine-checkable proposal-kernel identifier.

Constructive sampling can reject incomplete raw proposals.  Its accepted
proposal density therefore differs from the recorded raw density by one
observation-specific normalising constant.  That common constant cancels in
the independent-MH ratio, so the recorded ``log_q`` values are sufficient.
This requires ``log_q`` to be the complete world-level proposal mass: a world
must either have one generating trace or aggregate all traces that reach it.

The finite SIR population and a finite number of MH moves are not stationary
samples merely by construction.  The MH kernel has the constraint-uniform
target as its stationary distribution and moves the finite approximation
toward that target.  A node budget is only a fatal engineering guard here;
budget-stopped proposal batches are rejected because conditioning on their
runtime cost could otherwise censor the proposal distribution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Sequence

from fugitive.constructive_belief import (
    CompiledMarshalConstraints,
    ConstraintUniformTarget,
    ConstructiveSampleBatch,
    ConstructiveSamplingReport,
    ConstructiveWorld,
    ConstructiveWorldSampler,
    SamplingBudget,
    WorldKey,
)
from fugitive.model import Observation
from fugitive.particle_belief import MarshalParticle


class InvalidRejuvenationBatchError(ValueError):
    """Raised when an initial importance batch cannot seed valid chains."""


class IncompleteProposalBatchError(RuntimeError):
    """Raised when the independent proposal batch is partial or invalid."""

    def __init__(self, report: ConstructiveSamplingReport) -> None:
        self.report = report
        super().__init__(
            "independent MH requires a complete importance-valid proposal "
            f"batch; requested={report.requested}, produced={report.produced}, "
            f"degraded={report.degraded}, "
            f"importance_valid={report.importance_valid}, "
            f"termination_reason={report.termination_reason!r}, "
            f"exhausted_stage={report.exhausted_stage!r}"
        )


@dataclass(frozen=True, slots=True)
class RejuvenationDiagnostics:
    """Quality and work counters for one SIR + independent-MH pass."""

    initial_effective_sample_size: float
    initial_unique_worlds: int
    resampled_unique_worlds: int
    final_unique_worlds: int
    final_max_clone_fraction: float
    chains: int
    steps_per_chain: int
    proposals: int
    accepted: int
    changed: int

    @property
    def initial_ess(self) -> float:
        """Compact alias used by experiment tables and debug overlays."""

        return self.initial_effective_sample_size

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposals if self.proposals else 0.0

    @property
    def change_rate(self) -> float:
        return self.changed / self.proposals if self.proposals else 0.0


@dataclass(frozen=True, slots=True)
class RejuvenatedWorldBatch:
    """An explicitly equally weighted population of complete worlds."""

    worlds: tuple[ConstructiveWorld, ...]
    diagnostics: RejuvenationDiagnostics
    ancestor_indices: tuple[int, ...]
    proposal_report: ConstructiveSamplingReport | None

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        if not self.worlds:
            return ()
        weight = 1.0 / len(self.worlds)
        return (weight,) * len(self.worlds)

    def to_particles(self) -> tuple[MarshalParticle, ...]:
        """Return the population as equally weighted Marshal particles."""

        if not self.worlds:
            return ()
        weight = 1.0 / len(self.worlds)
        return tuple(world.to_particle(weight=weight) for world in self.worlds)


def independent_mh_log_acceptance(
    current: ConstructiveWorld,
    proposal: ConstructiveWorld,
    *,
    constraints: CompiledMarshalConstraints,
    target: ConstraintUniformTarget,
) -> float:
    """Return ``min(0, log pi(y) + log q(x) - log pi(x) - log q(y))``.

    Both worlds' ``log_q`` values must describe the same independent proposal
    kernel.  The target is evaluated afresh rather than trusting the cached
    ``log_target`` fields.
    """

    if not math.isfinite(current.log_q) or not math.isfinite(proposal.log_q):
        raise ValueError("independent MH requires finite proposal log densities")
    current_target = target.log_density(current, constraints)
    proposal_target = target.log_density(proposal, constraints)
    if not math.isfinite(current_target):
        raise ValueError("current world is outside the rejuvenation target")
    if proposal_target == -math.inf:
        return -math.inf
    if not math.isfinite(proposal_target):
        raise ValueError("proposal target log density must be finite or -inf")
    return min(
        0.0,
        proposal_target
        + current.log_q
        - current_target
        - proposal.log_q,
    )


class IndependentMHRejuvenator:
    """Run systematic resampling followed by independent-MH chain moves.

    ``sampler`` must use the same proposal density as the sampler that created
    ``initial_batch``.  One proposal batch of size ``N * steps_per_chain`` is
    generated and reused across all ``N`` chains, so its constraint compiler
    and constructive caches are shared across the whole rejuvenation pass.
    """

    def __init__(
        self,
        *,
        sampler: ConstructiveWorldSampler | None = None,
        target: ConstraintUniformTarget | None = None,
    ) -> None:
        self.target = target or ConstraintUniformTarget()
        if not isinstance(self.target, ConstraintUniformTarget):
            raise TypeError("target must be a ConstraintUniformTarget")
        self.sampler = sampler or ConstructiveWorldSampler(target=self.target)
        sampler_target = getattr(self.sampler, "target", self.target)
        if not isinstance(sampler_target, ConstraintUniformTarget):
            raise TypeError(
                "proposal sampler must use a ConstraintUniformTarget"
            )

    def rejuvenate(
        self,
        observation: Observation,
        initial_batch: ConstructiveSampleBatch,
        *,
        steps_per_chain: int,
        seed: int | None = None,
        rng: random.Random | None = None,
        proposal_budget: SamplingBudget | None = None,
    ) -> RejuvenatedWorldBatch:
        """Return an equally weighted SIR + independent-MH population."""

        _validate_steps(steps_per_chain)
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        constraints = CompiledMarshalConstraints.from_observation(observation)
        weights = self._validate_initial_batch(initial_batch, constraints)
        initial_log_q = _world_log_q_by_key(initial_batch.worlds)
        chain_count = len(initial_batch.worlds)
        source_rng = rng if rng is not None else random.Random(seed)
        master_seed = source_rng.getrandbits(256)
        resampling_rng = _domain_rng(master_seed, "resampling")
        proposal_rng = _domain_rng(master_seed, "proposal")
        acceptance_rngs = tuple(
            _domain_rng(master_seed, "acceptance", chain_index)
            for chain_index in range(chain_count)
        )
        initial_ess = 1.0 / sum(weight * weight for weight in weights)
        ancestor_indices = _systematic_resample_indices(
            weights,
            chain_count,
            rng=resampling_rng,
        )
        chains = [initial_batch.worlds[index] for index in ancestor_indices]
        resampled_unique = len({world.world_key for world in chains})

        proposal_count = chain_count * steps_per_chain
        proposals: tuple[ConstructiveWorld, ...] = ()
        proposal_report: ConstructiveSamplingReport | None = None
        if proposal_count:
            proposal_batch = self.sampler.sample_batch(
                observation,
                particle_count=proposal_count,
                rng=proposal_rng,
                budget=proposal_budget,
            )
            self._validate_proposal_batch(
                proposal_batch,
                proposal_count,
                constraints,
            )
            proposals = proposal_batch.worlds
            proposal_report = proposal_batch.report
            proposal_log_q = _world_log_q_by_key(proposals)
            for world_key in initial_log_q.keys() & proposal_log_q.keys():
                if not math.isclose(
                    initial_log_q[world_key],
                    proposal_log_q[world_key],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise InvalidRejuvenationBatchError(
                        "the same world has different proposal log densities; "
                        "initial and MH proposals must use one world-level kernel"
                    )

        accepted = 0
        changed = 0
        proposal_index = 0
        for chain_index in range(chain_count):
            for _step in range(steps_per_chain):
                proposal = proposals[proposal_index]
                proposal_index += 1
                log_alpha = independent_mh_log_acceptance(
                    chains[chain_index],
                    proposal,
                    constraints=constraints,
                    target=self.target,
                )
                if _accept_log_probability(
                    log_alpha, acceptance_rngs[chain_index]
                ):
                    if proposal.world_key != chains[chain_index].world_key:
                        changed += 1
                    chains[chain_index] = proposal
                    accepted += 1

        worlds = tuple(chains)
        return RejuvenatedWorldBatch(
            worlds=worlds,
            diagnostics=RejuvenationDiagnostics(
                initial_effective_sample_size=initial_ess,
                initial_unique_worlds=len(
                    {world.world_key for world in initial_batch.worlds}
                ),
                resampled_unique_worlds=resampled_unique,
                final_unique_worlds=len({world.world_key for world in worlds}),
                final_max_clone_fraction=_max_clone_fraction(worlds),
                chains=chain_count,
                steps_per_chain=steps_per_chain,
                proposals=proposal_count,
                accepted=accepted,
                changed=changed,
            ),
            ancestor_indices=ancestor_indices,
            proposal_report=proposal_report,
        )

    def _validate_initial_batch(
        self,
        batch: ConstructiveSampleBatch,
        constraints: CompiledMarshalConstraints,
    ) -> tuple[float, ...]:
        report = batch.report
        complete = (
            bool(batch.worlds)
            and report.requested == report.produced == len(batch.worlds)
            and not report.degraded
            and report.importance_valid
            and report.termination_reason is None
        )
        if not complete:
            raise InvalidRejuvenationBatchError(
                "initial batch must be non-empty, complete, and importance-valid"
            )
        self._validate_world_scores(batch.worlds, constraints, source="initial")
        try:
            weights = batch.normalized_weights
        except RuntimeError as error:
            raise InvalidRejuvenationBatchError(str(error)) from error
        if (
            len(weights) != len(batch.worlds)
            or any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise InvalidRejuvenationBatchError(
                "initial batch has invalid normalized importance weights"
            )
        return weights

    def _validate_proposal_batch(
        self,
        batch: ConstructiveSampleBatch,
        expected: int,
        constraints: CompiledMarshalConstraints,
    ) -> None:
        report = batch.report
        complete = (
            report.requested == report.produced == expected == len(batch.worlds)
            and not report.degraded
            and report.importance_valid
            and report.termination_reason is None
        )
        if not complete:
            raise IncompleteProposalBatchError(report)
        self._validate_world_scores(batch.worlds, constraints, source="proposal")

    def _validate_world_scores(
        self,
        worlds: Sequence[ConstructiveWorld],
        constraints: CompiledMarshalConstraints,
        *,
        source: str,
    ) -> None:
        for world in worlds:
            if not math.isfinite(world.log_q):
                raise InvalidRejuvenationBatchError(
                    f"{source} world has a non-finite proposal log density"
                )
            actual_target = self.target.log_density(world, constraints)
            if not math.isfinite(actual_target):
                raise InvalidRejuvenationBatchError(
                    f"{source} world is outside the rejuvenation target"
                )
            if not math.isclose(
                world.log_target,
                actual_target,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise InvalidRejuvenationBatchError(
                    f"{source} world carries a stale target log density"
                )
        _world_log_q_by_key(worlds)


def _validate_steps(steps_per_chain: int) -> None:
    if (
        isinstance(steps_per_chain, bool)
        or not isinstance(steps_per_chain, int)
        or steps_per_chain < 0
    ):
        raise ValueError("steps_per_chain must be a non-negative integer")


def _systematic_resample_indices(
    weights: Sequence[float],
    count: int,
    *,
    rng: random.Random,
) -> tuple[int, ...]:
    """Return systematic-resampling ancestor indices."""

    if not weights:
        raise ValueError("weights must be non-empty")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("systematic-resampling weights must be finite and non-negative")
        running += weight
        cumulative.append(running)
    if running <= 0.0:
        raise ValueError("systematic-resampling weights must have positive mass")
    cumulative = [value / running for value in cumulative]
    cumulative[-1] = 1.0

    step = 1.0 / count
    threshold = rng.random() * step
    source_index = 0
    result: list[int] = []
    for sample_index in range(count):
        point = threshold + sample_index * step
        while source_index < len(cumulative) - 1 and point > cumulative[source_index]:
            source_index += 1
        result.append(source_index)
    return tuple(result)


def _accept_log_probability(log_probability: float, rng: random.Random) -> bool:
    if math.isnan(log_probability) or log_probability > 0.0:
        raise ValueError("log acceptance probability must be at most zero")
    if log_probability == 0.0:
        return True
    if log_probability == -math.inf:
        return False
    return rng.random() < math.exp(log_probability)


def _max_clone_fraction(worlds: Sequence[ConstructiveWorld]) -> float:
    if not worlds:
        return 0.0
    counts = Counter(world.world_key for world in worlds)
    return max(counts.values()) / len(worlds)


def _world_log_q_by_key(
    worlds: Sequence[ConstructiveWorld],
) -> dict[WorldKey, float]:
    densities: dict[WorldKey, float] = {}
    for world in worlds:
        previous = densities.get(world.world_key)
        if previous is not None and not math.isclose(
            previous,
            world.log_q,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise InvalidRejuvenationBatchError(
                "one world_key has multiple proposal log densities; log_q must "
                "be an aggregated world-level proposal mass"
            )
        densities[world.world_key] = world.log_q
    return densities


def _domain_rng(master_seed: int, domain: str, index: int | None = None) -> random.Random:
    """Derive a replay-stable random stream without using Python ``repr``."""

    digest = hashlib.sha256()
    digest.update(b"fugitive.rejuvenation-rng\x00v1\x00")
    for part in (
        str(master_seed).encode("ascii"),
        domain.encode("ascii"),
        b"" if index is None else str(index).encode("ascii"),
    ):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return random.Random(int.from_bytes(digest.digest(), "big"))


__all__ = [
    "IncompleteProposalBatchError",
    "IndependentMHRejuvenator",
    "InvalidRejuvenationBatchError",
    "RejuvenatedWorldBatch",
    "RejuvenationDiagnostics",
    "independent_mh_log_acceptance",
]
