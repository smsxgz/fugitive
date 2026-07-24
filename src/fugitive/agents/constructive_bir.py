"""BIR-2 Marshal backed by constructive, importance-weighted worlds.

The action scoring and stochastic choice hierarchy are inherited unchanged
from BIR-1.  Only belief construction differs: every new Marshal observation
is sampled independently by :class:`ConstructiveWorldSampler`, and a belief is
published only when the sampler produced the complete requested batch with
valid importance weights.
"""

from __future__ import annotations

import math
import random

from fugitive.constructive_belief import (
    ConstructiveSamplingReport,
    ConstructiveWorldSampler,
)
from fugitive.model import Observation, Role
from fugitive.observation_protocol import stable_observation_seed
from fugitive.particle_belief import DEFAULT_PARTICLE_COUNT, MarshalParticleBelief

from .belief_informed_random import (
    DEFAULT_EPSILON,
    DEFAULT_MANHUNT_ALPHA,
    DEFAULT_MANHUNT_EPSILON,
    DEFAULT_MAX_GUESS_CANDIDATES,
    DEFAULT_TEMPERATURE,
    BeliefInformedRandomMarshalAgent,
)


CONSTRUCTIVE_SPRINT_BACKEND = "sequential"
_BELIEF_SEED_DOMAIN = "fugitive.bir2.constructive-belief.v1"


class ConstructiveBeliefConstructionError(RuntimeError):
    """Raised when BIR-2 cannot construct its complete particle batch."""

    def __init__(self, report: ConstructiveSamplingReport) -> None:
        self.report = report
        super().__init__(
            "BIR-2 requires a complete importance-valid constructive belief; "
            f"requested={report.requested}, produced={report.produced}, "
            f"degraded={report.degraded}, "
            f"importance_valid={report.importance_valid}, "
            f"termination_reason={report.termination_reason!r}, "
            f"exhausted_stage={report.exhausted_stage!r}, "
            f"proposals={report.proposals}, search_nodes={report.search_nodes}"
        )


class ConstructiveBeliefInformedRandomMarshalAgent(
    BeliefInformedRandomMarshalAgent
):
    """BIR-2 Marshal with observation-local constructive world sampling.

    The sampler never receives an engine or private Fugitive state.  Its seed
    is a stable function of the Marshal observation and the agent seed, so
    belief construction does not consume the policy's action RNG stream.
    """

    name = "constructive-belief-informed-random-marshal"

    def __init__(
        self,
        seed: int | random.Random | None = None,
        *,
        rng: random.Random | None = None,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
        particle_count: int = DEFAULT_PARTICLE_COUNT,
        max_guess_candidates: int = DEFAULT_MAX_GUESS_CANDIDATES,
        terminal_bonus_scale: float = 1.0,
        manhunt_epsilon: float = DEFAULT_MANHUNT_EPSILON,
        manhunt_alpha: float = DEFAULT_MANHUNT_ALPHA,
        _sampler: ConstructiveWorldSampler | None = None,
    ) -> None:
        # BIR-1 validates all shared policy parameters.  Its incremental
        # recovery threshold is unused because this subclass replaces belief().
        super().__init__(
            seed,
            rng=rng,
            epsilon=epsilon,
            temperature=temperature,
            particle_count=particle_count,
            min_unique_particles=1,
            max_guess_candidates=max_guess_candidates,
            terminal_bonus_scale=terminal_bonus_scale,
            manhunt_epsilon=manhunt_epsilon,
            manhunt_alpha=manhunt_alpha,
        )
        if (
            _sampler is not None
            and _sampler.sprint_backend != CONSTRUCTIVE_SPRINT_BACKEND
        ):
            raise ValueError(
                "injected sampler must use the sequential Sprint backend"
            )
        self.sprint_backend = CONSTRUCTIVE_SPRINT_BACKEND
        self._constructive_sampler = _sampler or ConstructiveWorldSampler(
            sprint_backend=CONSTRUCTIVE_SPRINT_BACKEND
        )
        self._constructive_reports: dict[
            Observation, ConstructiveSamplingReport
        ] = {}

    @property
    def last_sampling_report(self) -> ConstructiveSamplingReport | None:
        """Return the report for the most recently used successful belief."""

        if self._latest_belief is None:
            return None
        return self._constructive_reports.get(self._latest_belief.observation)

    def belief(self, observation: Observation) -> MarshalParticleBelief:
        """Construct or retrieve a strict complete belief for ``observation``."""

        if observation.role is not Role.MARSHAL:
            raise ValueError("Marshal belief requires a Marshal observation")
        cached = self._belief_cache.get(observation)
        if cached is not None:
            self._latest_belief = cached
            return cached

        belief_seed = stable_observation_seed(
            observation,
            domain=_BELIEF_SEED_DOMAIN,
            salt=self._belief_salt,
        )
        batch = self._constructive_sampler.sample_batch(
            observation,
            particle_count=self.particle_count,
            seed=belief_seed,
        )
        report = batch.report
        complete = (
            report.requested == self.particle_count
            and report.produced == self.particle_count
            and len(batch.worlds) == self.particle_count
            and not report.degraded
            and report.importance_valid
            and report.termination_reason is None
        )
        if not complete:
            raise ConstructiveBeliefConstructionError(report)

        weights = batch.normalized_weights
        if (
            len(weights) != self.particle_count
            or any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RuntimeError(
                "constructive sampler returned invalid normalized importance weights"
            )
        particles = tuple(
            world.to_particle(weight=weight)
            for world, weight in zip(batch.worlds, weights, strict=True)
        )
        sampled = MarshalParticleBelief(
            observation,
            particles,
            sampling_attempts=report.proposals,
            sampling_accepted=report.produced,
            sampling_exhausted=False,
        )

        if len(self._belief_cache) >= 8:
            oldest = next(iter(self._belief_cache))
            self._belief_cache.pop(oldest)
            self._constructive_reports.pop(oldest, None)
        self._belief_cache[observation] = sampled
        self._constructive_reports[observation] = report
        self._latest_belief = sampled
        self._belief_fresh_resamples += 1
        return sampled


__all__ = [
    "ConstructiveBeliefConstructionError",
    "ConstructiveBeliefInformedRandomMarshalAgent",
    "CONSTRUCTIVE_SPRINT_BACKEND",
]
