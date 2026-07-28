"""Run OpenSpiel's C++ outcome-sampling MCCFR solver on Fugitive.

This is intentionally a thin integration example. The solver, policy table,
and chance distributions all come from OpenSpiel.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyspiel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--evaluation-games", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.evaluation_games < 1:
        parser.error("--evaluation-games must be positive")
    if args.max_rounds < 1:
        parser.error("--max-rounds must be positive")
    return args


def sample_policy_episode(game: pyspiel.Game, policy: object, seed: int) -> list[float]:
    """Sample one episode from an OpenSpiel policy and explicit chance nodes."""

    rng = np.random.RandomState(seed)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probabilities = zip(*state.chance_outcomes())
        else:
            action_probabilities = policy.action_probabilities(  # type: ignore[attr-defined]
                state
            )
            actions = tuple(action_probabilities)
            probabilities = tuple(action_probabilities.values())
        state.apply_action(int(rng.choice(actions, p=probabilities)))
    return list(state.returns())


def main() -> None:
    args = parse_args()

    game = pyspiel.load_game("fugitive", {"max_rounds": args.max_rounds})
    solver = pyspiel.OutcomeSamplingMCCFRSolver(game, seed=args.seed)
    for _ in range(args.iterations):
        solver.run_iteration()

    average_policy = solver.average_policy()
    returns = [
        sample_policy_episode(game, average_policy, args.seed + 10_000 + index)
        for index in range(args.evaluation_games)
    ]

    mean_returns = np.mean(np.asarray(returns), axis=0)
    print(
        json.dumps(
            {
                "game": str(game),
                "iterations": args.iterations,
                "evaluation_games": args.evaluation_games,
                "mean_returns": mean_returns.tolist(),
                "solver": "pyspiel.OutcomeSamplingMCCFRSolver",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
