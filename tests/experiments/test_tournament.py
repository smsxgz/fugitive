from __future__ import annotations

import json
from types import SimpleNamespace

import experiments.tournament.runner as tournament_runner
from experiments.tournament.model import TournamentConfig, tournament_game_seed
from experiments.tournament.runner import run_tournament
from fugitive.game.model import Winner
from fugitive.runtime.manifest import MatchStatus


class FakeManifest:
    def __init__(self, status: MatchStatus, winner: Winner | None) -> None:
        self.status = status
        self.winner = winner
        self.reason = "terminal" if winner is not None else None
        self.rounds = 4 if winner is not None else None
        self.decision_count = 9

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            {
                "outcome": {
                    "status": self.status.value,
                    "winner": self.winner.value if self.winner else None,
                }
            },
            indent=indent,
        )


def fake_run(
    *,
    status: MatchStatus = MatchStatus.COMPLETED,
    winner: Winner | None = Winner.MARSHAL,
):
    return SimpleNamespace(
        manifest=FakeManifest(status, winner),
        inference_events=(),
        inference_diagnostic_failures=(),
        rollout_events=(),
        rollout_diagnostic_failures=(),
    )


def config(tmp_path, *, games: int, root_seed: int = 17) -> TournamentConfig:
    return TournamentConfig(
        fugitive_agents=("hierarchical-random",),
        marshal_agents=("hierarchical-random",),
        games_per_matchup=games,
        root_seed=root_seed,
        output_directory=tmp_path / "results",
    )


def test_tournament_writes_a_paired_matrix_and_summary(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_registered_match(**kwargs):
        calls.append(kwargs)
        winner = (
            Winner.MARSHAL
            if kwargs["marshal_name"] == "route-count-random"
            else Winner.FUGITIVE
        )
        return fake_run(winner=winner)

    monkeypatch.setattr(
        tournament_runner,
        "run_registered_match",
        fake_registered_match,
    )
    tournament_config = TournamentConfig(
        fugitive_agents=("hierarchical-random", "belief-informed-random"),
        marshal_agents=("hierarchical-random", "route-count-random"),
        games_per_matchup=2,
        root_seed=23,
        output_directory=tmp_path / "matrix",
    )

    report = run_tournament(tournament_config)

    assert len(calls) == 8
    assert [call["master_seed"] for call in calls[:4]] == [
        tournament_game_seed(23, 0)
    ] * 4
    assert [call["master_seed"] for call in calls[4:]] == [
        tournament_game_seed(23, 1)
    ] * 4
    assert len(report.matchups) == 4
    assert all(matchup.completed == 2 for matchup in report.matchups)
    assert all(
        matchup.marshal_win_rate == 1.0
        for matchup in report.matchups
        if matchup.marshal_agent == "route-count-random"
    )

    output = tournament_config.output_directory
    assert len((output / "games.jsonl").read_text(encoding="utf-8").splitlines()) == 8
    assert len(tuple((output / "manifests").rglob("*.json"))) == 8
    for filename in ("config.json", "summary.json", "summary.csv", "summary.md"):
        assert (output / filename).is_file()


def test_resume_extends_the_target_without_repeating_games(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    def fake_registered_match(**kwargs):
        calls.append(kwargs["master_seed"])
        return fake_run()

    monkeypatch.setattr(
        tournament_runner,
        "run_registered_match",
        fake_registered_match,
    )
    first = run_tournament(config(tmp_path, games=1))
    expanded = run_tournament(config(tmp_path, games=3), resume=True)
    repeated = run_tournament(config(tmp_path, games=3), resume=True)

    assert len(first.records) == 1
    assert len(expanded.records) == 3
    assert repeated == expanded
    assert calls == [tournament_game_seed(17, index) for index in range(3)]


def test_errors_are_excluded_from_completed_win_rates(monkeypatch, tmp_path) -> None:
    outcomes = iter(
        (
            fake_run(winner=Winner.MARSHAL),
            fake_run(status=MatchStatus.ERROR, winner=None),
            fake_run(winner=Winner.FUGITIVE),
        )
    )
    monkeypatch.setattr(
        tournament_runner,
        "run_registered_match",
        lambda **_kwargs: next(outcomes),
    )

    summary = run_tournament(config(tmp_path, games=3)).matchups[0]

    assert summary.recorded_games == 3
    assert summary.completed == 2
    assert summary.errors == 1
    assert summary.fugitive_wins == summary.marshal_wins == 1
    assert summary.marshal_win_rate == 0.5
