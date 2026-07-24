from __future__ import annotations

from pathlib import Path


STATIC = Path(__file__).parents[1] / "src" / "fugitive" / "web_static"


def test_refresh_preserves_an_unchanged_human_selection() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function selectionContext(game)" in javascript
    assert "const contextChanged =" in javascript
    assert "if (contextChanged) {\n    resetSelections();\n  }" in javascript


def test_mobile_hands_are_keyboard_accessible_horizontal_scrollers() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    stylesheet = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'row.setAttribute("aria-description"' in javascript
    assert "row.tabIndex = cards.length ? 0 : -1" in javascript
    assert "scroll-snap-type: x proximity" in stylesheet
    assert "scroll-snap-align: start" in stylesheet


def test_marshal_grid_marks_public_deduction_evidence_without_hiding_numbers() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    stylesheet = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'button.classList.toggle("is-revealed"' in javascript
    assert '"is-known-out"' in javascript
    assert '"is-failed-singleton"' in javascript
    assert "event.route_length === routeLength" in javascript
    assert ".guess-number.is-revealed" in stylesheet
    assert ".guess-number.is-known-out" in stylesheet
    assert ".guess-number.is-failed-singleton" in stylesheet


def test_stalled_session_has_explicit_continue_and_terminate_controls() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'id="continueButton"' in html
    assert 'id="terminateButton"' in html
    assert 'app.game.status === "stalled"' in javascript
    assert 'mutateGame("continue", {})' in javascript
    assert 'mutateGame("terminate", {})' in javascript


def test_starting_a_new_game_releases_the_old_server_session() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "async function showNewGameSetup()" in javascript
    assert "method: \"DELETE\"" in javascript
    assert "app.gameId = null" in javascript


def test_finished_game_can_copy_seed_and_export_trace() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'id="copySeedButton"' in html
    assert 'id="exportTraceButton"' in html
    assert "navigator.clipboard.writeText(app.game.seed)" in javascript
    assert "/export`" in javascript
    assert "new Blob" in javascript
