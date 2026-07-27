from __future__ import annotations

import unittest

from fugitive.game.engine import GameEngine, play_game
from fugitive.game.model import (
    FugitiveAction,
    IllegalActionError,
    Phase,
    Role,
    Winner,
)
from fugitive.game.rules import (
    is_legal_fugitive_action,
    legal_fugitive_actions,
    sprint_value,
)


def draw_from_first_legal_pile(engine: GameEngine) -> int:
    role = (
        Role.FUGITIVE
        if engine.phase is Phase.FUGITIVE_DRAW
        else Role.MARSHAL
    )
    observation = engine.observation(role)
    return engine.draw(observation.legal_draw_piles[0])


def finish_opening_at_one_two(engine: GameEngine) -> None:
    engine.apply_fugitive_action(FugitiveAction(1))
    engine.apply_fugitive_action(FugitiveAction(2))


def finish_first_marshal_draws(engine: GameEngine) -> None:
    draw_from_first_legal_pile(engine)
    draw_from_first_legal_pile(engine)


class PassiveFugitive:
    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_fugitive_action(self, observation):
        if observation.phase is Phase.FUGITIVE_OPENING:
            return FugitiveAction(len(observation.route))
        return FugitiveAction(None)


class HarmlessMarshal:
    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation):
        # Draw-pile cards cannot be Hideouts 1 or 2 in the passive route.
        return (observation.hand[0],)


class EngineTests(unittest.TestCase):
    def test_public_invariant_validator_accepts_a_complete_trajectory(self):
        engine = GameEngine(seed=5)
        engine.validate_invariants()
        finish_opening_at_one_two(engine)
        engine.validate_invariants()
        finish_first_marshal_draws(engine)
        engine.validate_invariants()
        engine.apply_guess((1, 2))
        engine.validate_invariants()
        self.assertTrue(engine.is_terminal)

    def test_public_invariant_validator_detects_duplicate_card_locations(self):
        engine = GameEngine(seed=5)
        engine._state.hands[Role.MARSHAL].append(1)
        with self.assertRaises(AssertionError):
            engine.validate_invariants()

    def test_round_40_does_not_create_a_non_rules_ending(self):
        engine = GameEngine(seed=5)
        finish_opening_at_one_two(engine)
        finish_first_marshal_draws(engine)
        engine._state.round_number = 40
        engine.apply_guess((41,))

        self.assertFalse(engine.is_terminal)
        self.assertEqual(engine.phase, Phase.FUGITIVE_DRAW)
        self.assertEqual(engine.observation(Role.FUGITIVE).round_number, 41)

    def test_setup_and_seed_reproducibility(self):
        first = GameEngine(seed=17)
        second = GameEngine(seed=17)
        fugitive = first.observation(Role.FUGITIVE)
        marshal = first.observation(Role.MARSHAL)

        self.assertEqual(fugitive, second.observation(Role.FUGITIVE))
        self.assertEqual(len(fugitive.hand), 9)
        self.assertTrue({1, 2, 3, 42}.issubset(fugitive.hand))
        self.assertEqual(fugitive.pile_sizes, (8, 12, 13))
        self.assertEqual(fugitive.phase, Phase.FUGITIVE_OPENING)
        self.assertEqual(fugitive.route[0].index, 0)
        self.assertEqual(fugitive.route[0].hideout, 0)
        self.assertEqual(fugitive.route[0].sprint_cards, ())
        self.assertTrue(fugitive.route[0].revealed)
        self.assertTrue(marshal.route[0].revealed)
        self.assertEqual(marshal.hand, ())
        self.assertEqual(len(fugitive.draw_history), 5)
        self.assertTrue(all(record.card is not None for record in fugitive.draw_history))
        self.assertTrue(all(record.card is None for record in marshal.draw_history))

    def test_sprint_parity_legality_and_overpay_visibility(self):
        self.assertEqual(sprint_value(1), 1)
        self.assertEqual(sprint_value(2), 2)
        self.assertEqual(sprint_value(41), 1)
        self.assertEqual(sprint_value(42), 2)
        with self.assertRaises(ValueError):
            sprint_value(0)
        self.assertTrue(
            is_legal_fugitive_action(
                FugitiveAction(5, (42,)),
                (5, 42),
                0,
                allow_pass=False,
            )
        )
        self.assertFalse(
            is_legal_fugitive_action(
                FugitiveAction(0),
                (0, 1),
                0,
                allow_pass=False,
            )
        )
        self.assertFalse(
            is_legal_fugitive_action(
                FugitiveAction(1, (0,)),
                (0, 1),
                0,
                allow_pass=False,
            )
        )
        self.assertTrue(
            is_legal_fugitive_action(
                FugitiveAction(5, (2, 4)),
                (2, 4, 5, 9),
                4,
                allow_pass=False,
            )
        )
        self.assertFalse(
            is_legal_fugitive_action(
                FugitiveAction(10, (2,)),
                (2, 10),
                4,
                allow_pass=False,
            )
        )

        engine = GameEngine(seed=3)
        engine.apply_fugitive_action(FugitiveAction(1, (2, 3)))
        fugitive_node = engine.observation(Role.FUGITIVE).route[-1]
        marshal_node = engine.observation(Role.MARSHAL).route[-1]
        self.assertEqual(fugitive_node.hideout, 1)
        self.assertEqual(fugitive_node.sprint_cards, (2, 3))
        self.assertIsNone(marshal_node.hideout)
        self.assertEqual(marshal_node.sprint_count, 2)
        self.assertIsNone(marshal_node.sprint_cards)
        self.assertFalse(fugitive_node.revealed)
        self.assertFalse(marshal_node.revealed)

    def test_successful_guess_marks_the_same_route_slot_revealed_for_both_roles(self):
        engine = GameEngine(seed=11)
        finish_opening_at_one_two(engine)
        finish_first_marshal_draws(engine)

        self.assertTrue(engine.apply_guess((1,)))
        fugitive_route = engine.observation(Role.FUGITIVE).route
        marshal_route = engine.observation(Role.MARSHAL).route

        self.assertEqual(
            [slot.revealed for slot in fugitive_route],
            [True, True, False],
        )
        self.assertEqual(
            [slot.revealed for slot in marshal_route],
            [True, True, False],
        )
        self.assertEqual(fugitive_route[1].hideout, 1)
        self.assertEqual(marshal_route[1].hideout, 1)
        self.assertIsNone(marshal_route[2].hideout)

    def test_illegal_actions_raise(self):
        engine = GameEngine(seed=1)
        with self.assertRaises(IllegalActionError):
            engine.apply_fugitive_action(FugitiveAction(None))
        with self.assertRaises(IllegalActionError):
            engine.apply_fugitive_action(FugitiveAction(2, (2,)))
        with self.assertRaises(IllegalActionError):
            engine.draw(0)

        hand = engine.observation(Role.FUGITIVE).hand
        with self.assertRaises(IllegalActionError):
            engine.apply_fugitive_action(
                FugitiveAction(1, tuple(card for card in hand if card != 1))
            )

        engine.apply_fugitive_action(FugitiveAction(1))
        with self.assertRaises(IllegalActionError):
            engine.apply_fugitive_action(FugitiveAction(1))

    def test_first_marshal_draws_are_sequential_private_decisions(self):
        engine = GameEngine(seed=23)
        finish_opening_at_one_two(engine)
        self.assertEqual(engine.phase, Phase.MARSHAL_DRAW)

        before = engine.observation(Role.MARSHAL)
        first_card = engine.draw(2)
        after = engine.observation(Role.MARSHAL)
        fugitive_view = engine.observation(Role.FUGITIVE)
        self.assertEqual(engine.phase, Phase.MARSHAL_DRAW)
        self.assertIn(first_card, after.hand)
        self.assertEqual(after.pile_sizes[2], before.pile_sizes[2] - 1)
        self.assertEqual(after.draw_history[-1].card, first_card)
        self.assertIsNone(fugitive_view.draw_history[-1].card)
        self.assertEqual(after.legal_draw_piles, (0, 1, 2))

        engine.draw(1)
        self.assertEqual(engine.phase, Phase.MARSHAL_GUESS)

    def test_pass_keeps_only_the_normal_start_of_turn_draw(self):
        engine = GameEngine(seed=5)
        finish_opening_at_one_two(engine)
        finish_first_marshal_draws(engine)
        engine.apply_guess((41,))
        self.assertEqual(engine.phase, Phase.FUGITIVE_DRAW)

        before_hand = len(engine.observation(Role.FUGITIVE).hand)
        before_route = len(engine.observation(Role.FUGITIVE).route)
        draw_from_first_legal_pile(engine)
        engine.apply_fugitive_action(FugitiveAction(None))
        after = engine.observation(Role.FUGITIVE)
        self.assertEqual(len(after.hand), before_hand + 1)
        self.assertEqual(len(after.route), before_route)
        self.assertEqual(engine.phase, Phase.MARSHAL_DRAW)

    def test_multi_guess_is_all_or_nothing(self):
        engine = GameEngine(seed=11)
        finish_opening_at_one_two(engine)
        finish_first_marshal_draws(engine)

        self.assertFalse(engine.apply_guess((1, 3)))
        marshal_route = engine.observation(Role.MARSHAL).route
        self.assertIsNone(marshal_route[1].hideout)
        self.assertIsNone(marshal_route[2].hideout)
        self.assertEqual(engine.observation(Role.MARSHAL).guess_history[-1].route_length, 2)

        draw_from_first_legal_pile(engine)
        engine.apply_fugitive_action(FugitiveAction(None))
        draw_from_first_legal_pile(engine)
        self.assertTrue(engine.apply_guess((1, 2)))
        self.assertTrue(engine.is_terminal)
        self.assertEqual(engine.result().winner, Winner.MARSHAL)

    def test_observations_do_not_leak_hands_route_or_deck_order(self):
        engine = GameEngine(seed=29)
        fugitive_private_cards = set(engine.observation(Role.FUGITIVE).hand)
        marshal = engine.observation(Role.MARSHAL)
        self.assertFalse(fugitive_private_cards.intersection(marshal.hand))
        self.assertTrue(all(record.card is None for record in marshal.draw_history))

        engine.apply_fugitive_action(FugitiveAction(1))
        engine.apply_fugitive_action(FugitiveAction(2))
        marshal = engine.observation(Role.MARSHAL)
        self.assertEqual([view.hideout for view in marshal.route], [0, None, None])
        self.assertFalse(hasattr(marshal, "piles"))
        self.assertFalse(hasattr(marshal, "opponent_hand"))

        card = engine.draw(0)
        fugitive = engine.observation(Role.FUGITIVE)
        self.assertNotIn(card, fugitive.hand)
        self.assertIsNone(fugitive.draw_history[-1].card)

    def test_complete_games_replay_exactly_with_same_seed(self):
        first = play_game(EscapePlanner(), AlwaysMissMarshal(), seed=7)
        second = play_game(EscapePlanner(), AlwaysMissMarshal(), seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first.winner, Winner.FUGITIVE)
        self.assertEqual(first.reason, "manhunt_failed")

    def test_winner_enum_contains_only_the_two_rules_defined_winners(self):
        self.assertEqual(set(Winner), {Winner.FUGITIVE, Winner.MARSHAL})

    def test_set_valued_actions_are_stored_in_canonical_order(self):
        first = GameEngine(seed=5)
        second = GameEngine(seed=5)

        first.apply_fugitive_action(FugitiveAction(1, (3, 2)))
        second.apply_fugitive_action(FugitiveAction(1, (2, 3)))
        self.assertEqual(
            first.observation(Role.FUGITIVE),
            second.observation(Role.FUGITIVE),
        )
        self.assertEqual(
            first.observation(Role.FUGITIVE).route[-1].sprint_cards,
            (2, 3),
        )

        # Complete both openings identically, then compare reversed guesses.
        next_action = legal_fugitive_actions(
            first.observation(Role.FUGITIVE),
            max_sprint_cards=3,
            include_pass=False,
        )[0]
        first.apply_fugitive_action(next_action)
        second.apply_fugitive_action(next_action)
        finish_first_marshal_draws(first)
        finish_first_marshal_draws(second)
        first.apply_guess((2, 1))
        second.apply_guess((1, 2))
        self.assertEqual(
            first.observation(Role.FUGITIVE),
            second.observation(Role.FUGITIVE),
        )
        self.assertEqual(
            first.observation(Role.MARSHAL),
            second.observation(Role.MARSHAL),
        )
        self.assertEqual(
            first.observation(Role.MARSHAL).guess_history[-1].numbers,
            (1, 2),
        )


class EscapePlanner:
    """Deterministic legal planner used to exercise the real 42 path."""

    def choose_draw_pile(self, observation):
        previous = observation.route[-1].hideout
        assert previous is not None
        target = 0 if previous < 15 else 1 if previous < 29 else 2
        return next(
            (
                pile
                for pile in observation.legal_draw_piles
                if pile >= target
            ),
            observation.legal_draw_piles[-1],
        )

    def choose_fugitive_action(self, observation):
        if observation.phase is Phase.FUGITIVE_OPENING:
            return FugitiveAction(len(observation.route))

        # Try Escape first.  Since Sprint values are positive and at most two,
        # taking +2 cards first constructs a minimum-card payment directly.
        previous = observation.route[-1].hideout
        assert previous is not None
        required = max(0, 42 - previous - 3)
        candidates = sorted(
            (card for card in observation.hand if card != 42),
            key=lambda card: (-sprint_value(card), card),
        )
        payment = []
        paid = 0
        for card in candidates:
            if paid >= required:
                break
            payment.append(card)
            paid += sprint_value(card)
        if 42 in observation.hand and paid >= required:
            return FugitiveAction(42, tuple(payment))

        # Advance using a small action abstraction, keeping 42 for Escape.
        actions = [
            action
            for action in legal_fugitive_actions(
                observation,
                max_sprint_cards=2,
                include_pass=False,
            )
            if action.hideout != 42 and 42 not in action.sprint_cards
        ]
        if not actions:
            return FugitiveAction(None)
        return max(
            actions,
            key=lambda action: (
                action.hideout,
                -len(action.sprint_cards),
                tuple(-card for card in action.sprint_cards),
            ),
        )


class AlwaysMissMarshal:
    def __init__(self):
        self.saw_public_42 = False
        self.saw_hidden_42_sprints = False
        self.saw_42_not_revealed = False

    def choose_draw_pile(self, observation):
        return observation.legal_draw_piles[0]

    def choose_guess(self, observation):
        if observation.phase is Phase.MANHUNT:
            self.saw_public_42 = observation.route[-1].hideout == 42
            self.saw_hidden_42_sprints = observation.route[-1].sprint_cards is None
            self.saw_42_not_revealed = not observation.route[-1].revealed
            # A card in the Marshal hand cannot be an unrevealed Hideout.
            return (observation.hand[0],)
        return (observation.hand[0],)


class ManhuntTests(unittest.TestCase):
    def test_playing_42_starts_manhunt_and_first_miss_ends_it(self):
        # This seed and deterministic planner produce a legal escape trajectory.
        marshal = AlwaysMissMarshal()
        result = play_game(
            EscapePlanner(),
            marshal,
            seed=7,
        )
        self.assertEqual(result.winner, Winner.FUGITIVE)
        self.assertEqual(result.reason, "manhunt_failed")
        self.assertTrue(marshal.saw_public_42)
        self.assertTrue(marshal.saw_hidden_42_sprints)
        self.assertTrue(marshal.saw_42_not_revealed)
        self.assertTrue(any(getattr(record, "manhunt", False) for record in result.history))

    def test_highest_revealed_29_starts_manhunt_and_correct_guess_wins(self):
        engine = GameEngine(seed=1)
        node = type(engine._state.route[0])
        engine._state.route = [
            node(0, (), revealed=True),
            node(29, (), revealed=True),
            node(39, (), revealed=False),
        ]
        engine._state.hands[Role.FUGITIVE] = [42]
        engine._state.phase = Phase.FUGITIVE_ACTION

        engine.apply_fugitive_action(FugitiveAction(42))
        self.assertEqual(engine.phase, Phase.MANHUNT)
        engine.apply_guess((39,))
        self.assertEqual(engine.result().winner, Winner.MARSHAL)
        self.assertEqual(engine.result().reason, "manhunt_success")

    def test_highest_revealed_30_skips_manhunt(self):
        engine = GameEngine(seed=1)
        node = type(engine._state.route[0])
        engine._state.route = [
            node(0, (), revealed=True),
            node(30, (), revealed=True),
            node(39, (), revealed=False),
        ]
        engine._state.hands[Role.FUGITIVE] = [42]
        engine._state.phase = Phase.FUGITIVE_ACTION

        engine.apply_fugitive_action(FugitiveAction(42))
        self.assertEqual(engine.result().winner, Winner.FUGITIVE)
        self.assertEqual(engine.result().reason, "escape_no_manhunt")


if __name__ == "__main__":
    unittest.main()
