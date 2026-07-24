# Fugitive Ruleset: Project Canonical Rules

This document formalizes the rules used by this project. The user-provided
rules in `CANONICAL_RULES.md` are authoritative and take precedence over rules
from published editions or online implementations.

## Ruleset switches

```text
edition = 2017_first_edition
opening_hideouts = exactly_two
pass_bonus_draw = false
events = false
shift = false
physical_tells = false
```

In particular, passing means that the Fugitive does not play a Hideout. The
Fugitive keeps the one card drawn at the start of the turn but receives no
additional card for passing.

## Cards and setup

- There are 43 unique Hideout cards, numbered 0 through 42.
- Card 0 starts face up as the first Hideout.
- Independently shuffle three draw piles: 4-14, 15-28, and 29-41.
- The Fugitive starts with cards 1, 2, 3, and 42, then privately draws three
  cards from 4-14 and two cards from 15-28.
- The Marshal starts with an empty hand.
- Odd-numbered cards have Sprint value +1. Even-numbered cards have Sprint
  value +2.

The pile chosen for each draw is public. The identity of the drawn card is
private to the player who drew it.

## Turn sequence

1. On the first turn, the Fugitive draws no card and plays exactly two legal
   Hideouts.
2. On the Marshal's first turn, the Marshal draws two cards in total from any
   non-empty piles, then makes one guess action.
3. On later Fugitive turns, the Fugitive draws one card from any non-empty
   pile, then either plays one legal Hideout or passes.
4. On later Marshal turns, the Marshal draws one card from any non-empty pile,
   then makes one guess action.

## Playing a Hideout

Let `p` be the previous Hideout, `h` the new Hideout, and `S` the set of cards
played face down as Sprint cards under `h`. The play is legal exactly when:

```text
h > p
h - p <= 3 + sum(sprint_value(card) for card in S)
```

- The Hideout and all Sprint cards must be in the Fugitive's hand.
- The Hideout and Sprint cards are removed from the hand and played face down.
- The number of Sprint cards is public, but their identities and total value
  are hidden until revealed.
- Any number of Sprint cards may be used.
- Sprint may be overpaid, including when no Sprint is needed. This is a legal
  bluff, so revealed Sprint value must not be treated as an exact distance.
- The first Hideout may also use Sprint cards.

## Marshal guesses

The Marshal announces a non-empty set `G` of distinct numbers from 1 through
41. Let `U` be the set of currently face-down Hideouts.

```text
if G is a subset of U:
    reveal every Hideout in G and all Sprint cards under those Hideouts
else:
    reveal nothing
```

- The Marshal guesses card numbers, not route positions.
- Sprint cards never need to be guessed.
- A multi-number guess is all-or-nothing. A failed guess only proves that at
  least one member of `G` was not in `U` at that time.
- A failed guess does not prevent the Fugitive from playing that number later.

## Winning and Manhunt

- The Marshal wins immediately after revealing every Hideout currently in the
  Fugitive's route.
- Card 42 must be played as a normal legal Hideout; it is not a free move.
- When the Fugitive plays card 42, inspect the highest already revealed
  Hideout before 42:
  - If it is 30 or higher, the Fugitive wins immediately.
  - If it is 29 or lower, start a Manhunt.
- During a Manhunt, the Marshal guesses one number at a time. A correct guess
  reveals that Hideout and its Sprint cards and permits another guess. The
  first incorrect guess makes the Fugitive win. Revealing every remaining
  Hideout before an error makes the Marshal win.

## Information boundary

Public information includes the turn history, chosen draw piles, remaining
pile sizes, route length, Sprint-card counts, revealed cards, guesses, and
guess outcomes. The Fugitive privately knows the Fugitive hand and all hidden
route cards. The Marshal privately knows the Marshal hand. The shuffled pile
orders are chance state known to neither player.

Agents must act from their own observations or information states. They must
never receive the full hidden game state as input.

## Engine conventions

The printed rules do not fully specify these edge cases. This project freezes
them as follows so simulations are reproducible:

- If all three piles are empty, skip the required draw step and continue to
  the player's action.
- The Marshal's two first-turn draws are sequential. The Marshal sees the
  first card before choosing the second draw pile.
- Card 42 becomes public when played, but Sprint identities under it remain
  hidden during Manhunt.
- The engine and active agents have no artificial round horizon. A completed
  simulation always has a Fugitive or Marshal rules-defined winner.

The printed rules do not give arbitrary legal policies a 40-round bound. For
example, policies can deliberately repeat Pass and the same failed guess after
the draw piles empty. Built-in Marshal agents avoid such no-progress cycles by
excluding failed singleton guesses while the route length is unchanged, but
custom-agent callers are responsible for making progress.
