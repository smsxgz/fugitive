# Fugitive Canonical Rules

This is the rules text supplied by the project owner on 2026-07-23. It is the
authoritative source for the implementation and experiments in this project.
Event and SHIFT cards are excluded. Passing does not grant a bonus draw.

## Overview

Fugitive is a quick two-player game where one player is the Fugitive moving
from hideout to hideout along an escape route, while the other player is the
Marshal trying to catch the Fugitive before they escape.

The deck is composed of 43 cards numbered 0-42, each representing a possible
hideout.

- If the Fugitive successfully plays Hideout #42, they have escaped and win
  the game, subject to the Manhunt rule below.
- If the Marshal uncovers every one of the Fugitive's hideouts before then,
  the Marshal wins.

## Setup

Place the 0 card face up in the center row as the Fugitive's starting location.

The Fugitive begins with cards 1, 2, 3, and 42.

Separate the remaining cards into three draw decks:

- Deck 1: cards 4-14. The Fugitive draws 3 cards from this deck.
- Deck 2: cards 15-28. The Fugitive draws 2 cards from this deck.
- Deck 3: cards 29-41. No cards are drawn from this deck during setup.

Shuffle each deck separately and place it face down. The Marshal begins with
no cards.

## Game Play

Players alternate turns throughout the game.

- During the Fugitive's turn, they may extend their escape route by creating a
  new hidden hideout to the right.
- During the Marshal's turn, they attempt to uncover those hidden hideouts
  through deduction and guessing.

## Fugitive First Turn

On the first turn, the Fugitive establishes exactly two hideouts. Each hideout
is placed face down in the center row to the right of the previous hideout,
starting with card 0.

After placing the initial two hideouts, the Fugitive's first turn ends.

## Creating Hideouts

Each newly played hideout must satisfy both rules:

- It has a higher number than the previous hideout.
- Without Sprint cards, it may increase by only 1, 2, or 3.

For example, if the previous hideout is 4, the next hideout may be 5, 6, or 7.
The Fugitive may never move backward by playing a lower-numbered hideout.

## Sprinting

To move farther than three spaces, the Fugitive may Sprint. Additional cards
are placed face down below the new hideout. Each Sprint card contributes +1 or
+2, as shown by its footsteps.

The card values follow parity:

- Odd-numbered cards 1, 3, 5, ..., 41 have Sprint value +1.
- Even-numbered cards 2, 4, 6, ..., 42 have Sprint value +2.
- Card 0 is the fixed starting Hideout and never enters a hand, so it cannot be
  used as Sprint.

The combined Sprint value must cover the extra distance beyond the normal
movement of three spaces.

Example:

- Previous hideout: 4.
- Desired hideout: 10.
- Normal movement reaches at most 7.
- Sprint cards totaling at least +3 must be placed beneath hideout 10.

The Fugitive may overpay for a Sprint by placing more Sprint value than
required, even when no Sprint movement is needed.

## Fugitive Normal Turn

At the start of each turn after the first, the Fugitive draws one card from any
of the three draw decks and adds it to their hand.

The Fugitive may then:

- Establish one new hideout by placing a card face down to the right of the
  most recent hideout, optionally using Sprint cards; or
- Announce "Pass" and establish no new hideout.

Passing does not grant an additional draw. The Fugitive may look at any of
their own face-down Hideout and Sprint cards at any time.

## Marshal First Turn

On the first turn, the Marshal draws two cards from any of the three draw decks
and adds them to their hand. The Marshal may then attempt to uncover the
Fugitive's hidden hideouts.

## Uncovering Hideouts

To uncover a hideout, the Marshal announces one or more hideout numbers. The
Fugitive compares the guesses with all face-down hideouts currently in the
escape route.

### Single Guess

- If the number matches any face-down hideout, the Fugitive reveals that
  hideout and every Sprint card beneath it.
- If the guess is incorrect, nothing is revealed.
- The Marshal does not point to a route position. A matching unrevealed
  hideout is revealed wherever it occurs in the route.

### Multiple Guesses

- Every guessed number must be an unrevealed hideout.
- If all guesses are correct, all matching hideouts and their Sprint cards are
  revealed.
- If even one number is incorrect, nothing is revealed.
- Sprint cards themselves are never guessed.

After the attempt, whether successful or not, the Marshal's turn ends.

## Marshal Normal Turn

At the start of each turn after the first, the Marshal draws one card from any
of the three draw decks and adds it to their hand. The Marshal may then attempt
to uncover hidden hideouts using the normal guessing rules.

## Game End

The game ends in one of two ways:

- Fugitive win: the Fugitive successfully plays Hideout #42 and survives any
  required Manhunt.
- Marshal win: the Marshal uncovers every existing Fugitive hideout before the
  Fugitive escapes.

There is no draw result in the game rules.

## Manhunt

After the Fugitive plays Hideout #42, check the highest-numbered revealed
hideout.

If the highest revealed hideout is less than 30, a Manhunt begins:

- The Marshal makes single-number guesses only.
- Each guess is resolved before the next guess is chosen.
- A correct guess reveals the hideout and allows the Marshal to continue.
- Revealing all remaining hideouts makes the Marshal win.
- The first incorrect guess immediately ends the Manhunt and makes the
  Fugitive win.

If the highest revealed hideout is 30 or higher, no Manhunt occurs and the
Fugitive wins immediately.

## Strategy Notes

### Fugitive

- Passing can be useful while waiting for a better normal draw.
- Sprinting consumes cards and may reduce future movement options.
- Overpaying Sprint can bluff the Marshal, but repeated bluffing is costly.
- Drawing from a higher-numbered deck can disguise future plans.
- A number the Marshal just guessed may be safer to play on a later turn.

### Marshal

- Record previous guesses and revealed hideouts.
- Multiple guesses are powerful but fail completely if one number is wrong.
- Drawing from higher-numbered decks can remove future options from the
  Fugitive.
- Revealing later hideouts and their Sprint cards may constrain earlier
  hideouts.
- The chosen draw decks are observable and may carry useful information.

Physical behavior, such as which face-down card a human player looks at, is a
table-play tell rather than a formal game action. It is excluded from the
computational model so that strategies and experiments are reproducible.

## Computational Edge Convention

The supplied rules do not specify what happens when all three draw piles are
empty. The engine uses the following deterministic engineering convention:
if a player is due to draw while every pile is empty, that draw is skipped and
the player proceeds directly to their action or guess. This does not create a
draw, winner, or extra turn. It only lets unusual long-running policy tests
continue to one of the two rules-defined game endings.
