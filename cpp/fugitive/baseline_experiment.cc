// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "open_spiel/games/fugitive/belief.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace fugitive {
namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

enum class GuardLowExhausted {
  kLift,
  kWait,
};

enum class MarshalGuessMode {
  kArgmaxOnly,
  kCertainOnly,
  kCertainPlusArgmax,
};

struct Options {
  int games = 100;
  std::uint64_t seed_start = 0;
  int max_rounds = kDefaultMaxRounds;
  int manhunt_particles = 64;
  int dead_card_sprints = 0;
  MarshalGuessMode guess_mode = MarshalGuessMode::kCertainPlusArgmax;
  GuardLowExhausted guard_low_exhausted = GuardLowExhausted::kLift;
};

struct FugitivePolicyState {
  std::vector<Action> pending_dead_sprints;
};

struct GameResult {
  Winner winner = Winner::kNone;
  std::string terminal_reason;
  int rounds = 0;
  int passes = 0;
  int sprint_cards = 0;
  int forced_opening_sprint_cards = 0;
  int dead_card_sprint_cards = 0;
  int dead_card_sprint_plays = 0;
  bool reached_42 = false;
  bool entered_manhunt = false;
  std::uint64_t normal_belief_calls = 0;
  std::int64_t normal_belief_time_us = 0;
  int unrestricted_argmax_ge_30_turns = 0;
  int guard_restriction_turns = 0;
  int forced_gamble_turns = 0;
  int forced_gamble_losses = 0;
  int forced_gamble_certain_cards_lost = 0;
  int banked_certain_turns = 0;
  int banked_certain_cards = 0;
  int cover_all_attempts = 0;
  int cover_all_wins = 0;
  int guard_lift_turns = 0;
  int guard_wait_turns = 0;
  bool manhunt_disabled_by_guard_lift = false;
  std::uint64_t manhunt_evaluator_calls = 0;
  std::uint64_t manhunt_evaluator_exact_empirical = 0;
  std::int64_t manhunt_evaluator_time_us = 0;
  std::int64_t elapsed_us = 0;
};

struct Summary {
  int games = 0;
  int fugitive_wins = 0;
  int marshal_wins = 0;
  int draws = 0;
  int timeouts = 0;
  std::map<std::string, int> terminal_reasons;
  std::int64_t round_sum = 0;
  std::int64_t pass_sum = 0;
  int games_with_pass = 0;
  std::int64_t sprint_card_sum = 0;
  int games_with_sprint = 0;
  std::int64_t forced_opening_sprint_card_sum = 0;
  std::int64_t dead_card_sprint_card_sum = 0;
  std::int64_t dead_card_sprint_play_sum = 0;
  int games_with_dead_card_sprint = 0;
  int reached_42 = 0;
  int entered_manhunt = 0;
  int marshal_wins_after_manhunt = 0;
  std::uint64_t normal_belief_calls = 0;
  std::int64_t normal_belief_time_us = 0;
  int unrestricted_argmax_ge_30_turns = 0;
  int guard_restriction_turns = 0;
  int games_with_guard_restriction = 0;
  int forced_gamble_turns = 0;
  int forced_gamble_losses = 0;
  int forced_gamble_certain_cards_lost = 0;
  int banked_certain_turns = 0;
  int banked_certain_cards = 0;
  int cover_all_attempts = 0;
  int cover_all_wins = 0;
  int guard_lift_turns = 0;
  int games_with_guard_lift = 0;
  int manhunt_disabled_by_guard_lift = 0;
  int guard_wait_turns = 0;
  int games_with_guard_wait = 0;
  std::uint64_t manhunt_evaluator_calls = 0;
  std::uint64_t manhunt_evaluator_exact_empirical = 0;
  std::int64_t manhunt_evaluator_time_us = 0;
  std::int64_t elapsed_us = 0;
};

FugitiveState& AsFugitive(State* state) {
  return down_cast<FugitiveState&>(*state);
}

const FugitiveState& AsFugitive(const State* state) {
  return down_cast<const FugitiveState&>(*state);
}

bool Contains(const std::vector<Action>& actions, Action action) {
  return std::find(actions.begin(), actions.end(), action) != actions.end();
}

const char* GuardLowExhaustedString(GuardLowExhausted mode) {
  switch (mode) {
    case GuardLowExhausted::kLift:
      return "lift";
    case GuardLowExhausted::kWait:
      return "wait";
  }
  SpielFatalError("Unknown guard low-exhausted mode");
}

const char* MarshalGuessModeString(MarshalGuessMode mode) {
  switch (mode) {
    case MarshalGuessMode::kArgmaxOnly:
      return "argmax_only";
    case MarshalGuessMode::kCertainOnly:
      return "certain_only";
    case MarshalGuessMode::kCertainPlusArgmax:
      return "certain_plus_argmax";
  }
  SpielFatalError("Unknown Marshal guess mode");
}

std::int64_t Microseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

Action SampleChance(const State& state, std::mt19937_64* rng) {
  const double sample = std::generate_canonical<double, 64>(*rng);
  return SampleAction(state.ChanceOutcomes(), sample).first;
}

std::uint64_t ManhuntSeed(std::uint64_t game_seed,
                          const std::string& information_state) {
  std::uint64_t hash = 1469598103934665603ULL ^ game_seed;
  for (unsigned char byte : information_state) {
    hash = (hash ^ byte) * 1099511628211ULL;
  }
  return hash ^ 0xd1b54a32d192ed03ULL;
}

int MinimumSprintsToCommit(const State& state) {
  const std::vector<Action> legal = state.LegalActions();
  if (Contains(legal, kCommitAction)) return 0;

  int best = std::numeric_limits<int>::max();
  for (Action action : legal) {
    if (!CanBeSprint(action)) continue;
    std::unique_ptr<State> child = state.Clone();
    child->ApplyAction(action);
    const int remaining = MinimumSprintsToCommit(*child);
    if (remaining != std::numeric_limits<int>::max()) {
      best = std::min(best, 1 + remaining);
    }
  }
  return best;
}

Action MinimumSprintAction(const State& state) {
  const std::vector<Action> legal = state.LegalActions();
  if (Contains(legal, kCommitAction)) return kCommitAction;

  int best_cost = std::numeric_limits<int>::max();
  Action best_action = kInvalidAction;
  for (Action action : legal) {
    if (!CanBeSprint(action)) continue;
    std::unique_ptr<State> child = state.Clone();
    child->ApplyAction(action);
    const int remaining = MinimumSprintsToCommit(*child);
    if (remaining < best_cost) {
      best_cost = remaining;
      best_action = action;
    }
  }
  SPIEL_CHECK_NE(best_action, kInvalidAction);
  return best_action;
}

Action ChooseFugitiveHideout(const FugitiveState& state) {
  const std::vector<Action> legal = state.LegalActions();
  const int previous = state.route().back().hideout;

  if (Contains(legal, 42)) return 42;

  Action best = kInvalidAction;
  for (Action action : legal) {
    if (action > previous && action < 42 && action - previous <= 3) {
      best = std::max(best, action);
    }
  }
  if (best != kInvalidAction) return best;
  if (Contains(legal, kPassAction)) return kPassAction;

  // Pass is unavailable during the two opening plays. If the second opening
  // play cannot be made within distance three, choose the route extension
  // requiring the fewest Sprint cards; ties keep the L1 preference for the
  // larger Hideout.
  int best_sprints = std::numeric_limits<int>::max();
  for (Action action : legal) {
    if (action <= previous || action >= 42) continue;
    std::unique_ptr<State> child = state.Clone();
    child->ApplyAction(action);
    const int sprints = MinimumSprintsToCommit(*child);
    if (sprints < best_sprints ||
        (sprints == best_sprints && action > best)) {
      best_sprints = sprints;
      best = action;
    }
  }
  SPIEL_CHECK_NE(best, kInvalidAction);
  SPIEL_CHECK_NE(best_sprints, std::numeric_limits<int>::max());
  return best;
}

std::vector<Action> ChooseDeadCardSprints(const FugitiveState& state,
                                          Action hideout, int maximum_cards) {
  if (maximum_cards == 0 || state.opening_plays_remaining() > 0 ||
      hideout == kPassAction || hideout == kMaxCard) {
    return {};
  }

  const int previous = state.route().back().hideout;
  std::vector<Action> dead_cards;
  for (int card : state.hand(kFugitivePlayer)) {
    if (card <= previous && CanBeSprint(card)) dead_cards.push_back(card);
  }
  std::sort(dead_cards.begin(), dead_cards.end(), [](Action lhs, Action rhs) {
    if (SprintValue(lhs) != SprintValue(rhs)) {
      return SprintValue(lhs) < SprintValue(rhs);
    }
    return lhs < rhs;
  });
  if (static_cast<int>(dead_cards.size()) > maximum_cards) {
    dead_cards.resize(maximum_cards);
  }
  std::sort(dead_cards.begin(), dead_cards.end());
  return dead_cards;
}

void ApplyFugitiveDecision(State* state, int dead_card_sprints,
                           FugitivePolicyState* policy, GameResult* result) {
  FugitiveState& fugitive = AsFugitive(state);
  const Phase phase = fugitive.phase();
  const std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_FALSE(legal.empty());

  if (phase == Phase::kFugitiveDrawChoice) {
    state->ApplyAction(legal.front());
    return;
  }
  if (phase == Phase::kFugitiveHideout) {
    const Action action = ChooseFugitiveHideout(fugitive);
    policy->pending_dead_sprints =
        ChooseDeadCardSprints(fugitive, action, dead_card_sprints);
    if (!policy->pending_dead_sprints.empty()) {
      ++result->dead_card_sprint_plays;
    }
    if (action == kPassAction) ++result->passes;
    state->ApplyAction(action);
    return;
  }

  SPIEL_CHECK_EQ(phase, Phase::kFugitiveSprint);
  const bool opening = fugitive.opening_plays_remaining() > 0;
  if (!policy->pending_dead_sprints.empty()) {
    const Action action = policy->pending_dead_sprints.front();
    SPIEL_CHECK_TRUE(Contains(legal, action));
    policy->pending_dead_sprints.erase(
        policy->pending_dead_sprints.begin());
    ++result->sprint_cards;
    ++result->dead_card_sprint_cards;
    state->ApplyAction(action);
    return;
  }

  const Action action = MinimumSprintAction(*state);
  if (action != kCommitAction) {
    ++result->sprint_cards;
    if (opening) ++result->forced_opening_sprint_cards;
  }
  state->ApplyAction(action);
}

struct MarshalGuessPlan {
  std::vector<Action> guesses;
  Action top_argmax = kInvalidAction;
  Action added_argmax = kInvalidAction;
  int certain_count = 0;
  bool forced_gamble = false;
};

MarshalGuessPlan BuildMarshalGuesses(
    const MarshalCompletionResult& completion, int hidden_positions,
    MarshalGuessMode mode, int maximum_guess) {
  MarshalGuessPlan plan;
  long double top_probability = 0.0L;
  int uncertain_argmax = -1;
  long double uncertain_probability = 0.0L;
  for (int card = kMinCard; card <= maximum_guess; ++card) {
    const long double probability =
        completion.UniformConsistentHiddenHideoutProbability(card);
    if (probability > top_probability) {
      top_probability = probability;
      plan.top_argmax = card;
    }
    const bool certain = probability >= 1.0L - 1e-15L;
    if (certain && mode != MarshalGuessMode::kArgmaxOnly) {
      plan.guesses.push_back(card);
    } else if (probability > uncertain_probability) {
      uncertain_probability = probability;
      uncertain_argmax = card;
    }
  }
  plan.certain_count = static_cast<int>(plan.guesses.size());
  SPIEL_CHECK_LE(plan.certain_count, hidden_positions);

  const bool needs_argmax =
      mode == MarshalGuessMode::kArgmaxOnly ||
      (mode == MarshalGuessMode::kCertainOnly && plan.guesses.empty()) ||
      (mode == MarshalGuessMode::kCertainPlusArgmax &&
       static_cast<int>(plan.guesses.size()) < hidden_positions);
  if (needs_argmax && uncertain_argmax != -1) {
    plan.guesses.push_back(uncertain_argmax);
    plan.added_argmax = uncertain_argmax;
  }
  plan.forced_gamble =
      mode == MarshalGuessMode::kCertainPlusArgmax &&
      plan.certain_count > 0 && plan.added_argmax != kInvalidAction;
  std::sort(plan.guesses.begin(), plan.guesses.end());
  plan.guesses.erase(
      std::unique(plan.guesses.begin(), plan.guesses.end()),
      plan.guesses.end());
  SPIEL_CHECK_FALSE(plan.guesses.empty());
  return plan;
}

void ApplyNormalMarshalGuess(State* state, bool guard,
                             MarshalGuessMode guess_mode,
                             GuardLowExhausted guard_low_exhausted,
                             GameResult* result) {
  const Clock::time_point belief_start = Clock::now();
  const MarshalBeliefInput input = BuildMarshalBeliefInput(
      state->InformationStateString(kMarshalPlayer));
  const MarshalCompletionResult completion = ComputeMarshalCompletion(input);
  result->normal_belief_time_us +=
      Microseconds(belief_start, Clock::now());
  ++result->normal_belief_calls;

  int hidden_positions = 0;
  for (const RoutePositionEvidence& position : input.route) {
    if (position.known_hideout < 0) ++hidden_positions;
  }
  SPIEL_CHECK_GT(hidden_positions, 0);
  SPIEL_CHECK_GT(completion.uniform_consistent_history_mass, 0.0L);

  const MarshalGuessPlan unrestricted = BuildMarshalGuesses(
      completion, hidden_positions, guess_mode, /*maximum_guess=*/41);
  if (unrestricted.top_argmax >= 30) {
    ++result->unrestricted_argmax_ge_30_turns;
  }
  MarshalGuessPlan plan = unrestricted;
  bool lifted = false;
  if (guard &&
      static_cast<int>(unrestricted.guesses.size()) != hidden_positions) {
    bool has_positive_low_card = false;
    for (int card = kMinCard; card <= 29; ++card) {
      if (completion.UniformConsistentHiddenHideoutProbability(card) > 0.0L) {
        has_positive_low_card = true;
        break;
      }
    }
    if (has_positive_low_card) {
      plan = BuildMarshalGuesses(completion, hidden_positions, guess_mode,
                                 /*maximum_guess=*/29);
      if (plan.guesses != unrestricted.guesses) {
        ++result->guard_restriction_turns;
      }
    } else if (guard_low_exhausted == GuardLowExhausted::kLift) {
      lifted = true;
      ++result->guard_lift_turns;
    } else {
      plan = MarshalGuessPlan{};
      plan.guesses = {kMinCard};
      ++result->guard_wait_turns;
    }
  }

  const bool banked_certain =
      guess_mode == MarshalGuessMode::kCertainOnly &&
      plan.certain_count > 0 && plan.certain_count < hidden_positions;
  const bool cover_all =
      static_cast<int>(plan.guesses.size()) == hidden_positions;
  if (plan.forced_gamble) ++result->forced_gamble_turns;
  if (banked_certain) {
    ++result->banked_certain_turns;
    result->banked_certain_cards += plan.certain_count;
  }
  if (cover_all) ++result->cover_all_attempts;

  for (Action action : plan.guesses) {
    SPIEL_CHECK_TRUE(Contains(state->LegalActions(), action));
    state->ApplyAction(action);
  }
  SPIEL_CHECK_TRUE(Contains(state->LegalActions(), kCommitAction));
  state->ApplyAction(kCommitAction);
  const FugitiveState& fugitive = AsFugitive(state);
  SPIEL_CHECK_FALSE(fugitive.guess_history().empty());
  const bool success = fugitive.guess_history().back().success;
  if (plan.forced_gamble && !success) {
    ++result->forced_gamble_losses;
    result->forced_gamble_certain_cards_lost += plan.certain_count;
  }
  if (cover_all && success) ++result->cover_all_wins;
  if (lifted) {
    if (success) {
      result->manhunt_disabled_by_guard_lift = true;
    }
  }
}

void ApplyManhuntGuess(State* state, std::uint64_t game_seed,
                       int particles, GameResult* result) {
  const std::string information_state =
      state->InformationStateString(kMarshalPlayer);
  const MarshalBeliefInput input = BuildMarshalBeliefInput(information_state);
  std::mt19937_64 rng(ManhuntSeed(game_seed, information_state));
  auto uniform = [&rng]() {
    return std::generate_canonical<double, 64>(rng);
  };

  const Clock::time_point evaluator_start = Clock::now();
  const MarshalSampledManhuntResult value = ComputeSampledManhuntValue(
      input, uniform,
      MarshalSampledManhuntOptions{/*particles=*/particles,
                                   /*max_solver_states=*/100000});
  result->manhunt_evaluator_time_us +=
      Microseconds(evaluator_start, Clock::now());
  ++result->manhunt_evaluator_calls;
  if (value.exact_for_empirical_belief) {
    ++result->manhunt_evaluator_exact_empirical;
  }

  SPIEL_CHECK_GE(value.best_guess, kMinCard);
  SPIEL_CHECK_LE(value.best_guess, 41);
  SPIEL_CHECK_TRUE(Contains(state->LegalActions(), value.best_guess));
  state->ApplyAction(value.best_guess);
  SPIEL_CHECK_TRUE(Contains(state->LegalActions(), kCommitAction));
  state->ApplyAction(kCommitAction);
}

GameResult RunGame(const Game& game, std::uint64_t seed, bool guard,
                   MarshalGuessMode guess_mode,
                   GuardLowExhausted guard_low_exhausted,
                   int manhunt_particles, int dead_card_sprints) {
  const Clock::time_point game_start = Clock::now();
  std::mt19937_64 chance_rng(seed);
  std::unique_ptr<State> state = game.NewInitialState();
  GameResult result;
  FugitivePolicyState fugitive_policy;

  while (!state->IsTerminal()) {
    FugitiveState& fugitive = AsFugitive(state.get());
    const Phase phase = fugitive.phase();
    if (state->CurrentPlayer() == kChancePlayerId) {
      state->ApplyAction(SampleChance(*state, &chance_rng));
    } else if (phase == Phase::kFugitiveDrawChoice ||
               phase == Phase::kFugitiveHideout ||
               phase == Phase::kFugitiveSprint) {
      ApplyFugitiveDecision(state.get(), dead_card_sprints,
                            &fugitive_policy, &result);
    } else if (phase == Phase::kMarshalDrawChoice) {
      const std::vector<Action> legal = state->LegalActions();
      SPIEL_CHECK_FALSE(legal.empty());
      state->ApplyAction(legal.front());
    } else if (phase == Phase::kMarshalGuess) {
      ApplyNormalMarshalGuess(state.get(), guard, guess_mode,
                              guard_low_exhausted, &result);
    } else {
      SPIEL_CHECK_EQ(phase, Phase::kManhuntGuess);
      result.entered_manhunt = true;
      ApplyManhuntGuess(state.get(), seed, manhunt_particles, &result);
    }
  }

  const FugitiveState& terminal = AsFugitive(state.get());
  result.winner = terminal.winner();
  result.terminal_reason = terminal.terminal_reason();
  result.rounds = terminal.round_number();
  result.reached_42 = std::any_of(
      terminal.route().begin(), terminal.route().end(),
      [](const RouteNode& node) { return node.hideout == 42; });
  result.elapsed_us = Microseconds(game_start, Clock::now());
  return result;
}

void AddResult(const GameResult& result, Summary* summary) {
  ++summary->games;
  switch (result.winner) {
    case Winner::kFugitive:
      ++summary->fugitive_wins;
      break;
    case Winner::kMarshal:
      ++summary->marshal_wins;
      break;
    case Winner::kDraw:
      ++summary->draws;
      break;
    case Winner::kNone:
      SpielFatalError("Completed game has no winner");
  }
  ++summary->terminal_reasons[result.terminal_reason];
  if (result.terminal_reason == "max_rounds") ++summary->timeouts;
  summary->round_sum += result.rounds;
  summary->pass_sum += result.passes;
  if (result.passes > 0) ++summary->games_with_pass;
  summary->sprint_card_sum += result.sprint_cards;
  if (result.sprint_cards > 0) ++summary->games_with_sprint;
  summary->forced_opening_sprint_card_sum +=
      result.forced_opening_sprint_cards;
  summary->dead_card_sprint_card_sum += result.dead_card_sprint_cards;
  summary->dead_card_sprint_play_sum += result.dead_card_sprint_plays;
  if (result.dead_card_sprint_cards > 0) {
    ++summary->games_with_dead_card_sprint;
  }
  if (result.reached_42) ++summary->reached_42;
  if (result.entered_manhunt) {
    ++summary->entered_manhunt;
    if (result.winner == Winner::kMarshal) {
      ++summary->marshal_wins_after_manhunt;
    }
  }
  summary->normal_belief_calls += result.normal_belief_calls;
  summary->normal_belief_time_us += result.normal_belief_time_us;
  summary->unrestricted_argmax_ge_30_turns +=
      result.unrestricted_argmax_ge_30_turns;
  summary->guard_restriction_turns += result.guard_restriction_turns;
  if (result.guard_restriction_turns > 0) {
    ++summary->games_with_guard_restriction;
  }
  summary->forced_gamble_turns += result.forced_gamble_turns;
  summary->forced_gamble_losses += result.forced_gamble_losses;
  summary->forced_gamble_certain_cards_lost +=
      result.forced_gamble_certain_cards_lost;
  summary->banked_certain_turns += result.banked_certain_turns;
  summary->banked_certain_cards += result.banked_certain_cards;
  summary->cover_all_attempts += result.cover_all_attempts;
  summary->cover_all_wins += result.cover_all_wins;
  summary->guard_lift_turns += result.guard_lift_turns;
  if (result.guard_lift_turns > 0) ++summary->games_with_guard_lift;
  if (result.manhunt_disabled_by_guard_lift) {
    ++summary->manhunt_disabled_by_guard_lift;
  }
  summary->guard_wait_turns += result.guard_wait_turns;
  if (result.guard_wait_turns > 0) ++summary->games_with_guard_wait;
  summary->manhunt_evaluator_calls += result.manhunt_evaluator_calls;
  summary->manhunt_evaluator_exact_empirical +=
      result.manhunt_evaluator_exact_empirical;
  summary->manhunt_evaluator_time_us += result.manhunt_evaluator_time_us;
  summary->elapsed_us += result.elapsed_us;
}

json SummaryJson(const Summary& summary) {
  json conditional_marshal_win_rate = nullptr;
  if (summary.entered_manhunt > 0) {
    conditional_marshal_win_rate =
        static_cast<double>(summary.marshal_wins_after_manhunt) /
        summary.entered_manhunt;
  }
  return {
      {"games", summary.games},
      {"wins",
       {{"fugitive", summary.fugitive_wins},
        {"marshal", summary.marshal_wins},
        {"draw", summary.draws}}},
      {"timeouts", summary.timeouts},
      {"terminal_reasons", summary.terminal_reasons},
      {"average_rounds",
       static_cast<double>(summary.round_sum) / summary.games},
      {"passes",
       {{"total", summary.pass_sum}, {"games", summary.games_with_pass}}},
      {"sprints",
       {{"cards", summary.sprint_card_sum},
        {"games", summary.games_with_sprint},
        {"forced_opening_cards",
         summary.forced_opening_sprint_card_sum},
        {"ordinary_dead_cards", summary.dead_card_sprint_card_sum},
        {"ordinary_dead_plays", summary.dead_card_sprint_play_sum},
        {"games_with_ordinary_dead_sprint",
         summary.games_with_dead_card_sprint}}},
      {"reached_42", summary.reached_42},
      {"manhunt",
       {{"games", summary.entered_manhunt},
        {"marshal_wins", summary.marshal_wins_after_manhunt},
        {"conditional_marshal_win_rate", conditional_marshal_win_rate},
        {"evaluator_calls", summary.manhunt_evaluator_calls},
        {"evaluator_time_us", summary.manhunt_evaluator_time_us},
        {"exact_empirical_calls",
         summary.manhunt_evaluator_exact_empirical}}},
      {"normal_belief",
       {{"calls", summary.normal_belief_calls},
        {"time_us", summary.normal_belief_time_us},
        {"unrestricted_argmax_ge_30_turns",
         summary.unrestricted_argmax_ge_30_turns}}},
      {"guard_restriction",
       {{"turns", summary.guard_restriction_turns},
        {"games", summary.games_with_guard_restriction}}},
      {"guess_mode_diagnostics",
       {{"forced_gamble",
         {{"turns", summary.forced_gamble_turns},
          {"losses", summary.forced_gamble_losses},
          {"certain_cards_lost",
           summary.forced_gamble_certain_cards_lost}}},
        {"banked_certain",
         {{"turns", summary.banked_certain_turns},
          {"cards", summary.banked_certain_cards}}},
        {"cover_all",
         {{"attempts", summary.cover_all_attempts},
          {"wins", summary.cover_all_wins}}}}},
      {"guard_fallback",
       {{"lift",
         {{"turns", summary.guard_lift_turns},
          {"games", summary.games_with_guard_lift},
          {"manhunt_disabled_games",
           summary.manhunt_disabled_by_guard_lift}}},
        {"wait",
         {{"turns", summary.guard_wait_turns},
          {"games", summary.games_with_guard_wait}}}}},
      {"elapsed_ms", summary.elapsed_us / 1000},
  };
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  bool parsed_guess_mode = false;
  bool parsed_legacy_guess_mode = false;
  for (int index = 1; index < argc; index += 2) {
    SPIEL_CHECK_LT(index + 1, argc);
    const std::string flag = argv[index];
    const std::string value = argv[index + 1];
    if (flag == "--games") {
      options.games = std::stoi(value);
    } else if (flag == "--seed_start") {
      options.seed_start = std::stoull(value);
    } else if (flag == "--max_rounds") {
      options.max_rounds = std::stoi(value);
    } else if (flag == "--manhunt_particles") {
      options.manhunt_particles = std::stoi(value);
    } else if (flag == "--dead_card_sprints") {
      options.dead_card_sprints = std::stoi(value);
    } else if (flag == "--guess_mode") {
      if (parsed_guess_mode || parsed_legacy_guess_mode) {
        SpielFatalError("Marshal guess mode specified more than once");
      }
      parsed_guess_mode = true;
      if (value == "argmax_only") {
        options.guess_mode = MarshalGuessMode::kArgmaxOnly;
      } else if (value == "certain_only") {
        options.guess_mode = MarshalGuessMode::kCertainOnly;
      } else if (value == "certain_plus_argmax") {
        options.guess_mode = MarshalGuessMode::kCertainPlusArgmax;
      } else {
        SpielFatalError("Unknown Marshal guess mode: " + value);
      }
    } else if (flag == "--add_certain_guesses") {
      if (parsed_guess_mode || parsed_legacy_guess_mode) {
        SpielFatalError("Marshal guess mode specified more than once");
      }
      parsed_legacy_guess_mode = true;
      if (value == "0") {
        options.guess_mode = MarshalGuessMode::kArgmaxOnly;
      } else if (value == "1") {
        options.guess_mode = MarshalGuessMode::kCertainPlusArgmax;
      } else {
        SpielFatalError("--add_certain_guesses expects 0 or 1");
      }
    } else if (flag == "--low_exhausted") {
      if (value == "lift") {
        options.guard_low_exhausted = GuardLowExhausted::kLift;
      } else if (value == "wait") {
        options.guard_low_exhausted = GuardLowExhausted::kWait;
      } else {
        SpielFatalError("Unknown guard low-exhausted mode: " + value);
      }
    } else {
      SpielFatalError("Unknown argument: " + flag);
    }
  }
  SPIEL_CHECK_GT(options.games, 0);
  SPIEL_CHECK_GE(options.max_rounds, 1);
  SPIEL_CHECK_LE(options.max_rounds, kMaximumMaxRounds);
  SPIEL_CHECK_GT(options.manhunt_particles, 0);
  SPIEL_CHECK_GE(options.dead_card_sprints, 0);
  SPIEL_CHECK_LE(options.dead_card_sprints, 2);
  return options;
}

int RunExperiment(const Options& options) {
  const auto game = LoadGame(
      "fugitive", {{"max_rounds", GameParameter(options.max_rounds)}});
  Summary guard;
  Summary noguard;
  std::map<std::string, int> paired_terminal_reasons;
  const Clock::time_point experiment_start = Clock::now();

  for (int offset = 0; offset < options.games; ++offset) {
    const std::uint64_t seed = options.seed_start + offset;
    const GameResult guard_result =
        RunGame(*game, seed, /*guard=*/true, options.guess_mode,
                options.guard_low_exhausted, options.manhunt_particles,
                options.dead_card_sprints);
    const GameResult noguard_result =
        RunGame(*game, seed, /*guard=*/false, options.guess_mode,
                options.guard_low_exhausted, options.manhunt_particles,
                options.dead_card_sprints);
    AddResult(guard_result, &guard);
    AddResult(noguard_result, &noguard);
    ++paired_terminal_reasons[guard_result.terminal_reason + "|" +
                              noguard_result.terminal_reason];
  }

  const json output = {
      {"type", "paired_summary"},
      {"experiment", "fugitive_l1_vs_marshal_l2"},
      {"games_per_policy", options.games},
      {"seed_start", options.seed_start},
      {"max_rounds", options.max_rounds},
      {"manhunt_particles", options.manhunt_particles},
      {"dead_card_sprints", options.dead_card_sprints},
      {"guess_mode", MarshalGuessModeString(options.guess_mode)},
      {"low_exhausted", GuardLowExhaustedString(options.guard_low_exhausted)},
      {"guard", SummaryJson(guard)},
      {"noguard", SummaryJson(noguard)},
      {"paired_terminal_reasons", paired_terminal_reasons},
      {"elapsed_ms", Microseconds(experiment_start, Clock::now()) / 1000},
  };
  std::cout << output.dump() << '\n';
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace fugitive
}  // namespace open_spiel

int main(int argc, char** argv) {
  return open_spiel::fugitive::RunExperiment(
      open_spiel::fugitive::ParseOptions(argc, argv));
}
