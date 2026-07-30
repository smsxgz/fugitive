// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef OPEN_SPIEL_GAMES_FUGITIVE_BELIEF_H_
#define OPEN_SPIEL_GAMES_FUGITIVE_BELIEF_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/fugitive/fugitive.h"

namespace open_spiel {
namespace fugitive {

struct RoutePositionEvidence {
  int known_hideout = -1;
  int sprint_count = 0;
  int known_sprint_value = -1;
  int play_round = 0;
  std::array<int, 3> known_sprint_draws = {0, 0, 0};
  std::vector<int> known_sprint_cards;

  bool operator==(const RoutePositionEvidence& other) const;
};

struct FailedGuessEvidence {
  int route_length = 0;
  std::vector<int> numbers;

  bool operator==(const FailedGuessEvidence& other) const;
};

// A deliberately player-local input. It contains no hidden Fugitive card
// identities. BuildMarshalBeliefInput only accepts a Marshal information-state
// string so the solver cannot inspect the full state.
struct MarshalBeliefInput {
  Phase phase = Phase::kMarshalGuess;
  std::vector<RoutePositionEvidence> route;
  std::array<std::vector<int>, 3> fugitive_draw_rounds;
  std::uint64_t unavailable_cards = 0;
  std::vector<FailedGuessEvidence> failed_guesses;

  bool operator==(const MarshalBeliefInput& other) const;
};

struct MarshalRouteSupportResult {
  std::uint64_t route_count = 0;
  // Exact counts under a uniform distribution over the Route support. These
  // are diagnostics, not Completion/history-weighted probabilities.
  std::array<std::uint64_t, kMaxCard + 1> hidden_card_route_count{};
  std::uint64_t memo_states = 0;
  // These include the counting pass and the subsequent marginal pass.
  std::uint64_t memo_hits = 0;
  std::uint64_t candidate_transitions = 0;

  double HiddenRouteSupportFraction(int card) const;
};

// Counts concrete hidden Fugitive draw sequences and hidden Sprint identities
// for every route in the Route support. It does not include a Fugitive policy,
// so the resulting mass is a uniform-consistent baseline, not a posterior.
struct MarshalCompletionResult {
  std::uint64_t route_support_count = 0;
  std::uint64_t completable_route_count = 0;
  long double uniform_consistent_history_mass = 0.0L;
  std::array<long double, kMaxCard + 1> hidden_card_history_mass{};

  std::uint64_t route_candidate_transitions = 0;
  std::uint64_t completion_memo_states = 0;
  std::uint64_t completion_memo_hits = 0;
  std::uint64_t completion_allocation_evaluations = 0;
  std::uint64_t max_completion_memo_states = 0;
  std::uint64_t completion_route_classes = 0;
  std::uint64_t completion_route_cache_hits = 0;

  long double UniformConsistentHiddenHideoutProbability(int card) const;
};

// One complete hidden history under the uniform-consistent baseline. The route
// excludes the public initial 0. Fugitive draw cards are ordered like the
// corresponding per-pile draw rounds in MarshalBeliefInput.
struct MarshalHistorySample {
  std::vector<int> route;
  std::vector<std::vector<int>> sprint_cards;
  std::array<std::vector<int>, 3> fugitive_draw_cards;
};

// Counts route assignments that satisfy all public route/guess constraints and
// the draw deadlines of Hideouts and already-revealed Sprint cards. Hidden
// Sprint identities are not assigned yet, so this is an upper bound on fully
// completable hidden histories rather than a posterior distribution.
MarshalBeliefInput BuildMarshalBeliefInput(
    const std::string& marshal_information_state);
MarshalRouteSupportResult ComputeMarshalRouteSupport(
    const MarshalBeliefInput& input);
MarshalCompletionResult ComputeMarshalCompletion(
    const MarshalBeliefInput& input);
MarshalHistorySample SampleMarshalHistory(const MarshalBeliefInput& input,
                                          std::function<double()> rng);

// Rebuilds a state from public information and a sampled hidden history. Every
// action is checked for legality, and the resulting Marshal information state
// must exactly equal marshal_information_state.
std::unique_ptr<State> ReplayMarshalHistory(
    const Game& game, const std::string& marshal_information_state,
    const MarshalHistorySample& sample);

}  // namespace fugitive
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_FUGITIVE_BELIEF_H_
