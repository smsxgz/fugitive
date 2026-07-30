// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "open_spiel/games/fugitive/belief.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace fugitive {
namespace {

std::unique_ptr<State> NewState() {
  return LoadGame("fugitive", {{"max_rounds", GameParameter(50)}})
      ->NewInitialState();
}

void PlayHideout(State* state, int hideout,
                 const std::vector<int>& sprints = {}) {
  state->ApplyAction(hideout);
  for (int sprint : sprints) state->ApplyAction(sprint);
  state->ApplyAction(kCommitAction);
}

void Draw(State* state, int pile, int card) {
  state->ApplyAction(kFirstPileAction + pile);
  state->ApplyAction(card);
}

std::unique_ptr<State> OpeningState(const std::vector<Action>& setup) {
  std::unique_ptr<State> state = NewState();
  for (Action card : setup) state->ApplyAction(card);
  PlayHideout(state.get(), 1);
  PlayHideout(state.get(), 2);
  Draw(state.get(), /*pile=*/0, /*card=*/7);
  Draw(state.get(), /*pile=*/1, /*card=*/17);
  return state;
}

MarshalBeliefInput TwoPositionInput() {
  MarshalBeliefInput input;
  input.route.resize(2);
  for (RoutePositionEvidence& position : input.route) {
    position.sprint_count = 0;
    position.play_round = 1;
  }
  input.fugitive_draw_rounds[0] = {0, 0, 0};
  input.fugitive_draw_rounds[1] = {0, 0};
  return input;
}

void TestOpeningCountAndMarginals() {
  std::unique_ptr<State> state = OpeningState({4, 5, 6, 15, 16});
  const MarshalBeliefInput input = BuildMarshalBeliefInput(
      state->InformationStateString(kMarshalPlayer));
  const MarshalRouteSupportResult result = ComputeMarshalRouteSupport(input);

  SPIEL_CHECK_EQ(result.route_count, 9);
  const std::vector<std::uint64_t> expected = {0, 3, 4, 5, 3, 2, 1};
  for (int card = 1; card <= 6; ++card) {
    SPIEL_CHECK_EQ(result.hidden_card_route_count[card], expected[card]);
  }
}

void TestFailedGuessAppliesOnlyToExistingPrefix() {
  MarshalBeliefInput input = TwoPositionInput();
  SPIEL_CHECK_EQ(ComputeMarshalRouteSupport(input).route_count, 9);

  input.failed_guesses = {{/*route_length=*/2, /*numbers=*/{2}}};
  SPIEL_CHECK_EQ(ComputeMarshalRouteSupport(input).route_count, 5);

  input.failed_guesses = {{/*route_length=*/2, /*numbers=*/{1, 2}}};
  SPIEL_CHECK_EQ(ComputeMarshalRouteSupport(input).route_count, 8);

  input.route.push_back(RoutePositionEvidence{
      /*known_hideout=*/-1, /*sprint_count=*/0,
      /*known_sprint_value=*/-1, /*play_round=*/2,
      /*known_sprint_draws=*/{0, 0, 0}});
  input.failed_guesses = {{/*route_length=*/2, /*numbers=*/{6}}};
  const MarshalRouteSupportResult later = ComputeMarshalRouteSupport(input);
  SPIEL_CHECK_GT(later.hidden_card_route_count[6], 0);
}

void TestDrawDeadlineIncludesRevealedSprint() {
  MarshalBeliefInput input;
  input.route.push_back(RoutePositionEvidence{
      /*known_hideout=*/4, /*sprint_count=*/1,
      /*known_sprint_value=*/1, /*play_round=*/1,
      /*known_sprint_draws=*/{1, 0, 0}});
  input.unavailable_cards = std::uint64_t{1} << 5;

  input.fugitive_draw_rounds[0] = {0, 2};
  SPIEL_CHECK_EQ(ComputeMarshalRouteSupport(input).route_count, 0);
  input.fugitive_draw_rounds[0] = {0, 1};
  SPIEL_CHECK_EQ(ComputeMarshalRouteSupport(input).route_count, 1);
}

void TestInputUsesOnlyMarshalInformation() {
  std::unique_ptr<State> first = OpeningState({4, 5, 6, 15, 16});
  std::unique_ptr<State> second = OpeningState({8, 9, 10, 20, 21});
  const std::string first_information =
      first->InformationStateString(kMarshalPlayer);
  const std::string second_information =
      second->InformationStateString(kMarshalPlayer);
  SPIEL_CHECK_EQ(first_information, second_information);

  const MarshalBeliefInput first_input =
      BuildMarshalBeliefInput(first_information);
  const MarshalBeliefInput second_input =
      BuildMarshalBeliefInput(second_information);
  SPIEL_CHECK_TRUE(first_input == second_input);
  SPIEL_CHECK_EQ(ComputeMarshalRouteSupport(first_input).route_count,
                 ComputeMarshalRouteSupport(second_input).route_count);
  const MarshalCompletionResult first_completion =
      ComputeMarshalCompletion(first_input);
  const MarshalCompletionResult second_completion =
      ComputeMarshalCompletion(second_input);
  SPIEL_CHECK_EQ(first_completion.completable_route_count,
                 second_completion.completable_route_count);
  SPIEL_CHECK_EQ(first_completion.uniform_consistent_history_mass,
                 second_completion.uniform_consistent_history_mass);
}

RoutePositionEvidence KnownHideout(int hideout, int round) {
  return RoutePositionEvidence{
      /*known_hideout=*/hideout, /*sprint_count=*/0,
      /*known_sprint_value=*/0, /*play_round=*/round,
      /*known_sprint_draws=*/{0, 0, 0}};
}

void TestCompletionCounting() {
  MarshalBeliefInput hand;
  hand.route = {KnownHideout(1, 1), KnownHideout(2, 1)};
  hand.fugitive_draw_rounds[0] = {0, 0, 0};
  hand.fugitive_draw_rounds[1] = {0, 0};
  hand.unavailable_cards = (std::uint64_t{1} << 4) |
                           (std::uint64_t{1} << 15);
  const MarshalCompletionResult hand_result =
      ComputeMarshalCompletion(hand);
  SPIEL_CHECK_EQ(hand_result.completable_route_count, 1);
  SPIEL_CHECK_EQ(hand_result.uniform_consistent_history_mass, 112320.0L);

  MarshalBeliefInput sprint;
  sprint.route.push_back(RoutePositionEvidence{
      /*known_hideout=*/5, /*sprint_count=*/1,
      /*known_sprint_value=*/-1, /*play_round=*/1,
      /*known_sprint_draws=*/{0, 0, 0}});
  sprint.fugitive_draw_rounds[0] = {0, 0};
  const MarshalCompletionResult sprint_result =
      ComputeMarshalCompletion(sprint);
  SPIEL_CHECK_EQ(sprint_result.route_support_count, 1);
  SPIEL_CHECK_EQ(sprint_result.completable_route_count, 1);
  SPIEL_CHECK_EQ(sprint_result.uniform_consistent_history_mass, 32.0L);

  for (int card = 2; card <= 14; card += 2) {
    sprint.unavailable_cards |= std::uint64_t{1} << card;
  }
  const MarshalCompletionResult impossible =
      ComputeMarshalCompletion(sprint);
  SPIEL_CHECK_EQ(impossible.route_support_count, 1);
  SPIEL_CHECK_EQ(impossible.completable_route_count, 0);
  SPIEL_CHECK_EQ(impossible.uniform_consistent_history_mass, 0.0L);

  MarshalBeliefInput deadline;
  deadline.route = {KnownHideout(1, 1), KnownHideout(4, 1),
                    KnownHideout(7, 2)};
  deadline.fugitive_draw_rounds[0] = {0, 0, 0, 2};
  deadline.fugitive_draw_rounds[1] = {0, 0};
  const MarshalCompletionResult deadline_result =
      ComputeMarshalCompletion(deadline);
  SPIEL_CHECK_EQ(deadline_result.completable_route_count, 1);
  SPIEL_CHECK_EQ(deadline_result.uniform_consistent_history_mass, 117936.0L);

  MarshalBeliefInput weighted = TwoPositionInput();
  weighted.fugitive_draw_rounds = {};
  weighted.fugitive_draw_rounds[0] = {0};
  const MarshalCompletionResult weighted_result =
      ComputeMarshalCompletion(weighted);
  SPIEL_CHECK_EQ(weighted_result.route_support_count, 9);
  SPIEL_CHECK_EQ(weighted_result.uniform_consistent_history_mass, 39.0L);
  const std::vector<long double> expected_mass = {0, 23, 24, 25, 3, 2, 1};
  for (int card = 1; card <= 6; ++card) {
    SPIEL_CHECK_EQ(weighted_result.hidden_card_history_mass[card],
                   expected_mass[card]);
  }
}

void TestSampledHistoriesReplay() {
  std::unique_ptr<State> state = NewState();
  for (Action card : {4, 5, 6, 15, 16}) state->ApplyAction(card);
  PlayHideout(state.get(), 1);
  PlayHideout(state.get(), 5, {2});
  Draw(state.get(), /*pile=*/0, /*card=*/7);
  Draw(state.get(), /*pile=*/1, /*card=*/17);

  state->ApplyAction(5);
  state->ApplyAction(kCommitAction);
  Draw(state.get(), /*pile=*/2, /*card=*/29);
  state->ApplyAction(kPassAction);
  Draw(state.get(), /*pile=*/0, /*card=*/8);
  state->ApplyAction(41);
  state->ApplyAction(kCommitAction);
  Draw(state.get(), /*pile=*/2, /*card=*/30);
  state->ApplyAction(kPassAction);
  Draw(state.get(), /*pile=*/1, /*card=*/18);

  const std::string target =
      state->InformationStateString(kMarshalPlayer);
  const MarshalBeliefInput input = BuildMarshalBeliefInput(target);
  std::mt19937_64 generator(20260730);
  auto rng = [&generator]() {
    return std::generate_canonical<double, 64>(generator);
  };
  for (int trial = 0; trial < 32; ++trial) {
    const MarshalHistorySample sample = SampleMarshalHistory(input, rng);
    std::unique_ptr<State> replay =
        ReplayMarshalHistory(*state->GetGame(), target, sample);
    SPIEL_CHECK_EQ(replay->InformationStateString(kMarshalPlayer), target);
  }

  state->ApplyAction(3);
  const std::string pending_target =
      state->InformationStateString(kMarshalPlayer);
  const MarshalHistorySample pending_sample =
      SampleMarshalHistory(BuildMarshalBeliefInput(pending_target), rng);
  ReplayMarshalHistory(*state->GetGame(), pending_target, pending_sample);
}

void TestSamplingMatchesWeightedMarginals() {
  MarshalBeliefInput input = TwoPositionInput();
  input.fugitive_draw_rounds = {};
  input.fugitive_draw_rounds[0] = {0};
  const MarshalCompletionResult expected = ComputeMarshalCompletion(input);

  constexpr int kTrials = 3900;
  std::array<int, kMaxCard + 1> observed{};
  std::mt19937_64 generator(17);
  auto rng = [&generator]() {
    return std::generate_canonical<double, 64>(generator);
  };
  for (int trial = 0; trial < kTrials; ++trial) {
    const MarshalHistorySample sample = SampleMarshalHistory(input, rng);
    for (int hideout : sample.route) ++observed[hideout];
  }

  for (int card = 1; card <= 6; ++card) {
    const long double probability =
        expected.UniformConsistentHiddenHideoutProbability(card);
    const long double mean = kTrials * probability;
    const long double deviation =
        std::sqrt(kTrials * probability * (1.0L - probability));
    SPIEL_CHECK_LE(std::abs(observed[card] - mean), 6.0L * deviation + 2.0L);
  }
}

}  // namespace
}  // namespace fugitive
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::fugitive::TestOpeningCountAndMarginals();
  open_spiel::fugitive::TestFailedGuessAppliesOnlyToExistingPrefix();
  open_spiel::fugitive::TestDrawDeadlineIncludesRevealedSprint();
  open_spiel::fugitive::TestInputUsesOnlyMarshalInformation();
  open_spiel::fugitive::TestCompletionCounting();
  open_spiel::fugitive::TestSampledHistoriesReplay();
  open_spiel::fugitive::TestSamplingMatchesWeightedMarginals();
}
