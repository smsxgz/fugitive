// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "open_spiel/games/fugitive/fugitive.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace fugitive {
namespace {

using json = nlohmann::json;
namespace testing = open_spiel::testing;

FugitiveState& AsFugitive(State* state) {
  return open_spiel::down_cast<FugitiveState&>(*state);
}

const FugitiveState& AsFugitive(const State* state) {
  return open_spiel::down_cast<const FugitiveState&>(*state);
}

std::unique_ptr<State> NewState(int max_rounds = kDefaultMaxRounds) {
  return LoadGame("fugitive", {{"max_rounds", GameParameter(max_rounds)}})
      ->NewInitialState();
}

void ApplyDeterministicSetup(State* state) {
  for (Action card : {4, 5, 6, 15, 16}) state->ApplyAction(card);
  SPIEL_CHECK_EQ(AsFugitive(state).phase(), Phase::kFugitiveHideout);
}

void PlayHideout(State* state, int hideout,
                 const std::vector<int>& sprints = {}) {
  state->ApplyAction(hideout);
  for (int sprint : sprints) state->ApplyAction(sprint);
  state->ApplyAction(kCommitAction);
}

void Draw(State* state, int pile, int card) {
  state->ApplyAction(kFirstPileAction + pile);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(card);
}

void CommitGuess(State* state, const std::vector<int>& numbers) {
  for (int number : numbers) state->ApplyAction(number);
  state->ApplyAction(kCommitAction);
}

void FinishOpening(State* state) {
  PlayHideout(state, 1);
  PlayHideout(state, 2);
  SPIEL_CHECK_EQ(AsFugitive(state).phase(), Phase::kMarshalDrawChoice);
}

void FinishFirstMarshalDraws(State* state) {
  Draw(state, /*pile=*/0, /*card=*/7);
  Draw(state, /*pile=*/1, /*card=*/17);
  SPIEL_CHECK_EQ(AsFugitive(state).phase(), Phase::kMarshalGuess);
}

void DrawFirstAvailableCard(State* state) {
  const std::vector<Action> piles = state->LegalActions();
  SPIEL_CHECK_FALSE(piles.empty());
  SPIEL_CHECK_GE(piles.front(), kFirstPileAction);
  state->ApplyAction(piles.front());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction(state->LegalActions().front());
}

std::unique_ptr<State> NewEscapeReadyState() {
  std::unique_ptr<State> state = NewState();
  for (Action card : {4, 6, 8, 16, 18}) state->ApplyAction(card);
  FinishOpening(state.get());

  Draw(state.get(), /*pile=*/0, /*card=*/5);
  Draw(state.get(), /*pile=*/0, /*card=*/7);
  CommitGuess(state.get(), {5});

  auto pass_after_draw = [&state](int fugitive_pile, int fugitive_card,
                                  int marshal_pile, int marshal_card) {
    Draw(state.get(), fugitive_pile, fugitive_card);
    state->ApplyAction(kPassAction);
    Draw(state.get(), marshal_pile, marshal_card);
    CommitGuess(state.get(), {5});
  };
  auto play_after_draw = [&state](int fugitive_pile, int fugitive_card,
                                  int hideout,
                                  const std::vector<int>& sprints,
                                  int marshal_pile, int marshal_card) {
    Draw(state.get(), fugitive_pile, fugitive_card);
    PlayHideout(state.get(), hideout, sprints);
    Draw(state.get(), marshal_pile, marshal_card);
    CommitGuess(state.get(), {5});
  };

  play_after_draw(/*fugitive_pile=*/1, /*fugitive_card=*/20,
                  /*hideout=*/8, /*sprints=*/{3, 4},
                  /*marshal_pile=*/0, /*marshal_card=*/9);
  play_after_draw(/*fugitive_pile=*/1, /*fugitive_card=*/22,
                  /*hideout=*/16, /*sprints=*/{6, 18, 20},
                  /*marshal_pile=*/0, /*marshal_card=*/10);
  pass_after_draw(/*fugitive_pile=*/1, /*fugitive_card=*/24,
                  /*marshal_pile=*/0, /*marshal_card=*/11);
  play_after_draw(/*fugitive_pile=*/1, /*fugitive_card=*/26,
                  /*hideout=*/22, /*sprints=*/{24, 26},
                  /*marshal_pile=*/0, /*marshal_card=*/12);
  pass_after_draw(/*fugitive_pile=*/1, /*fugitive_card=*/28,
                  /*marshal_pile=*/0, /*marshal_card=*/13);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/30,
                  /*marshal_pile=*/0, /*marshal_card=*/14);
  play_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/32,
                  /*hideout=*/28, /*sprints=*/{30, 32},
                  /*marshal_pile=*/1, /*marshal_card=*/15);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/34,
                  /*marshal_pile=*/1, /*marshal_card=*/17);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/36,
                  /*marshal_pile=*/1, /*marshal_card=*/19);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/38,
                  /*marshal_pile=*/1, /*marshal_card=*/21);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/40,
                  /*marshal_pile=*/1, /*marshal_card=*/23);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/29,
                  /*marshal_pile=*/1, /*marshal_card=*/25);
  pass_after_draw(/*fugitive_pile=*/2, /*fugitive_card=*/31,
                  /*marshal_pile=*/1, /*marshal_card=*/27);
  return state;
}

void BasicTests() {
  testing::LoadGameTest("fugitive");
  testing::ChanceOutcomesTest(*LoadGame("fugitive"));
  testing::RandomSimTest(
      *LoadGame("fugitive", {{"max_rounds", GameParameter(5)}}),
      /*num_sims=*/10, /*serialize=*/true, /*verbose=*/false);
}

void TestSetupAndExplicitChance() {
  std::unique_ptr<State> state = NewState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  const ActionsAndProbs outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_EQ(outcomes.size(), 11);
  SPIEL_CHECK_EQ(outcomes.front().first, 4);
  SPIEL_CHECK_EQ(outcomes.back().first, 14);

  ApplyDeterministicSetup(state.get());
  const FugitiveState& fugitive = AsFugitive(state.get());
  SPIEL_CHECK_EQ(fugitive.hand(kFugitivePlayer).size(), 9);
  SPIEL_CHECK_TRUE(
      std::all_of(fugitive.draw_history().begin(),
                  fugitive.draw_history().end(),
                  [](const DrawRecord& record) {
                    return record.player == kFugitivePlayer &&
                           record.round_number == 0;
                  }));

  const json observation = json::parse(state->ObservationString(0));
  SPIEL_CHECK_EQ(observation["schema"], "fugitive.observation");
  SPIEL_CHECK_EQ(observation["observation"]["pile_sizes"],
                 json({8, 12, 13}));
  SPIEL_CHECK_EQ(observation["observation"]["hand"].size(), 9);
}

void TestCompositeActionsAndHiddenInformation() {
  std::unique_ptr<State> state = NewState();
  ApplyDeterministicSetup(state.get());

  state->ApplyAction(4);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kFugitiveSprint);
  const json marshal_after_hideout =
      json::parse(state->ObservationString(kMarshalPlayer));
  const std::string marshal_info_after_hideout =
      state->InformationStateString(kMarshalPlayer);
  SPIEL_CHECK_EQ(
      marshal_after_hideout["observation"]["pending_action_count"], 1);
  SPIEL_CHECK_TRUE(
      marshal_after_hideout["observation"]["pending_action"].is_null());
  const std::vector<Action> first_sprints = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(first_sprints.begin(), first_sprints.end(),
                             Action{3}) != first_sprints.end());
  SPIEL_CHECK_TRUE(std::find(first_sprints.begin(), first_sprints.end(),
                             Action{42}) == first_sprints.end());
  state->ApplyAction(3);
  const json marshal_after_sprint =
      json::parse(state->ObservationString(kMarshalPlayer));
  SPIEL_CHECK_EQ(
      marshal_after_sprint["observation"]["pending_action_count"], 2);
  SPIEL_CHECK_NE(state->InformationStateString(kMarshalPlayer),
                 marshal_info_after_hideout);
  const std::vector<Action> after_three = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(after_three.begin(), after_three.end(), Action{1}) ==
                   after_three.end());
  SPIEL_CHECK_TRUE(std::find(after_three.begin(), after_three.end(), Action{2}) ==
                   after_three.end());
  SPIEL_CHECK_TRUE(std::find(after_three.begin(), after_three.end(),
                             kCommitAction) != after_three.end());
  state->ApplyAction(kCommitAction);
  PlayHideout(state.get(), 5);

  const json fugitive = json::parse(state->ObservationString(kFugitivePlayer));
  const json marshal = json::parse(state->ObservationString(kMarshalPlayer));
  SPIEL_CHECK_EQ(fugitive["observation"]["route"][1]["hideout"], 4);
  SPIEL_CHECK_EQ(fugitive["observation"]["route"][1]["sprint_cards"],
                 json({3}));
  SPIEL_CHECK_TRUE(marshal["observation"]["route"][1]["hideout"].is_null());
  SPIEL_CHECK_TRUE(
      marshal["observation"]["route"][1]["sprint_cards"].is_null());
  SPIEL_CHECK_EQ(marshal["observation"]["route"][1]["sprint_count"], 1);

  const json info = json::parse(state->InformationStateString(kMarshalPlayer));
  SPIEL_CHECK_EQ(info["schema"], "fugitive.information_state");
  SPIEL_CHECK_TRUE(info["perfect_recall"]);
}

void TestOpeningFiltersSelectionsWithoutASecondPlay() {
  std::unique_ptr<State> state = NewState();
  ApplyDeterministicSetup(state.get());

  const std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), Action{1}) !=
                   legal.end());
  // Hideout 15 can be paid for only by consuming every remaining higher-card
  // continuation. It is locally affordable but invalid as the first opening.
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), Action{15}) ==
                   legal.end());
}

void TestMarshalInfostateMergesFugitivePrivateDeals() {
  std::unique_ptr<State> first = NewState();
  std::unique_ptr<State> second = NewState();
  for (Action card : {4, 5, 6, 15, 16}) first->ApplyAction(card);
  for (Action card : {8, 9, 10, 20, 21}) second->ApplyAction(card);
  FinishOpening(first.get());
  FinishOpening(second.get());
  Draw(first.get(), /*pile=*/0, /*card=*/7);
  Draw(second.get(), /*pile=*/0, /*card=*/7);
  Draw(first.get(), /*pile=*/1, /*card=*/17);
  Draw(second.get(), /*pile=*/1, /*card=*/17);

  SPIEL_CHECK_EQ(first->CurrentPlayer(), kMarshalPlayer);
  SPIEL_CHECK_EQ(second->CurrentPlayer(), kMarshalPlayer);
  SPIEL_CHECK_EQ(first->InformationStateString(kMarshalPlayer),
                 second->InformationStateString(kMarshalPlayer));
  SPIEL_CHECK_EQ(first->ObservationString(kMarshalPlayer),
                 second->ObservationString(kMarshalPlayer));
  SPIEL_CHECK_EQ(first->LegalActions(), second->LegalActions());
}

void TestAllOrNothingGuessAndReveal() {
  std::unique_ptr<State> failed = NewState(/*max_rounds=*/1);
  ApplyDeterministicSetup(failed.get());
  FinishOpening(failed.get());
  FinishFirstMarshalDraws(failed.get());
  CommitGuess(failed.get(), {1, 3});
  SPIEL_CHECK_TRUE(failed->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(failed.get()).winner(), Winner::kDraw);
  SPIEL_CHECK_FALSE(AsFugitive(failed.get()).route()[1].revealed);
  SPIEL_CHECK_FALSE(AsFugitive(failed.get()).route()[2].revealed);

  std::unique_ptr<State> success = NewState();
  ApplyDeterministicSetup(success.get());
  FinishOpening(success.get());
  FinishFirstMarshalDraws(success.get());
  CommitGuess(success.get(), {1, 2});
  SPIEL_CHECK_TRUE(success->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(success.get()).winner(), Winner::kMarshal);
  SPIEL_CHECK_EQ(AsFugitive(success.get()).terminal_reason(),
                 "marshal_uncovered_route");
  SPIEL_CHECK_EQ(success->Returns(), std::vector<double>({-1.0, 1.0}));
}

void TestMaxRoundsIsDraw() {
  std::unique_ptr<State> state = NewState(/*max_rounds=*/1);
  ApplyDeterministicSetup(state.get());
  FinishOpening(state.get());
  FinishFirstMarshalDraws(state.get());
  CommitGuess(state.get(), {41});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(state.get()).winner(), Winner::kDraw);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).terminal_reason(), "max_rounds");
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({0.0, 0.0}));
}

void TestEmptyDeckSkipsBothDraws() {
  std::unique_ptr<State> state = NewState(/*max_rounds=*/18);
  ApplyDeterministicSetup(state.get());
  FinishOpening(state.get());
  FinishFirstMarshalDraws(state.get());
  CommitGuess(state.get(), {41});

  for (int round = 2; round <= 17; ++round) {
    SPIEL_CHECK_EQ(AsFugitive(state.get()).round_number(), round);
    SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(),
                   Phase::kFugitiveDrawChoice);
    DrawFirstAvailableCard(state.get());
    SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kFugitiveHideout);
    state->ApplyAction(kPassAction);
    if (round < 17) {
      SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(),
                     Phase::kMarshalDrawChoice);
      DrawFirstAvailableCard(state.get());
    }
    SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kMarshalGuess);
    CommitGuess(state.get(), {41});
  }

  SPIEL_CHECK_EQ(AsFugitive(state.get()).round_number(), 18);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kFugitiveHideout);
  const json observation = json::parse(state->ObservationString(0));
  SPIEL_CHECK_EQ(observation["observation"]["pile_sizes"], json({0, 0, 0}));
  state->ApplyAction(kPassAction);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kMarshalGuess);
  CommitGuess(state.get(), {41});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(state.get()).terminal_reason(), "max_rounds");
}

void TestEscapeAndManhuntVisibility() {
  std::unique_ptr<State> state = NewEscapeReadyState();
  Draw(state.get(), /*pile=*/2, /*card=*/33);
  PlayHideout(state.get(), /*hideout=*/42,
              /*sprints=*/{29, 31, 33, 34, 36, 38, 40});

  SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kManhuntGuess);
  const json marshal = json::parse(state->ObservationString(kMarshalPlayer));
  const json& escape = marshal["observation"]["route"].back();
  SPIEL_CHECK_EQ(escape["hideout"], 42);
  SPIEL_CHECK_FALSE(escape["revealed"]);
  SPIEL_CHECK_TRUE(escape["sprint_cards"].is_null());

  const int certain_miss = AsFugitive(state.get()).hand(kMarshalPlayer).front();
  CommitGuess(state.get(), {certain_miss});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(state.get()).winner(), Winner::kFugitive);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).terminal_reason(), "manhunt_failed");
}

void TestManhuntCorrectGuessesContinueAndWin() {
  std::unique_ptr<State> state = NewEscapeReadyState();
  Draw(state.get(), /*pile=*/2, /*card=*/33);
  PlayHideout(state.get(), /*hideout=*/42,
              /*sprints=*/{29, 31, 33, 34, 36, 38, 40});

  for (int hideout : {1, 2, 8, 16, 22}) {
    CommitGuess(state.get(), {hideout});
    SPIEL_CHECK_EQ(AsFugitive(state.get()).phase(), Phase::kManhuntGuess);
  }
  CommitGuess(state.get(), {28});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(state.get()).winner(), Winner::kMarshal);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).terminal_reason(), "manhunt_success");
}

void TestRevealedHighHideoutSkipsManhunt() {
  std::unique_ptr<State> state = NewEscapeReadyState();
  Draw(state.get(), /*pile=*/2, /*card=*/33);
  PlayHideout(state.get(), /*hideout=*/34, /*sprints=*/{29, 31, 33});
  Draw(state.get(), /*pile=*/2, /*card=*/35);
  CommitGuess(state.get(), {34});
  SPIEL_CHECK_TRUE(AsFugitive(state.get()).route().back().revealed);

  Draw(state.get(), /*pile=*/2, /*card=*/37);
  PlayHideout(state.get(), /*hideout=*/42, /*sprints=*/{36, 37, 38});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsFugitive(state.get()).winner(), Winner::kFugitive);
  SPIEL_CHECK_EQ(AsFugitive(state.get()).terminal_reason(),
                 "escape_no_manhunt");
}

}  // namespace
}  // namespace fugitive
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::fugitive::BasicTests();
  open_spiel::fugitive::TestSetupAndExplicitChance();
  open_spiel::fugitive::TestCompositeActionsAndHiddenInformation();
  open_spiel::fugitive::TestOpeningFiltersSelectionsWithoutASecondPlay();
  open_spiel::fugitive::TestMarshalInfostateMergesFugitivePrivateDeals();
  open_spiel::fugitive::TestAllOrNothingGuessAndReveal();
  open_spiel::fugitive::TestMaxRoundsIsDraw();
  open_spiel::fugitive::TestEmptyDeckSkipsBothDraws();
  open_spiel::fugitive::TestEscapeAndManhuntVisibility();
  open_spiel::fugitive::TestManhuntCorrectGuessesContinueAndWin();
  open_spiel::fugitive::TestRevealedHighHideoutSkipsManhunt();
}
