// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "open_spiel/games/fugitive/fugitive.h"

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace fugitive {
namespace {

using json = nlohmann::json;

const GameType kGameType{
    /*short_name=*/"fugitive",
    /*long_name=*/"Fugitive",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"max_rounds", GameParameter(kDefaultMaxRounds)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new FugitiveGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

const char* PublicPhaseString(Phase phase, int opening_plays_remaining) {
  switch (phase) {
    case Phase::kSetupChance:
      return "setup";
    case Phase::kFugitiveDrawChoice:
    case Phase::kFugitiveDrawChance:
      return "fugitive_draw";
    case Phase::kFugitiveHideout:
    case Phase::kFugitiveSprint:
      return opening_plays_remaining > 0 ? "fugitive_opening"
                                         : "fugitive_action";
    case Phase::kMarshalDrawChoice:
    case Phase::kMarshalDrawChance:
      return "marshal_draw";
    case Phase::kMarshalGuess:
      return "marshal_guess";
    case Phase::kManhuntGuess:
      return "manhunt";
    case Phase::kTerminal:
      return "terminal";
  }
  SpielFatalError("Unknown Fugitive phase");
}

const char* WinnerString(Winner winner) {
  switch (winner) {
    case Winner::kNone:
      return "none";
    case Winner::kFugitive:
      return "fugitive";
    case Winner::kMarshal:
      return "marshal";
    case Winner::kDraw:
      return "draw";
  }
  SpielFatalError("Unknown Fugitive winner");
}

template <typename T>
bool Contains(const std::vector<T>& values, const T& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

}  // namespace

std::ostream& operator<<(std::ostream& stream, Phase phase) {
  return stream << static_cast<int>(phase);
}

std::ostream& operator<<(std::ostream& stream, Winner winner) {
  return stream << static_cast<int>(winner);
}

FugitiveState::FugitiveState(std::shared_ptr<const Game> game, int max_rounds)
    : State(std::move(game)), max_rounds_(max_rounds) {
  remaining_cards_.fill(false);
  for (int card = 4; card <= 41; ++card) remaining_cards_[card] = true;
  hands_[kFugitivePlayer] = {1, 2, 3, 42};
  route_.push_back(RouteNode{/*hideout=*/0, /*sprint_cards=*/{},
                            /*revealed=*/true});
}

Player FugitiveState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (IsChancePhase()) return kChancePlayerId;
  switch (phase_) {
    case Phase::kFugitiveDrawChoice:
    case Phase::kFugitiveHideout:
    case Phase::kFugitiveSprint:
      return kFugitivePlayer;
    case Phase::kMarshalDrawChoice:
    case Phase::kMarshalGuess:
    case Phase::kManhuntGuess:
      return kMarshalPlayer;
    default:
      SpielFatalError("Phase has no current decision player");
  }
}

std::string FugitiveState::ActionToString(Player player, Action action) const {
  if (action == kPassAction) return "Pass";
  if (action == kCommitAction) return "Commit";
  if (action >= kFirstPileAction && action <= kLastPileAction) {
    return absl::StrCat("Draw pile ", action - kFirstPileAction);
  }
  if (action < kMinCard || action > kMaxCard) {
    return absl::StrCat("Invalid action ", action);
  }
  if (player == kChancePlayerId) return absl::StrCat("Deal card ", action);
  switch (phase_) {
    case Phase::kFugitiveHideout:
      return absl::StrCat("Hideout ", action);
    case Phase::kFugitiveSprint:
      return absl::StrCat("Sprint ", action);
    case Phase::kMarshalGuess:
    case Phase::kManhuntGuess:
      return absl::StrCat("Guess ", action);
    default:
      return absl::StrCat("Card ", action);
  }
}

std::string FugitiveState::ToString() const {
  json result;
  result["phase"] = PhaseString(phase_);
  result["round_number"] = round_number_;
  result["setup_draw_index"] = setup_draw_index_;
  result["opening_plays_remaining"] = opening_plays_remaining_;
  result["draws_remaining"] = draws_remaining_;
  result["pending_pile"] = pending_pile_;
  result["hands"] = hands_;
  result["route"] = json::array();
  for (const RouteNode& node : route_) {
    result["route"].push_back({{"hideout", node.hideout},
                               {"sprint_cards", node.sprint_cards},
                               {"revealed", node.revealed}});
  }
  result["remaining_cards"] = json::array();
  for (int card = kMinCard; card <= kMaxCard; ++card) {
    if (remaining_cards_[card]) result["remaining_cards"].push_back(card);
  }
  result["selected_hideout"] = selected_hideout_;
  result["selected_cards"] = selected_cards_;
  result["winner"] = WinnerString(winner_);
  result["terminal_reason"] = terminal_reason_;
  return result.dump();
}

std::vector<double> FugitiveState::Returns() const {
  if (!IsTerminal()) return {0.0, 0.0};
  switch (winner_) {
    case Winner::kFugitive:
      return {1.0, -1.0};
    case Winner::kMarshal:
      return {-1.0, 1.0};
    case Winner::kDraw:
      return {0.0, 0.0};
    case Winner::kNone:
      SpielFatalError("Terminal Fugitive state has no result");
  }
  return {};
}

std::string FugitiveState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ObservationJson(player, /*perfect_recall=*/true);
}

std::string FugitiveState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ObservationJson(player, /*perfect_recall=*/false);
}

std::unique_ptr<State> FugitiveState::Clone() const {
  return std::unique_ptr<State>(new FugitiveState(*this));
}

std::vector<Action> FugitiveState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChancePhase()) {
    std::vector<Action> actions;
    for (const auto& [action, probability] : ChanceOutcomes()) {
      actions.push_back(action);
    }
    return actions;
  }

  std::vector<Action> actions;
  switch (phase_) {
    case Phase::kFugitiveDrawChoice:
    case Phase::kMarshalDrawChoice:
      for (int pile : NonemptyPiles()) {
        actions.push_back(kFirstPileAction + pile);
      }
      break;

    case Phase::kFugitiveHideout: {
      if (opening_plays_remaining_ == 0) actions.push_back(kPassAction);
      const int previous = route_.back().hideout;
      for (int card : hands_[kFugitivePlayer]) {
        if (card <= previous) continue;
        if (CanCompleteFugitiveSelection(card, {}, /*last_sprint=*/0)) {
          actions.push_back(card);
        }
      }
      break;
    }

    case Phase::kFugitiveSprint: {
      const int last = selected_cards_.empty() ? 0 : selected_cards_.back();
      for (int card : hands_[kFugitivePlayer]) {
        if (!CanBeSprint(card) || card == selected_hideout_ || card <= last) {
          continue;
        }
        std::vector<int> next = selected_cards_;
        next.push_back(card);
        if (CanCompleteFugitiveSelection(selected_hideout_, next, card)) {
          actions.push_back(card);
        }
      }
      if (IsImmediateFugitiveCommitLegal()) actions.push_back(kCommitAction);
      break;
    }

    case Phase::kMarshalGuess:
    case Phase::kManhuntGuess: {
      if (phase_ == Phase::kManhuntGuess && !selected_cards_.empty()) {
        actions.push_back(kCommitAction);
        break;
      }
      const int last = selected_cards_.empty() ? 0 : selected_cards_.back();
      for (int card = last + 1; card <= 41; ++card) actions.push_back(card);
      if (!selected_cards_.empty()) actions.push_back(kCommitAction);
      break;
    }

    default:
      SpielFatalError("Unhandled decision phase in LegalActions");
  }
  return actions;
}

ActionsAndProbs FugitiveState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChancePhase());
  const int pile = phase_ == Phase::kSetupChance ? SetupPile() : pending_pile_;
  std::vector<int> cards;
  for (int card = kMinCard; card <= kMaxCard; ++card) {
    if (remaining_cards_[card] && PileForCard(card) == pile) {
      cards.push_back(card);
    }
  }
  SPIEL_CHECK_FALSE(cards.empty());
  const double probability = 1.0 / static_cast<double>(cards.size());
  ActionsAndProbs outcomes;
  outcomes.reserve(cards.size());
  for (int card : cards) outcomes.emplace_back(card, probability);
  return outcomes;
}

void FugitiveState::DoApplyAction(Action action) {
  const std::vector<Action> legal_actions = LegalActions();
  SPIEL_CHECK_TRUE(Contains(legal_actions, action));
  if (IsChancePhase()) {
    ApplyChanceCard(action);
  } else if (phase_ == Phase::kFugitiveDrawChoice ||
             phase_ == Phase::kMarshalDrawChoice) {
    ApplyDrawChoice(action);
  } else if (phase_ == Phase::kFugitiveHideout ||
             phase_ == Phase::kFugitiveSprint) {
    ApplyFugitiveAction(action);
  } else {
    ApplyGuessAction(action);
  }
}

bool FugitiveState::IsChancePhase() const {
  return phase_ == Phase::kSetupChance ||
         phase_ == Phase::kFugitiveDrawChance ||
         phase_ == Phase::kMarshalDrawChance;
}

int FugitiveState::SetupPile() const {
  SPIEL_CHECK_EQ(phase_, Phase::kSetupChance);
  SPIEL_CHECK_GE(setup_draw_index_, 0);
  SPIEL_CHECK_LT(setup_draw_index_, 5);
  return setup_draw_index_ < 3 ? 0 : 1;
}

std::array<int, 3> FugitiveState::PileSizes() const {
  std::array<int, 3> sizes = {0, 0, 0};
  for (int card = 4; card <= 41; ++card) {
    if (remaining_cards_[card]) ++sizes[PileForCard(card)];
  }
  return sizes;
}

std::vector<int> FugitiveState::NonemptyPiles() const {
  const std::array<int, 3> sizes = PileSizes();
  std::vector<int> result;
  for (int pile = 0; pile < 3; ++pile) {
    if (sizes[pile] > 0) result.push_back(pile);
  }
  return result;
}

bool FugitiveState::CardInHand(Player player, int card) const {
  return Contains(hands_[player], card);
}

int FugitiveState::SprintTotal(const std::vector<int>& cards) {
  int result = 0;
  for (int card : cards) result += SprintValue(card);
  return result;
}

bool FugitiveState::HasLegalHideoutPlay(const std::vector<int>& hand,
                                        int previous_hideout) {
  for (int hideout : hand) {
    if (hideout <= previous_hideout) continue;
    int max_distance = 3;
    for (int card : hand) {
      if (CanBeSprint(card) && card != hideout) {
        max_distance += SprintValue(card);
      }
    }
    if (hideout - previous_hideout <= max_distance) return true;
  }
  return false;
}

bool FugitiveState::IsImmediateFugitiveCommitLegal() const {
  SPIEL_CHECK_EQ(phase_, Phase::kFugitiveSprint);
  const int previous = route_.back().hideout;
  if (selected_hideout_ - previous > 3 + SprintTotal(selected_cards_)) {
    return false;
  }
  if (opening_plays_remaining_ != 2) return true;

  std::vector<int> remaining;
  for (int card : hands_[kFugitivePlayer]) {
    if (card != selected_hideout_ && !Contains(selected_cards_, card)) {
      remaining.push_back(card);
    }
  }
  return HasLegalHideoutPlay(remaining, selected_hideout_);
}

bool FugitiveState::CanCompleteFugitiveSelection(
    int hideout, const std::vector<int>& selected_sprints,
    int last_sprint) const {
  const int previous = route_.back().hideout;
  if (!CardInHand(kFugitivePlayer, hideout) || hideout <= previous) return false;

  std::vector<int> candidates;
  for (int card : hands_[kFugitivePlayer]) {
    if (CanBeSprint(card) && card != hideout && card > last_sprint &&
        !Contains(selected_sprints, card)) {
      candidates.push_back(card);
    }
  }

  if (opening_plays_remaining_ == 2) {
    std::vector<int> extension;
    return CanCompleteFirstOpeningSelection(
        hideout, selected_sprints, candidates, /*candidate_index=*/0,
        &extension);
  }

  int maximum_distance = 3 + SprintTotal(selected_sprints);
  for (int card : candidates) maximum_distance += SprintValue(card);
  return hideout - previous <= maximum_distance;
}

bool FugitiveState::CanCompleteFirstOpeningSelection(
    int hideout, const std::vector<int>& selected_sprints,
    const std::vector<int>& candidates, int candidate_index,
    std::vector<int>* extension) const {
  const int previous = route_.back().hideout;
  std::vector<int> payment = selected_sprints;
  payment.insert(payment.end(), extension->begin(), extension->end());
  if (hideout - previous <= 3 + SprintTotal(payment)) {
    std::vector<int> remaining;
    for (int card : hands_[kFugitivePlayer]) {
      if (card != hideout && !Contains(payment, card)) remaining.push_back(card);
    }
    if (HasLegalHideoutPlay(remaining, hideout)) return true;
  }

  if (candidate_index >= candidates.size()) return false;
  if (CanCompleteFirstOpeningSelection(hideout, selected_sprints, candidates,
                                       candidate_index + 1, extension)) {
    return true;
  }
  extension->push_back(candidates[candidate_index]);
  const bool found = CanCompleteFirstOpeningSelection(
      hideout, selected_sprints, candidates, candidate_index + 1, extension);
  extension->pop_back();
  return found;
}

void FugitiveState::ApplyChanceCard(int card) {
  SPIEL_CHECK_TRUE(remaining_cards_[card]);
  const bool setup = phase_ == Phase::kSetupChance;
  const int pile = setup ? SetupPile() : pending_pile_;
  SPIEL_CHECK_EQ(PileForCard(card), pile);
  remaining_cards_[card] = false;

  if (setup) {
    hands_[kFugitivePlayer].push_back(card);
    std::sort(hands_[kFugitivePlayer].begin(), hands_[kFugitivePlayer].end());
    draw_history_.push_back(
        DrawRecord{kFugitivePlayer, pile, card, /*round_number=*/0});
    ++setup_draw_index_;
    if (setup_draw_index_ == 5) phase_ = Phase::kFugitiveHideout;
    return;
  }

  const Player drawing_player = pending_draw_player_;
  SPIEL_CHECK_TRUE(drawing_player == kFugitivePlayer ||
                   drawing_player == kMarshalPlayer);
  hands_[drawing_player].push_back(card);
  std::sort(hands_[drawing_player].begin(), hands_[drawing_player].end());
  draw_history_.push_back(
      DrawRecord{drawing_player, pile, card, round_number_});
  pending_pile_ = -1;
  --draws_remaining_;

  if (draws_remaining_ > 0 && !NonemptyPiles().empty()) {
    phase_ = drawing_player == kFugitivePlayer
                 ? Phase::kFugitiveDrawChoice
                 : Phase::kMarshalDrawChoice;
    return;
  }

  draws_remaining_ = 0;
  pending_draw_player_ = kInvalidPlayer;
  phase_ = drawing_player == kFugitivePlayer ? Phase::kFugitiveHideout
                                              : Phase::kMarshalGuess;
  selected_cards_.clear();
}

void FugitiveState::ApplyDrawChoice(Action action) {
  pending_pile_ = action - kFirstPileAction;
  pending_draw_player_ = CurrentPlayer();
  phase_ = pending_draw_player_ == kFugitivePlayer
               ? Phase::kFugitiveDrawChance
               : Phase::kMarshalDrawChance;
}

void FugitiveState::ApplyFugitiveAction(Action action) {
  if (phase_ == Phase::kFugitiveHideout) {
    if (action == kPassAction) {
      play_history_.push_back(PlayRecord{/*hideout=*/0,
                                         /*sprint_cards=*/{},
                                         /*route_index=*/-1,
                                         round_number_});
      StartMarshalTurn(/*first_turn=*/false);
      return;
    }
    selected_hideout_ = action;
    selected_cards_.clear();
    phase_ = Phase::kFugitiveSprint;
    return;
  }

  SPIEL_CHECK_EQ(phase_, Phase::kFugitiveSprint);
  if (action == kCommitAction) {
    CommitFugitivePlay();
  } else {
    selected_cards_.push_back(action);
  }
}

void FugitiveState::ApplyGuessAction(Action action) {
  SPIEL_CHECK_TRUE(phase_ == Phase::kMarshalGuess ||
                   phase_ == Phase::kManhuntGuess);
  if (action == kCommitAction) {
    CommitGuess();
  } else {
    selected_cards_.push_back(action);
  }
}

void FugitiveState::CommitFugitivePlay() {
  SPIEL_CHECK_TRUE(IsImmediateFugitiveCommitLegal());
  const int hideout = selected_hideout_;
  for (int card : selected_cards_) {
    auto it = std::find(hands_[kFugitivePlayer].begin(),
                        hands_[kFugitivePlayer].end(), card);
    SPIEL_CHECK_TRUE(it != hands_[kFugitivePlayer].end());
    hands_[kFugitivePlayer].erase(it);
  }
  auto hideout_it = std::find(hands_[kFugitivePlayer].begin(),
                              hands_[kFugitivePlayer].end(), hideout);
  SPIEL_CHECK_TRUE(hideout_it != hands_[kFugitivePlayer].end());
  hands_[kFugitivePlayer].erase(hideout_it);

  const int route_index = route_.size();
  route_.push_back(RouteNode{hideout, selected_cards_, /*revealed=*/false});
  play_history_.push_back(
      PlayRecord{hideout, selected_cards_, route_index, round_number_});
  selected_hideout_ = -1;
  selected_cards_.clear();

  if (hideout == 42) {
    ResolveEscape();
    return;
  }
  if (opening_plays_remaining_ > 0) {
    --opening_plays_remaining_;
    if (opening_plays_remaining_ > 0) {
      phase_ = Phase::kFugitiveHideout;
    } else {
      StartMarshalTurn(/*first_turn=*/true);
    }
    return;
  }
  StartMarshalTurn(/*first_turn=*/false);
}

void FugitiveState::CommitGuess() {
  SPIEL_CHECK_FALSE(selected_cards_.empty());
  const bool manhunt = phase_ == Phase::kManhuntGuess;
  bool success = true;
  for (int number : selected_cards_) {
    const auto it = std::find_if(
        route_.begin() + 1, route_.end(), [number](const RouteNode& node) {
          return node.hideout == number && !node.revealed &&
                 node.hideout != 42;
        });
    if (it == route_.end()) {
      success = false;
      break;
    }
  }
  if (success) {
    for (int number : selected_cards_) {
      auto it = std::find_if(route_.begin() + 1, route_.end(),
                             [number](const RouteNode& node) {
                               return node.hideout == number;
                             });
      SPIEL_CHECK_TRUE(it != route_.end());
      it->revealed = true;
    }
  }

  guess_history_.push_back(GuessRecord{selected_cards_, success,
                                        static_cast<int>(route_.size()) - 1,
                                        round_number_, manhunt});
  selected_cards_.clear();

  if (manhunt) {
    if (!success) {
      SetTerminal(Winner::kFugitive, "manhunt_failed");
      return;
    }
    const bool any_hidden = std::any_of(
        route_.begin() + 1, route_.end(), [](const RouteNode& node) {
          return !node.revealed && node.hideout != 42;
        });
    if (!any_hidden) {
      SetTerminal(Winner::kMarshal, "manhunt_success");
    }
    return;
  }

  const bool all_revealed = route_.size() > 1 &&
                            std::all_of(route_.begin() + 1, route_.end(),
                                        [](const RouteNode& node) {
                                          return node.revealed;
                                        });
  if (success && all_revealed) {
    SetTerminal(Winner::kMarshal, "marshal_uncovered_route");
  } else {
    FinishRoundOrStartNext();
  }
}

void FugitiveState::StartMarshalTurn(bool first_turn) {
  phase_ = Phase::kMarshalDrawChoice;
  draws_remaining_ = first_turn ? 2 : 1;
  pending_draw_player_ = kMarshalPlayer;
  pending_pile_ = -1;
  SkipDrawIfDeckEmpty();
}

void FugitiveState::FinishRoundOrStartNext() {
  if (round_number_ >= max_rounds_) {
    SetTerminal(Winner::kDraw, "max_rounds");
    return;
  }
  ++round_number_;
  phase_ = Phase::kFugitiveDrawChoice;
  draws_remaining_ = 1;
  pending_draw_player_ = kFugitivePlayer;
  pending_pile_ = -1;
  SkipDrawIfDeckEmpty();
}

void FugitiveState::ResolveEscape() {
  int highest_revealed = 0;
  bool any_hidden = false;
  for (const RouteNode& node : route_) {
    if (node.hideout == 42) continue;
    if (node.revealed) highest_revealed = std::max(highest_revealed, node.hideout);
    if (node.hideout != 0 && !node.revealed) any_hidden = true;
  }
  if (highest_revealed >= 30) {
    SetTerminal(Winner::kFugitive, "escape_no_manhunt");
  } else if (!any_hidden) {
    SetTerminal(Winner::kMarshal, "manhunt_success");
  } else {
    phase_ = Phase::kManhuntGuess;
    selected_cards_.clear();
  }
}

void FugitiveState::SkipDrawIfDeckEmpty() {
  if (!NonemptyPiles().empty()) return;
  draws_remaining_ = 0;
  pending_draw_player_ = kInvalidPlayer;
  pending_pile_ = -1;
  if (phase_ == Phase::kFugitiveDrawChoice) {
    phase_ = Phase::kFugitiveHideout;
  } else if (phase_ == Phase::kMarshalDrawChoice) {
    phase_ = Phase::kMarshalGuess;
    selected_cards_.clear();
  }
}

void FugitiveState::SetTerminal(Winner winner, std::string reason) {
  winner_ = winner;
  terminal_reason_ = std::move(reason);
  phase_ = Phase::kTerminal;
  draws_remaining_ = 0;
  pending_pile_ = -1;
  pending_draw_player_ = kInvalidPlayer;
  selected_hideout_ = -1;
  selected_cards_.clear();
}

std::string FugitiveState::ObservationJson(Player player,
                                           bool perfect_recall) const {
  json observation;
  observation["role"] = RoleString(player);
  observation["hand"] = hands_[player];
  observation["pile_sizes"] = PileSizes();
  observation["route"] = json::array();
  for (int index = 0; index < route_.size(); ++index) {
    const RouteNode& node = route_[index];
    const bool hideout_visible = player == kFugitivePlayer || node.revealed ||
                                 node.hideout == 0 || node.hideout == 42;
    const bool sprint_visible =
        player == kFugitivePlayer || (node.revealed && node.hideout != 42);
    json route_node = {{"index", index},
                       {"sprint_count", node.sprint_cards.size()},
                       {"revealed", node.revealed}};
    route_node["hideout"] = hideout_visible ? json(node.hideout) : json(nullptr);
    route_node["sprint_cards"] =
        sprint_visible ? json(node.sprint_cards) : json(nullptr);
    observation["route"].push_back(std::move(route_node));
  }

  observation["guess_history"] = json::array();
  for (const GuessRecord& record : guess_history_) {
    observation["guess_history"].push_back(
        {{"numbers", record.numbers},
         {"success", record.success},
         {"route_length", record.route_length},
         {"round_number", record.round_number},
         {"manhunt", record.manhunt}});
  }

  observation["draw_history"] = json::array();
  for (const DrawRecord& record : draw_history_) {
    json draw = {{"role", RoleString(record.player)},
                 {"pile", record.pile},
                 {"round_number", record.round_number}};
    draw["card"] = record.player == player ? json(record.card) : json(nullptr);
    observation["draw_history"].push_back(std::move(draw));
  }

  observation["round_number"] = round_number_;
  observation["phase"] = PublicPhaseString(phase_, opening_plays_remaining_);
  observation["action_stage"] = PhaseString(phase_);
  observation["legal_draw_piles"] = json::array();
  if (CurrentPlayer() == player &&
      (phase_ == Phase::kFugitiveDrawChoice ||
       phase_ == Phase::kMarshalDrawChoice)) {
    observation["legal_draw_piles"] = NonemptyPiles();
  }

  observation["play_history"] = json::array();
  for (const PlayRecord& record : play_history_) {
    if (record.route_index < 0) {
      observation["play_history"].push_back(
          {{"hideout", nullptr},
           {"sprint_count", 0},
           {"sprint_cards", json::array()},
           {"route_index", nullptr},
           {"round_number", record.round_number},
           {"passed", true}});
      continue;
    }
    const RouteNode& node = route_[record.route_index];
    const bool hideout_visible = player == kFugitivePlayer || node.revealed ||
                                 node.hideout == 42;
    const bool sprint_visible =
        player == kFugitivePlayer || (node.revealed && node.hideout != 42);
    json play = {{"sprint_count", record.sprint_cards.size()},
                 {"route_index", record.route_index},
                 {"round_number", record.round_number},
                 {"passed", false}};
    play["hideout"] = hideout_visible ? json(record.hideout) : json(nullptr);
    play["sprint_cards"] =
        sprint_visible ? json(record.sprint_cards) : json(nullptr);
    observation["play_history"].push_back(std::move(play));
  }

  observation["pending_draw"] = nullptr;
  if (phase_ == Phase::kFugitiveDrawChance ||
      phase_ == Phase::kMarshalDrawChance) {
    observation["pending_draw"] =
        {{"role", RoleString(pending_draw_player_)}, {"pile", pending_pile_}};
  }

  observation["pending_action"] = nullptr;
  observation["pending_action_count"] = 0;
  if (phase_ == Phase::kFugitiveSprint && player == kFugitivePlayer) {
    observation["pending_action"] =
        {{"kind", "fugitive_play"},
         {"hideout", selected_hideout_},
         {"sprint_cards", selected_cards_}};
  }
  if (phase_ == Phase::kFugitiveSprint) {
    observation["pending_action_count"] = 1 + selected_cards_.size();
  } else if (phase_ == Phase::kMarshalGuess ||
             phase_ == Phase::kManhuntGuess) {
    observation["pending_action_count"] = selected_cards_.size();
  }
  if ((phase_ == Phase::kMarshalGuess ||
       phase_ == Phase::kManhuntGuess) &&
      player == kMarshalPlayer) {
    observation["pending_action"] =
        {{"kind", phase_ == Phase::kManhuntGuess ? "manhunt" : "guess"},
         {"numbers", selected_cards_}};
  }

  observation["terminal"] = nullptr;
  if (IsTerminal()) {
    observation["terminal"] = {{"winner", WinnerString(winner_)},
                                {"reason", terminal_reason_}};
  }

  json result = {
      {"schema", perfect_recall ? "fugitive.information_state"
                                : "fugitive.observation"},
      {"schema_version", 1},
      {"observation", std::move(observation)}};
  if (perfect_recall) result["perfect_recall"] = true;
  return result.dump();
}

const char* FugitiveState::PhaseString(Phase phase) {
  switch (phase) {
    case Phase::kSetupChance:
      return "setup_chance";
    case Phase::kFugitiveDrawChoice:
      return "fugitive_draw_choice";
    case Phase::kFugitiveDrawChance:
      return "fugitive_draw_chance";
    case Phase::kFugitiveHideout:
      return "fugitive_hideout";
    case Phase::kFugitiveSprint:
      return "fugitive_sprint";
    case Phase::kMarshalDrawChoice:
      return "marshal_draw_choice";
    case Phase::kMarshalDrawChance:
      return "marshal_draw_chance";
    case Phase::kMarshalGuess:
      return "marshal_guess";
    case Phase::kManhuntGuess:
      return "manhunt_guess";
    case Phase::kTerminal:
      return "terminal";
  }
  SpielFatalError("Unknown Fugitive phase");
}

const char* FugitiveState::RoleString(Player player) {
  if (player == kFugitivePlayer) return "fugitive";
  if (player == kMarshalPlayer) return "marshal";
  if (player == kChancePlayerId) return "chance";
  return "none";
}

FugitiveGame::FugitiveGame(const GameParameters& params)
    : Game(kGameType, params),
      max_rounds_(ParameterValue<int>("max_rounds")) {
  SPIEL_CHECK_GE(max_rounds_, 1);
  SPIEL_CHECK_LE(max_rounds_, kMaximumMaxRounds);
}

std::unique_ptr<State> FugitiveGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new FugitiveState(shared_from_this(), max_rounds_));
}

}  // namespace fugitive
}  // namespace open_spiel
