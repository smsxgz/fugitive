// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef OPEN_SPIEL_GAMES_FUGITIVE_FUGITIVE_H_
#define OPEN_SPIEL_GAMES_FUGITIVE_FUGITIVE_H_

#include <array>
#include <iosfwd>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace fugitive {

inline constexpr int kNumPlayers = 2;
inline constexpr Player kFugitivePlayer = 0;
inline constexpr Player kMarshalPlayer = 1;

inline constexpr int kMinCard = 1;
inline constexpr int kMaxCard = 42;
inline constexpr Action kPassAction = 0;
inline constexpr Action kFirstPileAction = 43;
inline constexpr Action kLastPileAction = 45;
inline constexpr Action kCommitAction = 46;
inline constexpr int kNumDistinctActions = 47;
inline constexpr int kDefaultMaxRounds = 50;
inline constexpr int kMaximumMaxRounds =
    (std::numeric_limits<int>::max() - 100) / 100;

inline int PileForCard(int card) {
  if (card >= 4 && card <= 14) return 0;
  if (card >= 15 && card <= 28) return 1;
  if (card >= 29 && card <= 41) return 2;
  return -1;
}

inline bool CanBeSprint(int card) {
  return card >= kMinCard && card < kMaxCard;
}

inline int SprintValue(int card) {
  SPIEL_CHECK_TRUE(CanBeSprint(card));
  return card % 2 == 0 ? 2 : 1;
}

enum class Phase {
  kSetupChance,
  kFugitiveDrawChoice,
  kFugitiveDrawChance,
  kFugitiveHideout,
  kFugitiveSprint,
  kMarshalDrawChoice,
  kMarshalDrawChance,
  kMarshalGuess,
  kManhuntGuess,
  kTerminal,
};

enum class Winner { kNone, kFugitive, kMarshal, kDraw };

std::ostream& operator<<(std::ostream& stream, Phase phase);
std::ostream& operator<<(std::ostream& stream, Winner winner);

struct RouteNode {
  int hideout;
  std::vector<int> sprint_cards;
  bool revealed;
};

struct DrawRecord {
  Player player;
  int pile;
  int card;
  int round_number;
};

struct PlayRecord {
  int hideout;  // 0 denotes Pass.
  std::vector<int> sprint_cards;
  int route_index;  // -1 denotes Pass.
  int round_number;
};

struct GuessRecord {
  std::vector<int> numbers;
  bool success;
  int route_length;
  int round_number;
  bool manhunt;
};

class FugitiveGame;

class FugitiveState : public State {
 public:
  FugitiveState(std::shared_ptr<const Game> game, int max_rounds);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kTerminal; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;

  Phase phase() const { return phase_; }
  int round_number() const { return round_number_; }
  int opening_plays_remaining() const { return opening_plays_remaining_; }
  Winner winner() const { return winner_; }
  const std::string& terminal_reason() const { return terminal_reason_; }
  const std::vector<int>& hand(Player player) const { return hands_.at(player); }
  const std::vector<RouteNode>& route() const { return route_; }
  const std::vector<DrawRecord>& draw_history() const { return draw_history_; }
  const std::vector<PlayRecord>& play_history() const { return play_history_; }
  const std::vector<GuessRecord>& guess_history() const {
    return guess_history_;
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool IsChancePhase() const;
  int SetupPile() const;
  std::array<int, 3> PileSizes() const;
  std::vector<int> NonemptyPiles() const;
  bool CardInHand(Player player, int card) const;
  static int SprintTotal(const std::vector<int>& cards);
  static bool HasLegalHideoutPlay(const std::vector<int>& hand,
                                  int previous_hideout);

  bool IsImmediateFugitiveCommitLegal() const;
  bool CanCompleteFugitiveSelection(int hideout,
                                    const std::vector<int>& selected_sprints,
                                    int last_sprint) const;
  bool CanCompleteFirstOpeningSelection(
      int hideout, const std::vector<int>& selected_sprints,
      const std::vector<int>& candidates, int candidate_index,
      std::vector<int>* extension) const;

  void ApplyChanceCard(int card);
  void ApplyDrawChoice(Action action);
  void ApplyFugitiveAction(Action action);
  void ApplyGuessAction(Action action);
  void CommitFugitivePlay();
  void CommitGuess();
  void StartMarshalTurn(bool first_turn);
  void FinishRoundOrStartNext();
  void ResolveEscape();
  void SkipDrawIfDeckEmpty();
  void SetTerminal(Winner winner, std::string reason);

  std::string ObservationJson(Player player, bool perfect_recall) const;
  static const char* PhaseString(Phase phase);
  static const char* RoleString(Player player);

  const int max_rounds_;
  Phase phase_ = Phase::kSetupChance;
  int round_number_ = 1;
  int setup_draw_index_ = 0;
  int opening_plays_remaining_ = 2;
  int draws_remaining_ = 0;
  int pending_pile_ = -1;
  Player pending_draw_player_ = kInvalidPlayer;

  std::array<bool, kMaxCard + 1> remaining_cards_{};
  std::array<std::vector<int>, kNumPlayers> hands_;
  std::vector<RouteNode> route_;

  int selected_hideout_ = -1;
  std::vector<int> selected_cards_;

  Winner winner_ = Winner::kNone;
  std::string terminal_reason_;
  std::vector<DrawRecord> draw_history_;
  std::vector<PlayRecord> play_history_;
  std::vector<GuessRecord> guess_history_;
};

class FugitiveGame : public Game {
 public:
  explicit FugitiveGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  int NumPlayers() const override { return kNumPlayers; }
  int MaxChanceOutcomes() const override { return kMaxCard; }
  int MaxChanceNodesInHistory() const override { return 38; }
  std::unique_ptr<State> NewInitialState() const override;
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  int MaxGameLength() const override { return 100 * max_rounds_ + 100; }

  int max_rounds() const { return max_rounds_; }

 private:
  const int max_rounds_;
};

}  // namespace fugitive
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_FUGITIVE_FUGITIVE_H_
