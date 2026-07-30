// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "open_spiel/games/fugitive/belief.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace fugitive {
namespace {

using json = nlohmann::json;

constexpr unsigned char kConstraintSatisfied = 0xff;

double DrawUnit(std::function<double()>* rng) {
  const double value = (*rng)();
  SPIEL_CHECK_GE(value, 0.0);
  SPIEL_CHECK_LT(value, 1.0);
  return value;
}

int DrawIndex(int size, std::function<double()>* rng) {
  SPIEL_CHECK_GT(size, 0);
  return std::min(size - 1, static_cast<int>(DrawUnit(rng) * size));
}

std::uint64_t CardBit(int card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LE(card, kMaxCard);
  return std::uint64_t{1} << card;
}

bool MaskContains(std::uint64_t mask, int card) {
  return (mask & CardBit(card)) != 0;
}

struct SearchState {
  std::uint8_t position = 0;
  std::uint8_t previous = 0;
  std::array<std::uint8_t, 3> used_draw_slots = {0, 0, 0};
  std::string failed_guess_progress;

  bool operator==(const SearchState& other) const {
    return position == other.position && previous == other.previous &&
           used_draw_slots == other.used_draw_slots &&
           failed_guess_progress == other.failed_guess_progress;
  }
};

struct SearchStateHash {
  std::size_t operator()(const SearchState& state) const {
    std::size_t hash = state.position;
    hash = hash * 47 + state.previous;
    for (int value : state.used_draw_slots) hash = hash * 41 + value;
    for (unsigned char value : state.failed_guess_progress) {
      hash = hash * 257 + value;
    }
    return hash;
  }
};

struct Transition {
  int hideout;
  SearchState next;
};

class RouteConstraintSearch {
 public:
  explicit RouteConstraintSearch(const MarshalBeliefInput& input)
      : input_(input) {}

  SearchState Root() const {
    SearchState root;
    root.failed_guess_progress.assign(input_.failed_guesses.size(), '\0');
    return root;
  }

  std::vector<Transition> Transitions(
      const SearchState& state, std::uint64_t* candidate_transitions) const {
    std::vector<Transition> transitions;
    if (state.position >= input_.route.size()) return transitions;

    const RoutePositionEvidence& evidence = input_.route[state.position];
    const int maximum_distance =
        evidence.known_sprint_value >= 0
            ? 3 + evidence.known_sprint_value
            : 3 + 2 * evidence.sprint_count;
    int upper = std::min(kMaxCard, static_cast<int>(state.previous) +
                                      maximum_distance);
    if (input_.phase != Phase::kManhuntGuess) upper = std::min(upper, 41);

    for (int hideout = state.previous + 1; hideout <= upper; ++hideout) {
      ++*candidate_transitions;
      if (evidence.known_hideout >= 0 &&
          hideout != evidence.known_hideout) {
        continue;
      }
      if (MaskContains(input_.unavailable_cards, hideout)) continue;
      if (hideout == kMaxCard &&
          (input_.phase != Phase::kManhuntGuess ||
           state.position + 1 != input_.route.size())) {
        continue;
      }

      SearchState next = state;
      ++next.position;
      next.previous = hideout;

      bool enough_draw_slots = true;
      for (int pile = 0; pile < 3; ++pile) {
        const int required = evidence.known_sprint_draws[pile] +
                             (PileForCard(hideout) == pile ? 1 : 0);
        next.used_draw_slots[pile] += required;
        const int available = std::upper_bound(
                                  input_.fugitive_draw_rounds[pile].begin(),
                                  input_.fugitive_draw_rounds[pile].end(),
                                  evidence.play_round) -
                              input_.fugitive_draw_rounds[pile].begin();
        if (next.used_draw_slots[pile] > available) {
          enough_draw_slots = false;
          break;
        }
      }
      if (!enough_draw_slots) continue;
      if (!AdvanceFailedGuesses(hideout, &next)) continue;
      transitions.push_back(std::move(Transition{hideout, std::move(next)}));
    }
    return transitions;
  }

 private:
  bool AdvanceFailedGuesses(int hideout, SearchState* state) const {
    const int route_position = state->position;
    for (int index = 0; index < input_.failed_guesses.size(); ++index) {
      unsigned char progress = static_cast<unsigned char>(
          state->failed_guess_progress[index]);
      if (progress == kConstraintSatisfied) continue;

      const FailedGuessEvidence& constraint = input_.failed_guesses[index];
      SPIEL_DCHECK_LT(progress, constraint.numbers.size());
      if (constraint.numbers[progress] < hideout) {
        state->failed_guess_progress[index] =
            static_cast<char>(kConstraintSatisfied);
        continue;
      }
      if (constraint.numbers[progress] == hideout) ++progress;
      if (progress == constraint.numbers.size()) return false;

      if (route_position == constraint.route_length) {
        progress = kConstraintSatisfied;
      }
      state->failed_guess_progress[index] = static_cast<char>(progress);
    }
    return true;
  }

  const MarshalBeliefInput& input_;
};

class RouteSupportCounter {
 public:
  explicit RouteSupportCounter(const MarshalBeliefInput& input)
      : input_(input), search_(input) {}

  MarshalRouteSupportResult Run() {
    const SearchState root = search_.Root();
    result_.route_count = Count(root);
    result_.memo_states = memo_.size();

    if (result_.route_count > 0) AccumulateMarginals(root);
    return result_;
  }

 private:
  std::uint64_t Count(const SearchState& state) {
    if (state.position == input_.route.size()) return 1;

    const auto found = memo_.find(state);
    if (found != memo_.end()) {
      ++result_.memo_hits;
      return found->second;
    }

    std::uint64_t total = 0;
    for (const Transition& transition :
         search_.Transitions(state, &result_.candidate_transitions)) {
      const std::uint64_t child_count = Count(transition.next);
      SPIEL_CHECK_LE(child_count,
                     std::numeric_limits<std::uint64_t>::max() - total);
      total += child_count;
    }
    memo_.emplace(state, total);
    return total;
  }

  void AccumulateMarginals(const SearchState& root) {
    std::unordered_map<SearchState, std::uint64_t, SearchStateHash> layer;
    layer.emplace(root, 1);

    while (!layer.empty()) {
      std::unordered_map<SearchState, std::uint64_t, SearchStateHash> next;
      for (const auto& [state, prefix_count] : layer) {
        for (const Transition& transition :
             search_.Transitions(state, &result_.candidate_transitions)) {
          const std::uint64_t suffix_count = Count(transition.next);
          if (suffix_count == 0) continue;
          SPIEL_CHECK_LE(
              prefix_count,
              std::numeric_limits<std::uint64_t>::max() / suffix_count);
          const std::uint64_t through_count = prefix_count * suffix_count;

          const RoutePositionEvidence& evidence =
              input_.route[state.position];
          if (evidence.known_hideout < 0 && transition.hideout != kMaxCard) {
            auto& card_count =
                result_.hidden_card_route_count[transition.hideout];
            SPIEL_CHECK_LE(
                through_count,
                std::numeric_limits<std::uint64_t>::max() - card_count);
            card_count += through_count;
          }

          auto [it, inserted] = next.emplace(transition.next, prefix_count);
          if (!inserted) {
            SPIEL_CHECK_LE(
                prefix_count,
                std::numeric_limits<std::uint64_t>::max() - it->second);
            it->second += prefix_count;
          }
        }
      }
      layer = std::move(next);
      if (!layer.empty() && layer.begin()->first.position == input_.route.size())
        break;
    }
  }

  const MarshalBeliefInput& input_;
  RouteConstraintSearch search_;
  MarshalRouteSupportResult result_;
  std::unordered_map<SearchState, std::uint64_t, SearchStateHash> memo_;
};

constexpr int kSprintBucketCount = 8;

int SprintBucketForCard(int card) {
  SPIEL_DCHECK_GE(card, kMinCard);
  SPIEL_DCHECK_LT(card, kMaxCard);
  const int parity = card % 2 == 0 ? 1 : 0;
  const int pile = PileForCard(card);
  return pile < 0 ? parity : 2 + 2 * pile + parity;
}

int PileForSprintBucket(int bucket) {
  return bucket < 2 ? -1 : (bucket - 2) / 2;
}

bool IsEvenSprintBucket(int bucket) { return bucket % 2 == 1; }

using CountTable =
    std::array<std::array<long double, kMaxCard + 1>, kMaxCard + 1>;

const CountTable& ChooseTable() {
  static const CountTable table = [] {
    CountTable values{};
    for (int n = 0; n <= kMaxCard; ++n) {
      values[n][0] = 1.0L;
      values[n][n] = 1.0L;
      for (int k = 1; k < n; ++k) {
        values[n][k] = values[n - 1][k - 1] + values[n - 1][k];
      }
    }
    return values;
  }();
  return table;
}

const CountTable& FallingFactorialTable() {
  static const CountTable table = [] {
    CountTable values{};
    for (int n = 0; n <= kMaxCard; ++n) {
      values[n][0] = 1.0L;
      for (int k = 1; k <= n; ++k) {
        values[n][k] = values[n][k - 1] * (n - k + 1);
      }
    }
    return values;
  }();
  return table;
}

struct CompletionState {
  std::uint8_t position = 0;
  std::array<std::uint8_t, kSprintBucketCount> remaining{};
  std::array<std::uint8_t, 3> used_draw_slots = {0, 0, 0};

  bool operator==(const CompletionState& other) const {
    return position == other.position && remaining == other.remaining &&
           used_draw_slots == other.used_draw_slots;
  }
};

std::uint64_t CompletionKey(const CompletionState& state) {
  // The 8 remaining-card counts and 3 used-slot counts occupy one nibble each.
  // Callers validate every packed count before the search starts.
  std::uint64_t key = state.position;
  for (int count : state.remaining) key = (key << 4) | count;
  for (int count : state.used_draw_slots) key = (key << 4) | count;
  return key;
}

struct FixedRouteCompletionResult {
  long double mass = 0.0L;
  std::uint64_t memo_states = 0;
  std::uint64_t memo_hits = 0;
  std::uint64_t allocation_evaluations = 0;
};

struct AllocationOption {
  std::array<std::uint8_t, kSprintBucketCount> take{};
  CompletionState next;
  long double mass = 0.0L;
};

class FixedRouteCompletionCounter {
 public:
  FixedRouteCompletionCounter(const MarshalBeliefInput& input,
                              const std::vector<int>& route)
      : input_(input), route_(route), eligible_draw_slots_(route.size()) {
    for (int position = 0; position < route.size(); ++position) {
      for (int pile = 0; pile < 3; ++pile) {
        eligible_draw_slots_[position][pile] = std::upper_bound(
            input_.fugitive_draw_rounds[pile].begin(),
            input_.fugitive_draw_rounds[pile].end(),
            input_.route[position].play_round) -
            input_.fugitive_draw_rounds[pile].begin();
      }
    }
  }

  FixedRouteCompletionResult Run() {
    const CompletionState root = RootState();
    result_.mass = Count(root);
    result_.memo_states = memo_.size();
    return result_;
  }

  MarshalHistorySample Sample(std::function<double()>* rng) {
    CompletionState state = RootState();
    SPIEL_CHECK_GT(Count(state), 0.0L);

    MarshalHistorySample sample;
    sample.route = route_;
    sample.sprint_cards.resize(route_.size());
    for (int pile = 0; pile < 3; ++pile) {
      sample.fugitive_draw_cards[pile].assign(
          input_.fugitive_draw_rounds[pile].size(), 0);
    }
    auto available = AvailableCards();

    while (state.position < route_.size()) {
      std::vector<AllocationOption> options;
      const long double total = ExpandPosition(state, &options);
      SPIEL_CHECK_GT(total, 0.0L);
      SPIEL_CHECK_FALSE(options.empty());

      long double target = DrawUnit(rng) * total;
      const AllocationOption* selected = &options.back();
      for (const AllocationOption& option : options) {
        if (target < option.mass) {
          selected = &option;
          break;
        }
        target -= option.mass;
      }

      const int position = state.position;
      const RoutePositionEvidence& evidence = input_.route[position];
      std::vector<int>& sprints = sample.sprint_cards[position];
      if (evidence.known_sprint_value >= 0) {
        SPIEL_CHECK_EQ(evidence.known_sprint_cards.size(),
                       evidence.sprint_count);
        sprints = evidence.known_sprint_cards;
      } else {
        for (int bucket = 0; bucket < kSprintBucketCount; ++bucket) {
          for (int count = 0; count < selected->take[bucket]; ++count) {
            std::vector<int>& cards = available[bucket];
            const int index = DrawIndex(cards.size(), rng);
            sprints.push_back(cards[index]);
            cards.erase(cards.begin() + index);
          }
        }
      }
      std::sort(sprints.begin(), sprints.end());

      std::array<std::vector<int>, 3> required;
      const int hideout_pile = PileForCard(route_[position]);
      if (hideout_pile >= 0) required[hideout_pile].push_back(route_[position]);
      for (int card : sprints) {
        const int pile = PileForCard(card);
        if (pile >= 0) required[pile].push_back(card);
      }
      for (int pile = 0; pile < 3; ++pile) {
        std::vector<int> free_slots;
        for (int slot = 0; slot < eligible_draw_slots_[position][pile]; ++slot) {
          if (sample.fugitive_draw_cards[pile][slot] == 0) {
            free_slots.push_back(slot);
          }
        }
        SPIEL_CHECK_GE(free_slots.size(), required[pile].size());
        for (int card : required[pile]) {
          const int index = DrawIndex(free_slots.size(), rng);
          const int slot = free_slots[index];
          free_slots.erase(free_slots.begin() + index);
          sample.fugitive_draw_cards[pile][slot] = card;
        }
      }
      state = selected->next;
    }

    for (int pile = 0; pile < 3; ++pile) {
      std::vector<int> remaining;
      const int first_bucket = 2 + 2 * pile;
      for (int bucket = first_bucket; bucket <= first_bucket + 1; ++bucket) {
        remaining.insert(remaining.end(), available[bucket].begin(),
                         available[bucket].end());
      }
      for (int& card : sample.fugitive_draw_cards[pile]) {
        if (card != 0) continue;
        const int index = DrawIndex(remaining.size(), rng);
        card = remaining[index];
        remaining.erase(remaining.begin() + index);
      }
    }
    return sample;
  }

 private:
  std::array<std::vector<int>, kSprintBucketCount> AvailableCards() const {
    std::array<std::vector<int>, kSprintBucketCount> available;
    std::uint64_t route_cards = 0;
    for (int hideout : route_) route_cards |= CardBit(hideout);
    for (int card = kMinCard; card < kMaxCard; ++card) {
      if (!MaskContains(route_cards | input_.unavailable_cards, card)) {
        available[SprintBucketForCard(card)].push_back(card);
      }
    }
    return available;
  }

  CompletionState RootState() const {
    CompletionState root;
    const auto available = AvailableCards();
    for (int bucket = 0; bucket < kSprintBucketCount; ++bucket) {
      SPIEL_CHECK_LT(available[bucket].size(), 16);
      root.remaining[bucket] = available[bucket].size();
    }
    return root;
  }

  long double Count(const CompletionState& state) {
    const std::uint64_t key = CompletionKey(state);
    const auto found = memo_.find(key);
    if (found != memo_.end()) {
      ++result_.memo_hits;
      return found->second;
    }

    long double total = 0.0L;
    if (state.position == route_.size()) {
      total = TerminalMass(state);
    } else {
      total = ExpandPosition(state, nullptr);
    }
    memo_.emplace(key, total);
    return total;
  }

  long double ExpandPosition(const CompletionState& state,
                             std::vector<AllocationOption>* options) {
    const RoutePositionEvidence& evidence = input_.route[state.position];
    std::array<std::uint8_t, kSprintBucketCount> take{};
    if (evidence.known_sprint_value >= 0 || evidence.sprint_count == 0) {
      CompletionState next;
      const long double mass = ApplyAllocation(state, take, 1.0L, &next);
      if (options != nullptr && mass > 0.0L) {
        options->push_back(AllocationOption{take, next, mass});
      }
      return mass;
    }

    const int previous =
        state.position == 0 ? 0 : route_[state.position - 1];
    const int gap = route_[state.position] - previous;
    const int minimum_even =
        std::max(0, gap - 3 - evidence.sprint_count);
    std::array<int, 3> pile_capacity{};
    for (int pile = 0; pile < 3; ++pile) {
      const int base_required =
          evidence.known_sprint_draws[pile] +
          (PileForCard(route_[state.position]) == pile ? 1 : 0);
      pile_capacity[pile] = eligible_draw_slots_[state.position][pile] -
                            state.used_draw_slots[pile] - base_required;
      if (pile_capacity[pile] < 0) return 0.0L;
    }
    return EnumerateAllocations(
        state, /*bucket=*/0, evidence.sprint_count, /*even_taken=*/0,
        minimum_even, /*identity_ways=*/1.0L, pile_capacity, &take, options);
  }

  long double EnumerateAllocations(
      const CompletionState& state, int bucket, int cards_left,
      int even_taken, int minimum_even, long double identity_ways,
      const std::array<int, 3>& pile_capacity,
      std::array<std::uint8_t, kSprintBucketCount>* take,
      std::vector<AllocationOption>* options) {
    if (bucket == kSprintBucketCount) {
      if (cards_left != 0 || even_taken < minimum_even) return 0.0L;
      CompletionState next;
      const long double mass =
          ApplyAllocation(state, *take, identity_ways, &next);
      if (options != nullptr && mass > 0.0L) {
        options->push_back(AllocationOption{*take, next, mass});
      }
      return mass;
    }

    long double total = 0.0L;
    int maximum = std::min<int>(cards_left, state.remaining[bucket]);
    const int pile = PileForSprintBucket(bucket);
    if (pile >= 0) {
      const int already_taken = bucket % 2 == 1 ? (*take)[bucket - 1] : 0;
      maximum = std::min(maximum, pile_capacity[pile] - already_taken);
    }
    for (int count = 0; count <= maximum; ++count) {
      (*take)[bucket] = count;
      total += EnumerateAllocations(
          state, bucket + 1, cards_left - count,
          even_taken + (IsEvenSprintBucket(bucket) ? count : 0),
          minimum_even,
          identity_ways * ChooseTable()[state.remaining[bucket]][count],
          pile_capacity, take, options);
    }
    (*take)[bucket] = 0;
    return total;
  }

  long double ApplyAllocation(
      const CompletionState& state,
      const std::array<std::uint8_t, kSprintBucketCount>& take,
      long double identity_ways, CompletionState* next_state) {
    ++result_.allocation_evaluations;
    CompletionState next = state;
    ++next.position;

    std::array<int, 3> required =
        input_.route[state.position].known_sprint_draws;
    const int hideout_pile = PileForCard(route_[state.position]);
    if (hideout_pile >= 0) ++required[hideout_pile];
    for (int bucket = 0; bucket < kSprintBucketCount; ++bucket) {
      next.remaining[bucket] -= take[bucket];
      const int pile = PileForSprintBucket(bucket);
      if (pile >= 0) required[pile] += take[bucket];
    }

    long double slot_ways = 1.0L;
    for (int pile = 0; pile < 3; ++pile) {
      const int free = eligible_draw_slots_[state.position][pile] -
                       state.used_draw_slots[pile];
      if (required[pile] > free) return 0.0L;
      slot_ways *= FallingFactorialTable()[free][required[pile]];
      next.used_draw_slots[pile] += required[pile];
    }
    if (next_state != nullptr) *next_state = next;
    return identity_ways * slot_ways * Count(next);
  }

  long double TerminalMass(const CompletionState& state) const {
    long double mass = 1.0L;
    for (int pile = 0; pile < 3; ++pile) {
      const int remaining_slots = input_.fugitive_draw_rounds[pile].size() -
                                  state.used_draw_slots[pile];
      const int first_bucket = 2 + 2 * pile;
      const int available_cards = state.remaining[first_bucket] +
                                  state.remaining[first_bucket + 1];
      if (remaining_slots > available_cards) return 0.0L;
      mass *= FallingFactorialTable()[available_cards][remaining_slots];
    }
    return mass;
  }

  const MarshalBeliefInput& input_;
  const std::vector<int>& route_;
  std::vector<std::array<std::uint8_t, 3>> eligible_draw_slots_;
  FixedRouteCompletionResult result_;
  std::unordered_map<std::uint64_t, long double> memo_;
};

class CompletionCounter {
 public:
  explicit CompletionCounter(const MarshalBeliefInput& input,
                             std::function<double()>* sample_rng = nullptr)
      : input_(input), search_(input), sample_rng_(sample_rng) {
    route_.reserve(input.route.size());
  }

  MarshalCompletionResult Run() {
    EnumerateRoutes(search_.Root());
    return result_;
  }

  const std::vector<int>& sampled_route() const {
    SPIEL_CHECK_FALSE(sampled_route_.empty());
    return sampled_route_;
  }

 private:
  void EnumerateRoutes(const SearchState& state) {
    if (state.position == input_.route.size()) {
      ++result_.route_support_count;
      const std::string signature = CompletionSignature();
      auto found = completion_cache_.find(signature);
      if (found == completion_cache_.end()) {
        const FixedRouteCompletionResult completion =
            FixedRouteCompletionCounter(input_, route_).Run();
        found = completion_cache_.emplace(signature, completion).first;
        ++result_.completion_route_classes;
        result_.completion_memo_states += completion.memo_states;
        result_.completion_memo_hits += completion.memo_hits;
        result_.completion_allocation_evaluations +=
            completion.allocation_evaluations;
        result_.max_completion_memo_states =
            std::max(result_.max_completion_memo_states,
                     completion.memo_states);
      } else {
        ++result_.completion_route_cache_hits;
      }
      const FixedRouteCompletionResult& completion = found->second;
      if (completion.mass > 0.0L) {
        ++result_.completable_route_count;
        for (int position = 0; position < route_.size(); ++position) {
          if (input_.route[position].known_hideout < 0 &&
              route_[position] != kMaxCard) {
            result_.hidden_card_history_mass[route_[position]] +=
                completion.mass;
          }
        }
        const long double new_mass =
            result_.uniform_consistent_history_mass + completion.mass;
        if (sample_rng_ != nullptr &&
            DrawUnit(sample_rng_) * new_mass < completion.mass) {
          sampled_route_ = route_;
        }
        result_.uniform_consistent_history_mass = new_mass;
      }
      return;
    }

    for (const Transition& transition : search_.Transitions(
             state, &result_.route_candidate_transitions)) {
      route_.push_back(transition.hideout);
      EnumerateRoutes(transition.next);
      route_.pop_back();
    }
  }

  std::string CompletionSignature() const {
    std::string signature;
    signature.reserve(2 * route_.size());
    for (int position = 0; position < route_.size(); ++position) {
      const int hideout_bucket = route_[position] == kMaxCard
                                     ? kSprintBucketCount
                                     : SprintBucketForCard(route_[position]);
      int minimum_even = 0;
      const RoutePositionEvidence& evidence = input_.route[position];
      if (evidence.known_sprint_value < 0) {
        const int previous = position == 0 ? 0 : route_[position - 1];
        const int gap = route_[position] - previous;
        minimum_even = std::max(0, gap - 3 - evidence.sprint_count);
      }
      signature.push_back(static_cast<char>(hideout_bucket));
      signature.push_back(static_cast<char>(minimum_even));
    }
    return signature;
  }

  const MarshalBeliefInput& input_;
  RouteConstraintSearch search_;
  std::vector<int> route_;
  std::vector<int> sampled_route_;
  std::function<double()>* sample_rng_;
  std::unordered_map<std::string, FixedRouteCompletionResult>
      completion_cache_;
  MarshalCompletionResult result_;
};

void ApplyChecked(State* state, Action action) {
  const std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), action) != legal.end());
  state->ApplyAction(action);
}

void ValidateInput(const MarshalBeliefInput& input) {
  SPIEL_CHECK_LE(input.route.size(), kMaxCard);
  for (const auto& rounds : input.fugitive_draw_rounds) {
    SPIEL_CHECK_LT(rounds.size(), 16);
    SPIEL_CHECK_TRUE(std::is_sorted(rounds.begin(), rounds.end()));
  }
  for (const FailedGuessEvidence& constraint : input.failed_guesses) {
    SPIEL_CHECK_FALSE(constraint.numbers.empty());
    SPIEL_CHECK_GE(constraint.route_length, 0);
    SPIEL_CHECK_LE(constraint.route_length, input.route.size());
    SPIEL_CHECK_TRUE(std::is_sorted(constraint.numbers.begin(),
                                    constraint.numbers.end()));
  }
}

}  // namespace

bool RoutePositionEvidence::operator==(
    const RoutePositionEvidence& other) const {
  return known_hideout == other.known_hideout &&
         sprint_count == other.sprint_count &&
         known_sprint_value == other.known_sprint_value &&
         play_round == other.play_round &&
         known_sprint_draws == other.known_sprint_draws &&
         known_sprint_cards == other.known_sprint_cards;
}

bool FailedGuessEvidence::operator==(
    const FailedGuessEvidence& other) const {
  return route_length == other.route_length && numbers == other.numbers;
}

bool MarshalBeliefInput::operator==(const MarshalBeliefInput& other) const {
  return phase == other.phase && route == other.route &&
         fugitive_draw_rounds == other.fugitive_draw_rounds &&
         unavailable_cards == other.unavailable_cards &&
         failed_guesses == other.failed_guesses;
}

double MarshalRouteSupportResult::HiddenRouteSupportFraction(int card) const {
  SPIEL_CHECK_GE(card, kMinCard);
  SPIEL_CHECK_LE(card, 41);
  if (route_count == 0) return 0.0;
  return static_cast<double>(hidden_card_route_count[card]) /
         static_cast<double>(route_count);
}

long double
MarshalCompletionResult::UniformConsistentHiddenHideoutProbability(
    int card) const {
  SPIEL_CHECK_GE(card, kMinCard);
  SPIEL_CHECK_LE(card, 41);
  if (uniform_consistent_history_mass == 0.0L) return 0.0L;
  return hidden_card_history_mass[card] /
         uniform_consistent_history_mass;
}

MarshalBeliefInput BuildMarshalBeliefInput(
    const std::string& marshal_information_state) {
  const json information_state = json::parse(marshal_information_state);
  SPIEL_CHECK_EQ(information_state.at("schema"),
                 "fugitive.information_state");
  SPIEL_CHECK_EQ(information_state.at("schema_version"), 1);
  SPIEL_CHECK_TRUE(information_state.at("perfect_recall").get<bool>());

  const json& observation = information_state.at("observation");
  SPIEL_CHECK_EQ(observation.at("role"), "marshal");
  const std::string action_stage =
      observation.at("action_stage").get<std::string>();

  MarshalBeliefInput input;
  if (action_stage == "marshal_guess") {
    input.phase = Phase::kMarshalGuess;
  } else {
    SPIEL_CHECK_EQ(action_stage, "manhunt_guess");
    input.phase = Phase::kManhuntGuess;
  }

  for (int card : observation.at("hand").get<std::vector<int>>()) {
    input.unavailable_cards |= CardBit(card);
  }

  const json& route = observation.at("route");
  std::vector<int> play_rounds(route.size(), -1);
  for (const json& play : observation.at("play_history")) {
    if (play.at("route_index").is_null()) continue;
    const int route_index = play.at("route_index").get<int>();
    SPIEL_CHECK_GE(route_index, 1);
    SPIEL_CHECK_LT(route_index, play_rounds.size());
    play_rounds[route_index] = play.at("round_number").get<int>();
  }

  for (int route_index = 1; route_index < route.size(); ++route_index) {
    const json& node = route[route_index];
    SPIEL_CHECK_GE(play_rounds[route_index], 1);
    RoutePositionEvidence evidence;
    if (!node.at("hideout").is_null()) {
      evidence.known_hideout = node.at("hideout").get<int>();
    }
    evidence.sprint_count = node.at("sprint_count").get<int>();
    evidence.play_round = play_rounds[route_index];
    if (!node.at("sprint_cards").is_null()) {
      const std::vector<int> sprint_cards =
          node.at("sprint_cards").get<std::vector<int>>();
      SPIEL_CHECK_EQ(sprint_cards.size(), evidence.sprint_count);
      evidence.known_sprint_cards = sprint_cards;
      evidence.known_sprint_value = 0;
      for (int sprint : sprint_cards) {
        input.unavailable_cards |= CardBit(sprint);
        evidence.known_sprint_value += SprintValue(sprint);
        const int pile = PileForCard(sprint);
        if (pile >= 0) ++evidence.known_sprint_draws[pile];
      }
    }
    input.route.push_back(std::move(evidence));
  }

  for (const json& draw : observation.at("draw_history")) {
    if (draw.at("role") != "fugitive") continue;
    const int pile = draw.at("pile").get<int>();
    SPIEL_CHECK_GE(pile, 0);
    SPIEL_CHECK_LT(pile, input.fugitive_draw_rounds.size());
    input.fugitive_draw_rounds[pile].push_back(
        draw.at("round_number").get<int>());
  }
  for (auto& rounds : input.fugitive_draw_rounds) {
    std::sort(rounds.begin(), rounds.end());
  }

  std::array<bool, kMaxCard + 1> revealed{};
  revealed[0] = true;
  for (const json& guess : observation.at("guess_history")) {
    const std::vector<int> numbers =
        guess.at("numbers").get<std::vector<int>>();
    if (guess.at("success").get<bool>()) {
      for (int number : numbers) revealed[number] = true;
      continue;
    }

    bool already_explained = false;
    for (int number : numbers) {
      if (revealed[number] ||
          MaskContains(input.unavailable_cards, number)) {
        already_explained = true;
        break;
      }
    }
    if (!already_explained) {
      FailedGuessEvidence evidence;
      evidence.route_length = guess.at("route_length").get<int>();
      evidence.numbers = numbers;
      std::sort(evidence.numbers.begin(), evidence.numbers.end());
      input.failed_guesses.push_back(std::move(evidence));
    }
  }
  return input;
}

MarshalRouteSupportResult ComputeMarshalRouteSupport(
    const MarshalBeliefInput& input) {
  ValidateInput(input);
  return RouteSupportCounter(input).Run();
}

MarshalCompletionResult ComputeMarshalCompletion(
    const MarshalBeliefInput& input) {
  ValidateInput(input);
  return CompletionCounter(input).Run();
}

MarshalHistorySample SampleMarshalHistory(const MarshalBeliefInput& input,
                                          std::function<double()> rng) {
  ValidateInput(input);
  CompletionCounter route_counter(input, &rng);
  const MarshalCompletionResult completion = route_counter.Run();
  SPIEL_CHECK_GT(completion.uniform_consistent_history_mass, 0.0L);

  FixedRouteCompletionCounter fixed_counter(input,
                                             route_counter.sampled_route());
  const FixedRouteCompletionResult fixed = fixed_counter.Run();
  SPIEL_CHECK_GT(fixed.mass, 0.0L);
  return fixed_counter.Sample(&rng);
}

std::unique_ptr<State> ReplayMarshalHistory(
    const Game& game, const std::string& marshal_information_state,
    const MarshalHistorySample& sample) {
  SPIEL_CHECK_EQ(game.GetType().short_name, "fugitive");
  const json information_state = json::parse(marshal_information_state);
  SPIEL_CHECK_EQ(information_state.at("schema"),
                 "fugitive.information_state");
  const json& observation = information_state.at("observation");
  SPIEL_CHECK_EQ(observation.at("role"), "marshal");
  SPIEL_CHECK_EQ(sample.route.size() + 1,
                 observation.at("route").size());
  SPIEL_CHECK_EQ(sample.sprint_cards.size(), sample.route.size());
  const json& draws = observation.at("draw_history");
  const json& plays = observation.at("play_history");
  const json& guesses = observation.at("guess_history");

  std::unique_ptr<State> replay = game.NewInitialState();
  std::size_t draw_index = 0;
  std::size_t play_index = 0;
  std::size_t guess_index = 0;
  std::array<std::size_t, 3> fugitive_draw_index{};

  while (draw_index < draws.size() || play_index < plays.size() ||
         guess_index < guesses.size()) {
    SPIEL_CHECK_FALSE(replay->IsTerminal());
    const Phase phase = down_cast<const FugitiveState&>(*replay).phase();

    if (replay->CurrentPlayer() == kChancePlayerId) {
      SPIEL_CHECK_LT(draw_index, draws.size());
      const json& draw = draws[draw_index];
      const int pile = draw.at("pile").get<int>();
      SPIEL_CHECK_GE(pile, 0);
      SPIEL_CHECK_LT(pile, 3);
      const bool fugitive_draw = draw.at("role") == "fugitive";
      SPIEL_CHECK_EQ(fugitive_draw,
                     phase == Phase::kSetupChance ||
                         phase == Phase::kFugitiveDrawChance);
      int card = 0;
      if (fugitive_draw) {
        SPIEL_CHECK_LT(fugitive_draw_index[pile],
                       sample.fugitive_draw_cards[pile].size());
        card = sample.fugitive_draw_cards[pile]
                                           [fugitive_draw_index[pile]++];
      } else {
        card = draw.at("card").get<int>();
      }
      ApplyChecked(replay.get(), card);
      ++draw_index;
      continue;
    }

    if (phase == Phase::kFugitiveDrawChoice ||
        phase == Phase::kMarshalDrawChoice) {
      SPIEL_CHECK_LT(draw_index, draws.size());
      const json& draw = draws[draw_index];
      const bool fugitive_draw = draw.at("role") == "fugitive";
      SPIEL_CHECK_EQ(fugitive_draw, phase == Phase::kFugitiveDrawChoice);
      ApplyChecked(replay.get(),
                   kFirstPileAction + draw.at("pile").get<int>());
      continue;
    }

    if (phase == Phase::kFugitiveHideout) {
      SPIEL_CHECK_LT(play_index, plays.size());
      const json& play = plays[play_index++];
      if (play.at("passed").get<bool>()) {
        ApplyChecked(replay.get(), kPassAction);
        continue;
      }
      const int route_index = play.at("route_index").get<int>();
      SPIEL_CHECK_GE(route_index, 1);
      SPIEL_CHECK_LE(route_index, sample.route.size());
      SPIEL_CHECK_EQ(sample.sprint_cards.size(), sample.route.size());
      ApplyChecked(replay.get(), sample.route[route_index - 1]);
      for (int sprint : sample.sprint_cards[route_index - 1]) {
        ApplyChecked(replay.get(), sprint);
      }
      ApplyChecked(replay.get(), kCommitAction);
      continue;
    }

    if (phase == Phase::kMarshalGuess || phase == Phase::kManhuntGuess) {
      SPIEL_CHECK_LT(guess_index, guesses.size());
      const json& guess = guesses[guess_index++];
      SPIEL_CHECK_EQ(guess.at("manhunt").get<bool>(),
                     phase == Phase::kManhuntGuess);
      for (int number : guess.at("numbers").get<std::vector<int>>()) {
        ApplyChecked(replay.get(), number);
      }
      ApplyChecked(replay.get(), kCommitAction);
      continue;
    }

    SpielFatalError("Cannot replay Marshal history from this phase");
  }

  for (int pile = 0; pile < 3; ++pile) {
    SPIEL_CHECK_EQ(fugitive_draw_index[pile],
                   sample.fugitive_draw_cards[pile].size());
  }
  if (!observation.at("pending_action").is_null()) {
    const json& pending = observation.at("pending_action");
    SPIEL_CHECK_TRUE(pending.at("kind") == "guess" ||
                     pending.at("kind") == "manhunt");
    for (int number : pending.at("numbers").get<std::vector<int>>()) {
      ApplyChecked(replay.get(), number);
    }
  }
  SPIEL_CHECK_EQ(replay->InformationStateString(kMarshalPlayer),
                 marshal_information_state);
  return replay;
}

}  // namespace fugitive
}  // namespace open_spiel
