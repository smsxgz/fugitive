// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "open_spiel/games/fugitive/belief.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
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

struct Options {
  std::string mode = "fixed";
  int samples_per_bucket = 32;
  std::uint64_t seed_start = 0;
  int max_seeds = 10000;
  int replay_samples = 1;
  std::uint64_t manhunt_completion_calls = 0;
  int manhunt_particles = 0;
};

struct Bucket {
  std::string name;
  std::set<std::string> information_states;
  std::vector<std::int64_t> route_microseconds;
  std::vector<std::int64_t> completion_microseconds;
  std::vector<std::int64_t> sample_microseconds;
  std::vector<std::int64_t> replay_microseconds;
};

struct Measurements {
  int count = 0;
  std::vector<std::int64_t> information_state_bytes;
  std::vector<std::int64_t> information_state_microseconds;
  std::vector<std::int64_t> parse_microseconds;
  std::vector<std::int64_t> route_microseconds;
  std::vector<std::int64_t> completion_microseconds;
  std::vector<std::int64_t> pipeline_microseconds;
  std::vector<std::int64_t> sample_microseconds;
  std::vector<std::int64_t> replay_microseconds;
};

FugitiveState& AsFugitive(State* state) {
  return down_cast<FugitiveState&>(*state);
}

const FugitiveState& AsFugitive(const State* state) {
  return down_cast<const FugitiveState&>(*state);
}

int UniformIndex(std::mt19937_64* rng, int size) {
  SPIEL_CHECK_GT(size, 0);
  return std::uniform_int_distribution<int>(0, size - 1)(*rng);
}

Action SampleChance(const State& state, std::mt19937_64* rng) {
  const double sample = std::generate_canonical<double, 64>(*rng);
  return SampleAction(state.ChanceOutcomes(), sample).first;
}

void ApplyMarshalGuess(State* state, std::mt19937_64* rng) {
  const bool manhunt = AsFugitive(state).phase() == Phase::kManhuntGuess;
  int count = 1;
  if (!manhunt && std::bernoulli_distribution(0.2)(*rng)) count = 2;

  std::vector<int> numbers(41);
  for (int index = 0; index < numbers.size(); ++index) {
    numbers[index] = index + 1;
  }
  std::shuffle(numbers.begin(), numbers.end(), *rng);
  numbers.resize(count);
  std::sort(numbers.begin(), numbers.end());
  for (int number : numbers) state->ApplyAction(number);
  state->ApplyAction(kCommitAction);
}

void AdvanceNonMarshalPhase(State* state, std::mt19937_64* rng) {
  const Phase phase = AsFugitive(state).phase();
  if (state->CurrentPlayer() == kChancePlayerId) {
    state->ApplyAction(SampleChance(*state, rng));
    return;
  }

  std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_FALSE(legal.empty());
  if (phase == Phase::kFugitiveDrawChoice ||
      phase == Phase::kMarshalDrawChoice) {
    state->ApplyAction(legal[UniformIndex(rng, legal.size())]);
    return;
  }

  if (phase == Phase::kFugitiveHideout) {
    std::vector<Action> plays;
    for (Action action : legal) {
      if (action != kPassAction) plays.push_back(action);
    }
    if (plays.empty()) {
      state->ApplyAction(kPassAction);
    } else {
      const int candidate_count = std::min<int>(3, plays.size());
      state->ApplyAction(plays[UniformIndex(rng, candidate_count)]);
    }
    return;
  }

  SPIEL_CHECK_EQ(phase, Phase::kFugitiveSprint);
  if (std::find(legal.begin(), legal.end(), kCommitAction) != legal.end()) {
    state->ApplyAction(kCommitAction);
  } else {
    state->ApplyAction(legal[UniformIndex(rng, legal.size())]);
  }
}

MarshalBeliefInput RestrictToActualRoute(const FugitiveState& state,
                                         MarshalBeliefInput input) {
  SPIEL_CHECK_EQ(input.route.size(), state.route().size() - 1);
  for (int index = 0; index < input.route.size(); ++index) {
    input.route[index].known_hideout = state.route()[index + 1].hideout;
  }
  return input;
}

std::int64_t Microseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

std::string Scientific(long double value) {
  std::ostringstream output;
  output << std::scientific
         << std::setprecision(std::numeric_limits<long double>::digits10)
         << value;
  return output.str();
}

json Evaluate(const FugitiveState& state,
              const std::string& information_state,
              std::int64_t information_state_microseconds,
              const std::string& checkpoint, std::uint64_t seed,
              int replay_samples, std::uint64_t manhunt_completion_calls,
              int manhunt_particles) {
  const Clock::time_point parse_start = Clock::now();
  const MarshalBeliefInput input = BuildMarshalBeliefInput(information_state);
  const Clock::time_point completion_start = Clock::now();
  const MarshalCompletionResult completion_result =
      ComputeMarshalCompletion(input);
  const Clock::time_point completion_end = Clock::now();
  const Clock::time_point route_start = Clock::now();
  const MarshalRouteSupportResult route_result =
      ComputeMarshalRouteSupport(input);
  const Clock::time_point route_end = Clock::now();

  SPIEL_CHECK_GT(route_result.route_count, 0);
  SPIEL_CHECK_EQ(completion_result.route_support_count,
                 route_result.route_count);
  SPIEL_CHECK_GT(completion_result.completable_route_count, 0);
  SPIEL_CHECK_LE(completion_result.completable_route_count,
                 completion_result.route_support_count);
  SPIEL_CHECK_GE(completion_result.uniform_consistent_history_mass,
                 completion_result.completable_route_count);
  SPIEL_CHECK_TRUE(std::isfinite(
      completion_result.uniform_consistent_history_mass));

  const MarshalBeliefInput actual_route_input =
      RestrictToActualRoute(state, input);
  const Clock::time_point oracle_start = Clock::now();
  const MarshalCompletionResult actual_route_result =
      ComputeMarshalCompletion(actual_route_input);
  const Clock::time_point oracle_end = Clock::now();
  SPIEL_CHECK_EQ(actual_route_result.route_support_count, 1);
  SPIEL_CHECK_EQ(actual_route_result.completable_route_count, 1);
  SPIEL_CHECK_GT(actual_route_result.uniform_consistent_history_mass, 0.0L);
  SPIEL_CHECK_LE(actual_route_result.uniform_consistent_history_mass,
                 completion_result.uniform_consistent_history_mass);

  std::uint64_t sample_seed = seed ^ 0x9e3779b97f4a7c15ULL;
  for (unsigned char byte : information_state) {
    sample_seed = (sample_seed ^ byte) * 1099511628211ULL;
  }
  std::mt19937_64 sample_rng(sample_seed);
  auto uniform = [&sample_rng]() {
    return std::generate_canonical<double, 64>(sample_rng);
  };
  std::int64_t sample_microseconds = 0;
  std::int64_t replay_microseconds = 0;
  for (int trial = 0; trial < replay_samples; ++trial) {
    const Clock::time_point sample_start = Clock::now();
    const MarshalHistorySample sample = SampleMarshalHistory(input, uniform);
    const Clock::time_point replay_start = Clock::now();
    ReplayMarshalHistory(*state.GetGame(), information_state, sample);
    const Clock::time_point replay_end = Clock::now();
    sample_microseconds += Microseconds(sample_start, replay_start);
    replay_microseconds += Microseconds(replay_start, replay_end);
  }

  json manhunt = nullptr;
  if (input.phase == Phase::kManhuntGuess &&
      manhunt_completion_calls > 0) {
    const Clock::time_point manhunt_start = Clock::now();
    const MarshalManhuntResult result =
        ComputeUniformConsistentManhuntValue(
            input, MarshalManhuntOptions{
                       /*max_completion_calls=*/manhunt_completion_calls,
                       /*max_solver_states=*/1000});
    manhunt = {
        {"lower_bound", static_cast<double>(result.lower_bound)},
        {"upper_bound", static_cast<double>(result.upper_bound)},
        {"best_guess", result.best_guess},
        {"exact", result.exact},
        {"solver_states", result.solver_states},
        {"solver_cache_hits", result.solver_cache_hits},
        {"completion_calls", result.completion_calls},
        {"completion_cache_hits", result.completion_cache_hits},
        {"positive_reveal_outcomes", result.positive_reveal_outcomes},
        {"time_us", Microseconds(manhunt_start, Clock::now())},
    };
  }

  json sampled_manhunt = nullptr;
  if (input.phase == Phase::kManhuntGuess && manhunt_particles > 0) {
    std::mt19937_64 manhunt_rng(sample_seed ^ 0xd1b54a32d192ed03ULL);
    auto manhunt_uniform = [&manhunt_rng]() {
      return std::generate_canonical<double, 64>(manhunt_rng);
    };
    const Clock::time_point manhunt_start = Clock::now();
    const MarshalSampledManhuntResult result = ComputeSampledManhuntValue(
        input, manhunt_uniform,
        MarshalSampledManhuntOptions{
            /*particles=*/manhunt_particles,
            /*max_solver_states=*/100000});
    sampled_manhunt = {
        {"lower_bound", static_cast<double>(result.lower_bound)},
        {"upper_bound", static_cast<double>(result.upper_bound)},
        {"best_guess", result.best_guess},
        {"particles", result.particles},
        {"solver_states", result.solver_states},
        {"solver_cache_hits", result.solver_cache_hits},
        {"exact_for_empirical_belief", result.exact_for_empirical_belief},
        {"time_us", Microseconds(manhunt_start, Clock::now())},
    };
  }

  int hidden_positions = 0;
  int hidden_sprints = 0;
  int total_sprints = 0;
  for (const RoutePositionEvidence& position : input.route) {
    if (position.known_hideout < 0) ++hidden_positions;
    if (position.known_sprint_value < 0) {
      hidden_sprints += position.sprint_count;
    }
    total_sprints += position.sprint_count;
  }
  if (hidden_sprints == 0) {
    SPIEL_CHECK_EQ(completion_result.completable_route_count,
                   completion_result.route_support_count);
  }

  std::vector<std::uint64_t> card_counts;
  std::vector<double> hidden_hideout_probabilities;
  std::vector<int> supported_cards;
  std::uint64_t marginal_sum = 0;
  for (int card = 1; card <= 41; ++card) {
    card_counts.push_back(route_result.hidden_card_route_count[card]);
    hidden_hideout_probabilities.push_back(static_cast<double>(
        completion_result.UniformConsistentHiddenHideoutProbability(card)));
    marginal_sum += route_result.hidden_card_route_count[card];
    if (route_result.hidden_card_route_count[card] > 0) {
      supported_cards.push_back(card);
    }
  }
  SPIEL_CHECK_EQ(
      marginal_sum,
      route_result.route_count * static_cast<std::uint64_t>(hidden_positions));

  const json parsed = json::parse(information_state);
  const json& observation = parsed.at("observation");
  int successful_guesses = 0;
  int failed_guesses = 0;
  for (const json& guess : observation.at("guess_history")) {
    if (guess.at("success").get<bool>()) {
      ++successful_guesses;
    } else {
      ++failed_guesses;
    }
  }
  std::array<std::array<int, 3>, kNumPlayers> draw_counts{};
  for (const json& draw : observation.at("draw_history")) {
    const int player = draw.at("role") == "fugitive" ? kFugitivePlayer
                                                      : kMarshalPlayer;
    ++draw_counts[player][draw.at("pile").get<int>()];
  }

  json record = {
      {"type", "sample"},
      {"checkpoint", checkpoint},
      {"seed", seed},
      {"phase", observation.at("action_stage")},
      {"round", observation.at("round_number")},
      {"route_length", input.route.size()},
      {"hidden_route_positions", hidden_positions},
      {"known_route_positions",
       static_cast<int>(input.route.size()) - hidden_positions},
      {"total_sprint_count", total_sprints},
      {"hidden_sprint_count", hidden_sprints},
      {"guess_count", successful_guesses + failed_guesses},
      {"successful_guess_count", successful_guesses},
      {"failed_guess_count", failed_guesses},
      {"active_failed_guess_constraints", input.failed_guesses.size()},
      {"fugitive_draw_count_by_pile", draw_counts[kFugitivePlayer]},
      {"marshal_draw_count_by_pile", draw_counts[kMarshalPlayer]},
      {"belief_model", "uniform_consistent_chance_histories"},
      {"policy_weighted", false},
      {"route_support_upper_bound", route_result.route_count},
      {"log10_route_support_upper_bound",
       std::log10(static_cast<long double>(route_result.route_count))},
      {"fully_completable_route_count",
       completion_result.completable_route_count},
      {"routes_pruned_by_completion",
       completion_result.route_support_count -
           completion_result.completable_route_count},
      {"uniform_consistent_history_mass_scientific",
       Scientific(completion_result.uniform_consistent_history_mass)},
      {"log10_uniform_consistent_history_mass",
       std::log10(completion_result.uniform_consistent_history_mass)},
      {"actual_route_completion_mass_scientific",
       Scientific(actual_route_result.uniform_consistent_history_mass)},
      {"actual_route_uniform_neg_log10_probability",
       std::log10(completion_result.uniform_consistent_history_mass) -
           std::log10(actual_route_result.uniform_consistent_history_mass)},
      {"hidden_card_route_count", card_counts},
      {"uniform_consistent_hidden_hideout_probability",
       hidden_hideout_probabilities},
      {"supported_hidden_cards", supported_cards},
      {"actual_route_supported", true},
      {"actual_route_completable", true},
      {"information_state_bytes", information_state.size()},
      {"information_state_us", information_state_microseconds},
      {"parse_us", Microseconds(parse_start, completion_start)},
      {"route_us", Microseconds(route_start, route_end)},
      {"completion_us", Microseconds(completion_start, completion_end)},
      {"information_state_parse_completion_us",
       information_state_microseconds +
           Microseconds(parse_start, completion_end)},
      {"replay_sample_count", replay_samples},
      {"sample_us", sample_microseconds},
      {"replay_us", replay_microseconds},
      {"sample_replay_checked", replay_samples > 0},
      {"manhunt", manhunt},
      {"sampled_manhunt", sampled_manhunt},
      {"oracle_us", Microseconds(oracle_start, oracle_end)},
      {"route_memo_states", route_result.memo_states},
      {"route_memo_cache_hits", route_result.memo_hits},
      {"route_candidate_transition_evaluations",
       route_result.candidate_transitions},
      {"completion_route_candidate_transition_evaluations",
       completion_result.route_candidate_transitions},
      {"completion_memo_states", completion_result.completion_memo_states},
      {"completion_memo_cache_hits",
       completion_result.completion_memo_hits},
      {"completion_allocation_evaluations",
       completion_result.completion_allocation_evaluations},
      {"max_completion_memo_states_per_route",
       completion_result.max_completion_memo_states},
      {"completion_route_classes",
       completion_result.completion_route_classes},
      {"completion_route_cache_hits",
       completion_result.completion_route_cache_hits},
      {"action_history", state.History()},
  };
  return record;
}

Bucket* CheckpointBucket(const FugitiveState& state, bool* saw_early,
                         bool* saw_middle, bool* saw_late,
                         std::array<Bucket, 3>* buckets) {
  const int route_length = state.route().size() - 1;
  if (state.phase() == Phase::kMarshalGuess && route_length == 2 &&
      !*saw_early) {
    *saw_early = true;
    return &(*buckets)[0];
  }
  if (state.phase() == Phase::kMarshalGuess && route_length == 5 &&
      !*saw_middle) {
    *saw_middle = true;
    return &(*buckets)[1];
  }
  if ((route_length >= 8 || state.phase() == Phase::kManhuntGuess) &&
      !*saw_late) {
    *saw_late = true;
    return &(*buckets)[2];
  }
  return nullptr;
}

bool BucketsFull(const std::array<Bucket, 3>& buckets, int target) {
  return std::all_of(buckets.begin(), buckets.end(), [target](const Bucket& b) {
    return b.information_states.size() >= target;
  });
}

std::int64_t Percentile(std::vector<std::int64_t> values, double quantile) {
  SPIEL_CHECK_FALSE(values.empty());
  std::sort(values.begin(), values.end());
  const int index = std::max<int>(
      0, static_cast<int>(std::ceil(quantile * values.size())) - 1);
  return values[index];
}

json TimingDistribution(const std::vector<std::int64_t>& values) {
  if (values.empty()) return json::object();
  return {
      {"p50", Percentile(values, 0.50)},
      {"p95", Percentile(values, 0.95)},
      {"p99", Percentile(values, 0.99)},
      {"max", *std::max_element(values.begin(), values.end())},
  };
}

void AddMeasurement(const json& record, Measurements* measurements) {
  ++measurements->count;
  measurements->information_state_bytes.push_back(
      record.at("information_state_bytes").get<std::int64_t>());
  measurements->information_state_microseconds.push_back(
      record.at("information_state_us").get<std::int64_t>());
  measurements->parse_microseconds.push_back(
      record.at("parse_us").get<std::int64_t>());
  measurements->route_microseconds.push_back(
      record.at("route_us").get<std::int64_t>());
  measurements->completion_microseconds.push_back(
      record.at("completion_us").get<std::int64_t>());
  measurements->pipeline_microseconds.push_back(
      record.at("information_state_parse_completion_us")
          .get<std::int64_t>());
  measurements->sample_microseconds.push_back(
      record.at("sample_us").get<std::int64_t>());
  measurements->replay_microseconds.push_back(
      record.at("replay_us").get<std::int64_t>());
}

json SummarizeMeasurements(const Measurements& measurements) {
  return {
      {"count", measurements.count},
      {"information_state_bytes",
       TimingDistribution(measurements.information_state_bytes)},
      {"information_state_time_us",
       TimingDistribution(measurements.information_state_microseconds)},
      {"parse_time_us", TimingDistribution(measurements.parse_microseconds)},
      {"route_time_us", TimingDistribution(measurements.route_microseconds)},
      {"completion_time_us",
       TimingDistribution(measurements.completion_microseconds)},
      {"information_state_parse_completion_time_us",
       TimingDistribution(measurements.pipeline_microseconds)},
      {"sample_time_us", TimingDistribution(measurements.sample_microseconds)},
      {"replay_time_us", TimingDistribution(measurements.replay_microseconds)},
  };
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    SPIEL_CHECK_LT(index + 1, argc);
    const std::string flag = argv[index];
    const std::string value = argv[index + 1];
    if (flag == "--mode") {
      options.mode = value;
    } else if (flag == "--samples_per_bucket") {
      options.samples_per_bucket = std::stoi(value);
    } else if (flag == "--seed_start") {
      options.seed_start = std::stoull(value);
    } else if (flag == "--max_seeds") {
      options.max_seeds = std::stoi(value);
    } else if (flag == "--replay_samples") {
      options.replay_samples = std::stoi(value);
    } else if (flag == "--manhunt_completion_calls") {
      options.manhunt_completion_calls = std::stoull(value);
    } else if (flag == "--manhunt_particles") {
      options.manhunt_particles = std::stoi(value);
    } else {
      SpielFatalError("Unknown argument: " + flag);
    }
  }
  SPIEL_CHECK_GT(options.samples_per_bucket, 0);
  SPIEL_CHECK_GT(options.max_seeds, 0);
  SPIEL_CHECK_GE(options.replay_samples, 0);
  SPIEL_CHECK_GE(options.manhunt_particles, 0);
  SPIEL_CHECK_TRUE(options.mode == "fixed" || options.mode == "sweep");
  return options;
}

int RunFixedExperiment(const Options& options) {
  const auto game =
      LoadGame("fugitive", {{"max_rounds", GameParameter(50)}});
  std::array<Bucket, 3> buckets = {
      Bucket{"early"}, Bucket{"middle"}, Bucket{"late"}};
  const Clock::time_point experiment_start = Clock::now();
  int seeds_examined = 0;

  for (; seeds_examined < options.max_seeds &&
         !BucketsFull(buckets, options.samples_per_bucket);
       ++seeds_examined) {
    const std::uint64_t seed = options.seed_start + seeds_examined;
    std::mt19937_64 rng(seed);
    std::unique_ptr<State> state = game->NewInitialState();
    bool saw_early = false;
    bool saw_middle = false;
    bool saw_late = false;

    while (!state->IsTerminal() &&
           !BucketsFull(buckets, options.samples_per_bucket)) {
      const Phase phase = AsFugitive(state.get()).phase();
      if (phase == Phase::kMarshalGuess || phase == Phase::kManhuntGuess) {
        const std::vector<Action> legal = state->LegalActions();
        SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), kCommitAction) ==
                         legal.end());
        Bucket* bucket = CheckpointBucket(
            AsFugitive(state.get()), &saw_early, &saw_middle, &saw_late,
            &buckets);
        if (bucket != nullptr &&
            bucket->information_states.size() < options.samples_per_bucket) {
          const Clock::time_point information_state_start = Clock::now();
          const std::string information_state =
              state->InformationStateString(kMarshalPlayer);
          const std::int64_t information_state_microseconds = Microseconds(
              information_state_start, Clock::now());
          if (bucket->information_states.insert(information_state).second) {
            json record = Evaluate(
                AsFugitive(state.get()), information_state,
                information_state_microseconds, bucket->name, seed,
                options.replay_samples, options.manhunt_completion_calls,
                options.manhunt_particles);
            bucket->route_microseconds.push_back(
                record.at("route_us").get<std::int64_t>());
            bucket->completion_microseconds.push_back(
                record.at("completion_us").get<std::int64_t>());
            bucket->sample_microseconds.push_back(
                record.at("sample_us").get<std::int64_t>());
            bucket->replay_microseconds.push_back(
                record.at("replay_us").get<std::int64_t>());
            std::cout << record.dump() << '\n';
          }
        }
        ApplyMarshalGuess(state.get(), &rng);
      } else {
        AdvanceNonMarshalPhase(state.get(), &rng);
      }
    }
  }

  json route_timing;
  json completion_timing;
  json sample_timing;
  json replay_timing;
  json collected;
  for (const Bucket& bucket : buckets) {
    collected[bucket.name] = bucket.information_states.size();
    if (!bucket.route_microseconds.empty()) {
      route_timing[bucket.name] = {
          {"p50_us", Percentile(bucket.route_microseconds, 0.50)},
          {"p95_us", Percentile(bucket.route_microseconds, 0.95)},
          {"max_us", *std::max_element(bucket.route_microseconds.begin(),
                                        bucket.route_microseconds.end())},
      };
      completion_timing[bucket.name] = {
          {"p50_us", Percentile(bucket.completion_microseconds, 0.50)},
          {"p95_us", Percentile(bucket.completion_microseconds, 0.95)},
          {"max_us",
           *std::max_element(bucket.completion_microseconds.begin(),
                             bucket.completion_microseconds.end())},
      };
      sample_timing[bucket.name] = {
          {"p50_us", Percentile(bucket.sample_microseconds, 0.50)},
          {"p95_us", Percentile(bucket.sample_microseconds, 0.95)},
          {"max_us", *std::max_element(bucket.sample_microseconds.begin(),
                                        bucket.sample_microseconds.end())},
      };
      replay_timing[bucket.name] = {
          {"p50_us", Percentile(bucket.replay_microseconds, 0.50)},
          {"p95_us", Percentile(bucket.replay_microseconds, 0.95)},
          {"max_us", *std::max_element(bucket.replay_microseconds.begin(),
                                        bucket.replay_microseconds.end())},
      };
    }
  }
  const json summary = {
      {"type", "summary"},
      {"mode", "fixed"},
      {"samples_per_bucket", options.samples_per_bucket},
      {"replay_samples", options.replay_samples},
      {"manhunt_completion_calls", options.manhunt_completion_calls},
      {"manhunt_particles", options.manhunt_particles},
      {"seed_start", options.seed_start},
      {"seeds_examined", seeds_examined},
      {"collected", collected},
      {"route_time", route_timing},
      {"completion_time", completion_timing},
      {"sample_time", sample_timing},
      {"replay_time", replay_timing},
      {"elapsed_ms", Microseconds(experiment_start, Clock::now()) / 1000},
      {"complete", BucketsFull(buckets, options.samples_per_bucket)},
  };
  std::cout << summary.dump() << '\n';
  return BucketsFull(buckets, options.samples_per_bucket) ? EXIT_SUCCESS
                                                          : EXIT_FAILURE;
}

int RunSweepExperiment(const Options& options) {
  const auto game =
      LoadGame("fugitive", {{"max_rounds", GameParameter(50)}});
  std::set<std::string> information_states;
  Measurements all;
  Measurements normal_high_support;
  Measurements many_hidden_sprints;
  Measurements normal_high_support_many_hidden_sprints;
  Measurements normal_deep_route;
  std::uint64_t marshal_boundaries = 0;
  std::uint64_t duplicate_boundaries = 0;
  const Clock::time_point experiment_start = Clock::now();

  for (int seed_offset = 0; seed_offset < options.max_seeds; ++seed_offset) {
    const std::uint64_t seed = options.seed_start + seed_offset;
    std::mt19937_64 rng(seed);
    std::unique_ptr<State> state = game->NewInitialState();

    while (!state->IsTerminal()) {
      const Phase phase = AsFugitive(state.get()).phase();
      if (phase == Phase::kMarshalGuess || phase == Phase::kManhuntGuess) {
        ++marshal_boundaries;
        const std::vector<Action> legal = state->LegalActions();
        SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), kCommitAction) ==
                         legal.end());

        const Clock::time_point information_state_start = Clock::now();
        const std::string information_state =
            state->InformationStateString(kMarshalPlayer);
        const std::int64_t information_state_microseconds =
            Microseconds(information_state_start, Clock::now());
        if (information_states.insert(information_state).second) {
          json record = Evaluate(
              AsFugitive(state.get()), information_state,
              information_state_microseconds, "sweep", seed,
              options.replay_samples, options.manhunt_completion_calls,
              options.manhunt_particles);
          AddMeasurement(record, &all);

          const bool normal_guess = phase == Phase::kMarshalGuess;
          const std::uint64_t route_support =
              record.at("route_support_upper_bound").get<std::uint64_t>();
          const int hidden_sprints =
              record.at("hidden_sprint_count").get<int>();
          const int route_length = record.at("route_length").get<int>();
          const bool high_support = normal_guess && route_support >= 1000;
          const bool many_sprints = hidden_sprints >= 10;
          if (high_support) AddMeasurement(record, &normal_high_support);
          if (many_sprints) AddMeasurement(record, &many_hidden_sprints);
          if (high_support && many_sprints) {
            AddMeasurement(record,
                           &normal_high_support_many_hidden_sprints);
          }
          if (normal_guess && route_length >= 10) {
            AddMeasurement(record, &normal_deep_route);
          }
          std::cout << record.dump() << '\n';
        } else {
          ++duplicate_boundaries;
        }
        ApplyMarshalGuess(state.get(), &rng);
      } else {
        AdvanceNonMarshalPhase(state.get(), &rng);
      }
    }
  }

  const json summary = {
      {"type", "summary"},
      {"mode", "sweep"},
      {"replay_samples", options.replay_samples},
      {"manhunt_completion_calls", options.manhunt_completion_calls},
      {"manhunt_particles", options.manhunt_particles},
      {"seed_start", options.seed_start},
      {"seeds_examined", options.max_seeds},
      {"marshal_boundaries", marshal_boundaries},
      {"unique_marshal_states", information_states.size()},
      {"duplicate_marshal_boundaries", duplicate_boundaries},
      {"subsets",
       {
           {"all", SummarizeMeasurements(all)},
           {"normal_guess_route_support_ge_1000",
            SummarizeMeasurements(normal_high_support)},
           {"hidden_sprints_ge_10",
            SummarizeMeasurements(many_hidden_sprints)},
           {"normal_guess_route_support_ge_1000_and_hidden_sprints_ge_10",
            SummarizeMeasurements(normal_high_support_many_hidden_sprints)},
           {"normal_guess_route_length_ge_10",
            SummarizeMeasurements(normal_deep_route)},
       }},
      {"elapsed_ms", Microseconds(experiment_start, Clock::now()) / 1000},
  };
  std::cout << summary.dump() << '\n';
  return all.count > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int RunExperiment(const Options& options) {
  if (options.mode == "sweep") return RunSweepExperiment(options);
  return RunFixedExperiment(options);
}

}  // namespace
}  // namespace fugitive
}  // namespace open_spiel

int main(int argc, char** argv) {
  return open_spiel::fugitive::RunExperiment(
      open_spiel::fugitive::ParseOptions(argc, argv));
}
