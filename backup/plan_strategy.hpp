#pragma once

#include "erl_common/yaml.hpp"

#include <stdexcept>
#include <string>

namespace erl::active_mapping {

    enum class PlanStrategy {
        kMaxScore = 0,
        kMinCost = 1,
        kMaxScoreCostRatio = 2,
    };

    enum class ReplanStrategy {
        kGoalReached = 0,
        kScoreDropped = 1,
        kGoalSeen = 2,
    };
}  // namespace erl::active_mapping

// ReSharper disable CppInconsistentNaming
namespace YAML {
    template<>
    struct convert<erl::active_mapping::PlanStrategy> {
        static Node
        encode(const erl::active_mapping::PlanStrategy &rhs) {
            switch (rhs) {
                case erl::active_mapping::PlanStrategy::kMaxScore:
                    return Node("kMaxScore");
                case erl::active_mapping::PlanStrategy::kMinCost:
                    return Node("kMinCost");
                case erl::active_mapping::PlanStrategy::kMaxScoreCostRatio:
                    return Node("kMaxScoreCostRatio");
                default:
                    throw std::runtime_error(
                        "Unknown plan strategy code: " + std::to_string(static_cast<int>(rhs)));
            }
        }

        static bool
        decode(const Node &node, erl::active_mapping::PlanStrategy &rhs) {
            if (!node.IsScalar()) { return false; }
            if (const std::string &strategy_name = node.as<std::string>();
                strategy_name == "kMaxScore") {
                rhs = erl::active_mapping::PlanStrategy::kMaxScore;
            } else if (strategy_name == "kMinCost") {
                rhs = erl::active_mapping::PlanStrategy::kMinCost;
            } else if (strategy_name == "kMaxScoreCostRatio") {
                rhs = erl::active_mapping::PlanStrategy::kMaxScoreCostRatio;
            } else {
                return false;
            }
            return true;
        }
    };

    template<>
    struct convert<erl::active_mapping::ReplanStrategy> {
        static Node
        encode(const erl::active_mapping::ReplanStrategy &rhs) {
            switch (rhs) {
                case erl::active_mapping::ReplanStrategy::kGoalReached:
                    return Node("kGoalReached");
                case erl::active_mapping::ReplanStrategy::kScoreDropped:
                    return Node("kScoreDropped");
                case erl::active_mapping::ReplanStrategy::kGoalSeen:
                    return Node("kGoalSeen");
                default:
                    throw std::runtime_error(
                        "Unknown replan strategy code: " + std::to_string(static_cast<int>(rhs)));
            }
        }

        static bool
        decode(const Node &node, erl::active_mapping::ReplanStrategy &rhs) {
            if (!node.IsScalar()) { return false; }
            if (const std::string &strategy_name = node.as<std::string>();
                strategy_name == "kGoalReached") {
                rhs = erl::active_mapping::ReplanStrategy::kGoalReached;
            } else if (strategy_name == "kScoreDropped") {
                rhs = erl::active_mapping::ReplanStrategy::kScoreDropped;
            } else if (strategy_name == "kGoalSeen") {
                rhs = erl::active_mapping::ReplanStrategy::kGoalSeen;
            } else {
                return false;
            }
            return true;
        }
    };
}  // namespace YAML

// ReSharper restore CppInconsistentNaming
