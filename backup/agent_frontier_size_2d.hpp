#pragma once

#include "agent_base_2d.hpp"
#include "plan_strategy.hpp"

#include "erl_common/yaml.hpp"
#include "erl_env/environment_2d.hpp"
#include "erl_path_planning/astar.hpp"

namespace erl::active_mapping {
    class AgentFrontierSize2D : public AgentBase2D {
    public:
        struct Setting : public common::Yamlable<Setting> {
            struct Frontier : public Yamlable<Frontier> {
                bool clean_at_first = true;
                int approx_iters = 6;
                int min_size = 20;
                int max_num_frontiers = 10;
                bool sampling_goals = true;   // if false, use the mean of frontier points as goal
                double sampling_ratio = 0.1;  // ratio of frontier points to sample
                int max_num_goals_per_frontier = 10;  // max number of goals to sample per frontier
            };

            struct Plan : public Yamlable<Plan> {
                PlanStrategy strategy = PlanStrategy::kMaxScore;
                int max_retries = 5;
                Eigen::Vector2d goal_tolerance = Eigen::Vector2d(0.25, 0.25);
                std::shared_ptr<env::Environment2D::Setting> env =
                    std::make_shared<env::Environment2D::Setting>();
                std::shared_ptr<search_planning::astar::AStar::Setting> astar =
                    std::make_shared<search_planning::astar::AStar::Setting>();
            };

            uint64_t random_seed = 0;
            std::shared_ptr<Frontier> frontier = std::make_shared<Frontier>();
            std::shared_ptr<Plan> plan = std::make_shared<Plan>();
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

    public:
        explicit AgentFrontierSize2D(const std::shared_ptr<Setting>& setting)
            : m_setting_(setting) {}

        bool
        Plan(
            Eigen::Vector3d agent_pose,
            const std::shared_ptr<geometry::LogOddMap2D>& log_odd_map,
            Eigen::Vector3d& goal,
            double& goal_score,
            bool& random_goal,
            Eigen::Matrix3Xd& path,
            double& path_cost) override;

    private:
        [[nodiscard]] std::vector<Eigen::Matrix2Xi>
        GetGridFrontiers(const std::shared_ptr<geometry::LogOddMap2D>& log_odd_map) const;

        [[nodiscard]] std::pair<std::vector<double>, std::vector<Eigen::Vector2d>>
        GetMetricGoals(
            const Eigen::Vector3d& agent_pose,
            const std::shared_ptr<geometry::LogOddMap2D>& log_odd_map,
            const std::vector<Eigen::Matrix2Xi>& grid_frontiers) const;
    };
}  // namespace erl::active_mapping

// ReSharper disable CppInconsistentNaming
namespace YAML {
    template<>
    struct convert<erl::active_mapping::AgentFrontierSize2D::Setting::Frontier> {
        static Node
        encode(const erl::active_mapping::AgentFrontierSize2D::Setting::Frontier& rhs) {
            Node node;
            node["clean_at_first"] = rhs.clean_at_first;
            node["approx_iters"] = rhs.approx_iters;
            node["min_size"] = rhs.min_size;
            node["max_num_frontiers"] = rhs.max_num_frontiers;
            node["sampling_goals"] = rhs.sampling_goals;
            node["sampling_ratio"] = rhs.sampling_ratio;
            node["max_num_goals_per_frontier"] = rhs.max_num_goals_per_frontier;
            return node;
        }

        static bool
        decode(const Node& node, erl::active_mapping::AgentFrontierSize2D::Setting::Frontier& rhs) {
            if (!node.IsMap()) { return false; }
            rhs.clean_at_first = node["clean_at_first"].as<bool>();
            rhs.approx_iters = node["approx_iters"].as<int>();
            rhs.min_size = node["min_size"].as<int>();
            rhs.max_num_frontiers = node["max_num_frontiers"].as<int>();
            rhs.sampling_goals = node["sampling_goals"].as<bool>();
            rhs.sampling_ratio = node["sampling_ratio"].as<double>();
            rhs.max_num_goals_per_frontier = node["max_num_goals_per_frontier"].as<int>();
            return true;
        }
    };

    template<>
    struct convert<erl::active_mapping::AgentFrontierSize2D::Setting::Plan> {
        static Node
        encode(const erl::active_mapping::AgentFrontierSize2D::Setting::Plan& rhs) {
            Node node;
            node["strategy"] = rhs.strategy;
            node["max_retries"] = rhs.max_retries;
            node["goal_tolerance"] = rhs.goal_tolerance;
            node["env"] = rhs.env;
            node["astar"] = rhs.astar;
            return node;
        }

        static bool
        decode(const Node& node, erl::active_mapping::AgentFrontierSize2D::Setting::Plan& rhs) {
            if (!node.IsMap()) { return false; }
            rhs.strategy = node["strategy"].as<erl::active_mapping::PlanStrategy>();
            rhs.max_retries = node["max_retries"].as<int>();
            rhs.goal_tolerance = node["goal_tolerance"].as<Eigen::Vector2d>();
            rhs.env = node["env"].as<std::shared_ptr<erl::env::Environment2D::Setting>>();
            rhs.astar =
                node["astar"].as<std::shared_ptr<erl::search_planning::astar::AStar::Setting>>();
            return true;
        }
    };

    template<>
    struct convert<erl::active_mapping::AgentFrontierSize2D::Setting> {
        static Node
        encode(const erl::active_mapping::AgentFrontierSize2D::Setting& rhs) {
            Node node;
            node["random_seed"] = rhs.random_seed;
            node["frontier"] = rhs.frontier;
            node["plan"] = rhs.plan;
            return node;
        }

        static bool
        decode(const Node& node, erl::active_mapping::AgentFrontierSize2D::Setting& rhs) {
            if (!node.IsMap()) { return false; }
            rhs.random_seed = node["random_seed"].as<uint64_t>();
            rhs.frontier = node["frontier"]
                               .as<std::shared_ptr<
                                   erl::active_mapping::AgentFrontierSize2D::Setting::Frontier>>();
            rhs.plan =
                node["plan"]
                    .as<std::shared_ptr<erl::active_mapping::AgentFrontierSize2D::Setting::Plan>>();
            return true;
        }
    };
}  // namespace YAML

// ReSharper restore CppInconsistentNaming
