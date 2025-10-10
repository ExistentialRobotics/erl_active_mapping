#pragma once

#include "agent_base.hpp"

#include "erl_common/random.hpp"
#include "erl_env/cost.hpp"
#include "erl_env/environment_2d.hpp"
#include "erl_geometry/log_odd_map_2d.hpp"
#include "erl_path_planning/astar.hpp"
#include "erl_path_planning/heuristic.hpp"
#include "erl_path_planning/search_planning_interface.hpp"

#include <queue>

namespace erl::active_mapping::frontier_based {

    enum class PlanStrategy {
        kMaxScore = 0,
        kMinPathLength = 1,
        kMaxScorePathLengthRatio = 2,
    };

    enum class ReplanStrategy {
        kFrontierSeen = 0,
        kGoalReached = 1,
    };

    namespace grid_frontiers {
        struct GridFrontierSetting : public common::Yamlable<GridFrontierSetting> {
            bool clean_at_first = true;
            int approx_iters = 6;
            long min_size = 20;
            std::size_t max_num_frontiers = 10;
            bool sample_goals = true;
            float sampling_ratio = 0.1f;
            long max_num_goals_per_frontier = 5;
        };
    }  // namespace grid_frontiers

    template<typename Dtype, typename Observation = Eigen::Matrix2X<Dtype>>
    class AgentFrontierBasedGrid2D : public AgentBase<Dtype, 2, Observation> {

    public:
        using Super = AgentBase<Dtype, 2, Observation>;
        using Pose = typename Super::Pose;    // sensor pose
        using State = typename Super::State;  // env state used in path planning
        using MetricState = typename State::MetricState;
        using LogOddMap = geometry::LogOddMap2D<Dtype>;
        using GridMapInfo = typename LogOddMap::GridMapInfo;
        using Matrix2X = Eigen::Matrix2X<Dtype>;
        using Environment2D = env::Environment2D<Dtype>;
        using EnvSetting = typename Environment2D::Setting;
        using EnvCost = env::EuclideanDistanceCost<Dtype, 2>;
        using PlanningInterface = path_planning::SearchPlanningInterface<Dtype, 2>;
        using Astar = path_planning::astar::AStar<Dtype, 2>;
        using AstarSetting = path_planning::astar::AstarSetting<Dtype>;
        using Path = typename Super::Path;

        struct Setting : public common::Yamlable<Setting> {
            PlanStrategy plan_strategy = PlanStrategy::kMaxScore;
            ReplanStrategy replan_strategy = ReplanStrategy::kFrontierSeen;
            Dtype frontier_seen_ratio = 0.8f;  // replan when this ratio of frontier is seen
            Dtype goal_tolerance = 0.25f;      // in meters
            bool check_possible_collision = true;
            long collision_check_window_size = 10;

            std::shared_ptr<typename LogOddMap::Setting> log_odd_map =
                std::make_shared<typename LogOddMap::Setting>();

            grid_frontiers::GridFrontierSetting frontier{};

            std::shared_ptr<EnvSetting> env = std::make_shared<EnvSetting>();

            int env_max_step_size = 1;
            bool env_allow_diagonal = true;

            std::shared_ptr<AstarSetting> astar = std::make_shared<AstarSetting>();
            long max_random_planning_trials = 100;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting) {
                    YAML::Node node;
                    ERL_YAML_SAVE_ENUM_ATTR(
                        node,
                        setting,
                        plan_strategy,
                        {{"kMaxScore", PlanStrategy::kMaxScore},
                         {"kMinPathLength", PlanStrategy::kMinPathLength},
                         {"kMaxScorePathLengthRatio", PlanStrategy::kMaxScorePathLengthRatio}});
                    ERL_YAML_SAVE_ENUM_ATTR(
                        node,
                        setting,
                        replan_strategy,
                        {{"kFrontierSeen", ReplanStrategy::kFrontierSeen},
                         {"kGoalReached", ReplanStrategy::kGoalReached}});
                    ERL_YAML_SAVE_ATTR(node, setting, frontier_seen_ratio);
                    ERL_YAML_SAVE_ATTR(node, setting, goal_tolerance);
                    ERL_YAML_SAVE_ATTR(node, setting, check_possible_collision);
                    ERL_YAML_SAVE_ATTR(node, setting, collision_check_window_size);
                    ERL_YAML_SAVE_ATTR(node, setting, log_odd_map);
                    ERL_YAML_SAVE_ATTR(node, setting, frontier);
                    ERL_YAML_SAVE_ATTR(node, setting, env);
                    ERL_YAML_SAVE_ATTR(node, setting, env_max_step_size);
                    ERL_YAML_SAVE_ATTR(node, setting, env_allow_diagonal);
                    ERL_YAML_SAVE_ATTR(node, setting, astar);
                    ERL_YAML_SAVE_ATTR(node, setting, max_random_planning_trials);
                    return node;
                }

                static bool
                decode(const YAML::Node &node, Setting &setting) {
                    if (!node.IsMap()) { return false; }
                    ERL_YAML_LOAD_ENUM_ATTR(
                        node,
                        setting,
                        plan_strategy,
                        {{"kMaxScore", PlanStrategy::kMaxScore},
                         {"kMinPathLength", PlanStrategy::kMinPathLength},
                         {"kMaxScorePathLengthRatio", PlanStrategy::kMaxScorePathLengthRatio}});
                    ERL_YAML_LOAD_ENUM_ATTR(
                        node,
                        setting,
                        replan_strategy,
                        {{"kFrontierSeen", ReplanStrategy::kFrontierSeen},
                         {"kGoalReached", ReplanStrategy::kGoalReached}});
                    ERL_YAML_LOAD_ATTR(node, setting, frontier_seen_ratio);
                    ERL_YAML_LOAD_ATTR(node, setting, goal_tolerance);
                    ERL_YAML_LOAD_ATTR(node, setting, check_possible_collision);
                    ERL_YAML_LOAD_ATTR(node, setting, collision_check_window_size);
                    if (!ERL_YAML_LOAD_ATTR(node, setting, log_odd_map)) { return false; }
                    if (!ERL_YAML_LOAD_ATTR(node, setting, frontier)) { return false; }
                    if (!ERL_YAML_LOAD_ATTR(node, setting, env)) { return false; }
                    ERL_YAML_LOAD_ATTR(node, setting, env_max_step_size);
                    ERL_YAML_LOAD_ATTR(node, setting, env_allow_diagonal);
                    if (!ERL_YAML_LOAD_ATTR(node, setting, astar)) { return false; }
                    ERL_YAML_LOAD_ATTR(node, setting, max_random_planning_trials);
                    return true;
                }
            };
        };

        struct Frontier {
            std::vector<State> points;                // points (states) that make up the frontier
            std::vector<MetricState> goals{};         // goal states
            std::vector<std::size_t> goal_indices{};  // Indices of goals
            Dtype score = 0;                          // Score of the frontier
        };

    private:
        std::shared_ptr<Setting> m_setting_;
        std::shared_ptr<GridMapInfo> m_grid_map_info_;
        std::shared_ptr<LogOddMap> m_log_odd_map_;
        std::vector<Frontier> m_frontiers_;
        long m_best_frontier_index_ = 0;
        std::shared_ptr<Environment2D> m_env_;
        bool m_env_outdated_ = true;
        std::shared_ptr<EnvCost> m_env_cost_ = std::make_shared<EnvCost>();
        cv::Mat m_env_cost_map_;
        Path m_current_path_{};
        long m_current_wp_idx_ = 0;

    public:
        AgentFrontierBasedGrid2D(
            std::shared_ptr<Setting> setting,
            std::shared_ptr<GridMapInfo> grid_map_info)
            : m_setting_(NotNull(std::move(setting), true, "setting is nullptr.")),
              m_grid_map_info_(
                  NotNull(std::move(grid_map_info), true, "grid_map_info is nullptr.")),
              m_log_odd_map_(
                  std::make_shared<LogOddMap>(
                      m_setting_->log_odd_map,
                      m_grid_map_info_,
                      m_setting_->env->robot_metric_contour)) {
            if (m_setting_->env->motions.empty()) {
                m_setting_->env->SetGridMotionPrimitive(
                    m_setting_->env_max_step_size,
                    m_setting_->env_allow_diagonal);
            }
        }

        [[nodiscard]] std::shared_ptr<LogOddMap>
        GetLogOddMap() const {
            return m_log_odd_map_;
        }

        [[nodiscard]] std::shared_ptr<GridMapInfo>
        GetGridMapInfo() const {
            return m_grid_map_info_;
        }

        [[nodiscard]] const std::vector<Frontier> &
        GetFrontiers() const {
            return m_frontiers_;
        }

        [[nodiscard]] long
        GetBestFrontierIndex() const {
            return m_best_frontier_index_;
        }

        [[nodiscard]] const Path &
        GetCurrentPath() const {
            return m_current_path_;
        }

        [[nodiscard]] long
        GetCurrentWpIndex() const {
            return m_current_wp_idx_;
        }

        void
        SetCurrentWpIndex(long idx) {
            if (idx < 0 || idx >= m_current_path_.cols()) {
                ERL_WARN("Invalid waypoint index: {}.", idx);
                return;
            }
            m_current_wp_idx_ = idx;
        }

        void
        Step(const Pose &pose, const Observation &observation) override {
            Dtype theta = std::atan2(pose(1, 0), pose(0, 0));
            m_log_odd_map_->Update(pose.col(2), theta, observation /*points*/);
            m_env_outdated_ = true;
        }

        [[nodiscard]] const Path &
        Plan(const State &state) override {
            m_current_wp_idx_ = 0;

            (void) ExtractFrontiers();
            UpdateEnv();

            if (m_frontiers_.empty()) {
                ERL_INFO("No frontier found.");
                return RandomPlan(state);
            }

            if (GetPathToBestFrontier(state, m_current_path_) >= 0) { return m_current_path_; }

            (void) RandomPlan(state);

            return m_current_path_;
        }

        [[nodiscard]] const Path &
        RandomPlan(const State &state) override {
            long trial = 0;
            while (m_setting_->max_random_planning_trials < 0 ||
                   trial < m_setting_->max_random_planning_trials) {
                trial++;
                State goal = m_env_->SampleValidStates(1)[0];
                auto planning_interface = std::make_shared<PlanningInterface>(
                    m_env_,
                    state.metric,
                    goal.metric,
                    MetricState::Constant(m_setting_->goal_tolerance / std::sqrt(2.0f)),
                    static_cast<Dtype>(0.0f));

                auto astar_output = Astar(planning_interface, m_setting_->astar).Plan();
                if (astar_output->plan_records.empty()) { continue; }  // no path found
                m_current_path_ = astar_output->plan_records[astar_output->latest_plan_itr].path;
                if (m_current_path_.cols() > 0) { break; }  // path found
            }

            if (m_current_path_.cols() == 0) {
                ERL_WARN("Failed to find a path to any frontier or random goal.");
            }

            ERL_INFO("Generate a random path with {} waypoints.", m_current_path_.cols());
            return m_current_path_;
        }

        [[nodiscard]] bool
        ShouldReplan(const State &state) override {
            for (long i = m_current_wp_idx_; i < m_current_path_.cols(); ++i) {
                // find the first waypoint that is close enough to the current state
                Dtype dist = (m_current_path_.col(i) - state.metric).norm();
                if (dist < m_setting_->goal_tolerance) {
                    m_current_wp_idx_ = i;
                    break;
                }
            }

            if (m_current_wp_idx_ + 1 >= m_current_path_.cols()) {
                // deviation? the plan is done? replan
                ERL_INFO("Replan because the current path is done.");
                return true;
            }

            if (m_setting_->check_possible_collision) {
                UpdateEnv();
                State wp_state;
                const long imax = std::min(
                    m_current_wp_idx_ + m_setting_->collision_check_window_size,
                    m_current_path_.cols());
                for (long i = m_current_wp_idx_; i < imax; ++i) {
                    wp_state.metric = m_current_path_.col(i);
                    wp_state.grid = m_env_->MetricToGrid(wp_state.metric);
                    // check if any waypoint is in collision
                    if (!m_env_->IsValidState(wp_state)) {
                        ERL_INFO("Potential collision detected.");
                        return true;
                    }
                }
            }

            MetricState goal = m_current_path_.template rightCols<1>();

            if (m_setting_->replan_strategy == ReplanStrategy::kGoalReached ||
                m_best_frontier_index_ < 0) {
                if ((state.metric - goal).norm() < m_setting_->goal_tolerance) {
                    ERL_INFO("Goal reached.");
                    return true;
                }
                return false;
            }

            if (m_setting_->replan_strategy == ReplanStrategy::kFrontierSeen) {
                cv::Mat free_mask;
                m_log_odd_map_->GetCleanedFreeMask().copyTo(free_mask);
                cv::erode(
                    free_mask,
                    free_mask,
                    cv::getStructuringElement(cv::MorphShapes::MORPH_CROSS, cv::Size{3, 3}));
                const Frontier &frontier = m_frontiers_[m_best_frontier_index_];
                long n_free = 0;
                for (const State &p: frontier.points) {
                    if (free_mask.at<uint8_t>(p.grid[0], p.grid[1])) { n_free++; }
                }
                const Dtype th = m_setting_->frontier_seen_ratio;
                if (static_cast<Dtype>(n_free) / static_cast<Dtype>(frontier.points.size()) > th) {
                    ERL_INFO("Frontier seen.");
                    return true;
                }
                return false;
            }
            ERL_WARN("Unknown ReplanStrategy.");
            return false;
        }

        [[nodiscard]] std::vector<Frontier> &
        ExtractFrontiers() {
            grid_frontiers::GridFrontierSetting &f_setting = m_setting_->frontier;
            std::vector<Eigen::Matrix2Xi> grid_frontiers =
                m_log_odd_map_->GetFrontiers(f_setting.clean_at_first, f_setting.approx_iters);

            // remove small frontiers and sort frontiers by size in descending order
            struct Compare {
                bool
                operator()(
                    const std::pair<std::size_t, long> &lhs,
                    const std::pair<std::size_t, long> &rhs) const {
                    return lhs.second < rhs.second;
                }
            };

            const std::size_t num_frontiers = grid_frontiers.size();

            std::priority_queue<
                std::pair<std::size_t, long>,               // element
                std::vector<std::pair<std::size_t, long>>,  // sequence
                Compare>
                frontier_indices;

            for (std::size_t i = 0; i < num_frontiers; ++i) {
                long frontier_size = grid_frontiers[i].cols();
                if (frontier_size < f_setting.min_size) { continue; }
                frontier_indices.emplace(i, frontier_size);
            }

            m_frontiers_.clear();
            if (frontier_indices.empty()) { return m_frontiers_; }

            while (m_frontiers_.size() < f_setting.max_num_frontiers && !frontier_indices.empty()) {
                Eigen::Matrix2Xi &grid_frontier = grid_frontiers[frontier_indices.top().first];
                frontier_indices.pop();

                // convert to metric frontier
                Frontier frontier;
                for (long i = 0; i < grid_frontier.cols(); ++i) {
                    auto p = grid_frontier.col(i);
                    frontier.points.emplace_back(
                        Eigen::Vector2<Dtype>(
                            m_grid_map_info_->GridToMeterAtDim(p[0], 0),
                            m_grid_map_info_->GridToMeterAtDim(p[1], 1)),
                        p);
                }

                // compute score
                frontier.score = static_cast<Dtype>(frontier.points.size());

                // compute goals
                if (f_setting.sample_goals) {
                    long n_goals = std::min(
                        static_cast<long>(
                            f_setting.sampling_ratio * static_cast<Dtype>(frontier.points.size())),
                        f_setting.max_num_goals_per_frontier);
                    n_goals = std::max(n_goals, 1l);
                    frontier.goal_indices.resize(frontier.points.size());
                    std::iota(frontier.goal_indices.begin(), frontier.goal_indices.end(), 0);
                    std::shuffle(
                        frontier.goal_indices.begin(),
                        frontier.goal_indices.end(),
                        common::g_random_engine);
                    frontier.goal_indices.resize(n_goals);
                    for (auto &idx: frontier.goal_indices) {
                        frontier.goals.emplace_back(frontier.points[idx].metric);
                    }
                } else {
                    MetricState mean = MetricState::Zero();
                    for (auto &pt: frontier.points) { mean += pt.metric; }
                    mean /= static_cast<Dtype>(frontier.points.size());
                    frontier.goals.emplace_back(mean);
                }

                m_frontiers_.emplace_back(std::move(frontier));
            }
            return m_frontiers_;
        }

        long
        GetPathToBestFrontier(const State &state, Path &path) {
            m_best_frontier_index_ = -1;

            if (m_frontiers_.empty()) { return m_best_frontier_index_; }

            std::vector<MetricState> goals;
            std::vector<Dtype> terminal_costs;
            std::vector<long> goal_frontier_indices;
            goals.reserve(m_frontiers_.size());
            terminal_costs.reserve(m_frontiers_.size());
            goal_frontier_indices.reserve(m_frontiers_.size());

            switch (m_setting_->plan_strategy) {
                case PlanStrategy::kMaxScore: {
                    Dtype score = m_frontiers_[0].score;  // the first one has the largest score
                    goals = m_frontiers_[0].goals;
                    terminal_costs.resize(goals.size(), -score);
                    goal_frontier_indices.resize(goals.size(), 0);
                    // there might be other frontiers with the same max score
                    for (std::size_t i = 1; i < m_frontiers_.size(); ++i) {
                        const Frontier &frontier = m_frontiers_[i];
                        if (frontier.score < score) { break; }  // frontier list is sorted
                        goals.insert(goals.end(), frontier.goals.begin(), frontier.goals.end());
                        terminal_costs.insert(terminal_costs.end(), frontier.goals.size(), -score);
                        goal_frontier_indices.insert(
                            goal_frontier_indices.end(),
                            frontier.goals.size(),
                            i);
                    }
                    break;
                }
                case PlanStrategy::kMinPathLength:
                case PlanStrategy::kMaxScorePathLengthRatio: {
                    for (std::size_t i = 0; i < m_frontiers_.size(); ++i) {
                        const Frontier &frontier = m_frontiers_[i];
                        goals.insert(goals.end(), frontier.goals.begin(), frontier.goals.end());
                        terminal_costs.insert(
                            terminal_costs.end(),
                            frontier.goals.size(),
                            -frontier.score);
                        goal_frontier_indices.insert(
                            goal_frontier_indices.end(),
                            frontier.goals.size(),
                            static_cast<long>(i));
                    }
                    break;
                }
                default:
                    ERL_WARN("Unknown PlanStrategy.");
                    return m_best_frontier_index_;
            }

            if (goals.empty()) { return m_best_frontier_index_; }

            auto planning_interface = std::make_shared<PlanningInterface>(
                m_env_,
                state.metric,
                goals,
                std::vector<MetricState>{
                    MetricState::Constant(m_setting_->goal_tolerance / std::sqrt(2.0f))},
                terminal_costs);

            Astar astar(planning_interface, m_setting_->astar);

            if (m_setting_->plan_strategy == PlanStrategy::kMaxScore) {
                auto astar_output = astar.Plan();
                if (astar_output->plan_records.empty()) { return m_best_frontier_index_; }
                auto &plan_record = astar_output->plan_records[astar_output->latest_plan_itr];
                path = plan_record.path;
                m_best_frontier_index_ = goal_frontier_indices[plan_record.goal_index];
                return m_best_frontier_index_;
            }

            // find all reachable goals
            std::shared_ptr<typename Astar::Output_t> astar_output = astar.Plan();
            std::size_t n_goals_reached = astar_output->plan_records.size();
            while (true) {
                astar_output = astar.Plan();
                if (n_goals_reached == astar_output->plan_records.size()) { break; }  // no new goal
                n_goals_reached = astar_output->plan_records.size();
            }
            if (n_goals_reached == 0) { return m_best_frontier_index_; }  // no reachable goal

            if (m_setting_->plan_strategy == PlanStrategy::kMinPathLength) {
                long best_plan_itr = -1;
                long best_goal_index = -1;
                Dtype min_path_length = std::numeric_limits<Dtype>::max();
                for (auto &[plan_itr, record]: astar_output->plan_records) {
                    Dtype path_length = record.cost - terminal_costs[record.goal_index];
                    if (path_length < min_path_length) {
                        min_path_length = path_length;
                        best_plan_itr = plan_itr;
                        best_goal_index = record.goal_index;
                    }
                }
                path = astar_output->plan_records[best_plan_itr].path;
                m_best_frontier_index_ = goal_frontier_indices[best_goal_index];
                return m_best_frontier_index_;
            }

            // PlanStrategy::kMaxScorePathLengthRatio
            long best_plan_itr = -1;
            long best_goal_index = -1;
            Dtype max_score_path_length_ratio = -std::numeric_limits<Dtype>::max();
            for (auto &[plan_itr, record]: astar_output->plan_records) {
                Dtype path_length = record.cost - terminal_costs[record.goal_index] + 1e-6f;
                if (path_length < m_setting_->goal_tolerance) { continue; }  // avoid close goals
                Dtype score_path_length_ratio = -terminal_costs[record.goal_index] / path_length;
                if (score_path_length_ratio > max_score_path_length_ratio) {
                    max_score_path_length_ratio = score_path_length_ratio;
                    best_plan_itr = plan_itr;
                    best_goal_index = record.goal_index;
                }
            }
            path = astar_output->plan_records[best_plan_itr].path;
            m_best_frontier_index_ = goal_frontier_indices[best_goal_index];
            return m_best_frontier_index_;
        }

    private:
        void
        UpdateEnv() {
            if (m_env_outdated_) {
                m_env_cost_map_ = 1 - m_log_odd_map_->GetCleanedFreeMask();
                m_env_outdated_ = false;

                m_env_ = std::make_shared<Environment2D>(
                    m_grid_map_info_,
                    m_env_cost_map_,
                    m_setting_->env,
                    m_env_cost_);
            }
        }
    };

    extern template class AgentFrontierBasedGrid2D<float>;
    extern template class AgentFrontierBasedGrid2D<double>;

}  // namespace erl::active_mapping::frontier_based

template<>
struct YAML::convert<erl::active_mapping::frontier_based::AgentFrontierBasedGrid2D<float>::Setting>
    : public erl::active_mapping::frontier_based::AgentFrontierBasedGrid2D<
          float>::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::active_mapping::frontier_based::AgentFrontierBasedGrid2D<double>::Setting>
    : public erl::active_mapping::frontier_based::AgentFrontierBasedGrid2D<
          double>::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting> {
    static Node
    encode(const erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting &setting);

    static bool
    decode(
        const YAML::Node &node,
        erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting &setting);
};
