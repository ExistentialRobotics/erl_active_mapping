#include "agent_frontier_size_2d.hpp"

#include "erl_common/random.hpp"
#include "erl_path_planning/astar.hpp"

#include <queue>

namespace erl::active_mapping {

    bool
    AgentFrontierSize2D::Plan(
        Eigen::Vector3d agent_pose,
        const std::shared_ptr<geometry::LogOddMap2D> &log_odd_map,
        Eigen::Vector3d &goal,
        double &goal_score,
        bool &random_goal,
        Eigen::Matrix3Xd &path,
        double &path_cost) {

        const auto setting = std::static_pointer_cast<Setting>(m_setting_);

        // free_mask: 1: free, 0: occupied
        // collision_map: 1: occupied, 0: free
        const cv::Mat collision_map =
            1 - log_odd_map->GetCleanedFreeMask();  // rows are x, cols are y, row-major, compatible
                                                    // with common::GridMapUnsigned2D

        // get frontiers and goals
        const std::vector<Eigen::Matrix2Xi> grid_frontiers = GetGridFrontiers(log_odd_map);
        std::vector<double> goal_scores;
        std::vector<Eigen::Vector2d> metric_goals;
        if (m_setting_->plan->strategy == PlanStrategy::kMaxScore) {
            std::tie(goal_scores, metric_goals) = GetMetricGoals(
                agent_pose,
                log_odd_map,
                {grid_frontiers[0]});  // only use the largest frontier
        } else {
            std::tie(goal_scores, metric_goals) =
                GetMetricGoals(agent_pose, log_odd_map, grid_frontiers);
        }

        // get path to goals
        const auto environment = std::make_shared<env::Environment2D>(
            std::const_pointer_cast<common::GridMapInfo2D>(log_odd_map->GetGridMapInfo()),
            collision_map,
            setting->plan->env);

        std::vector<std::tuple<int, double, Eigen::Matrix2Xd>>
            paths;  // goal_index, path_length, path

        std::vector<Eigen::VectorXd> metric_goals_coords;
        metric_goals_coords.reserve(metric_goals.size());
        std::transform(
            metric_goals_coords.begin(),
            metric_goals_coords.end(),
            std::back_inserter(metric_goals_coords),
            [](const Eigen::Vector2d &goal) -> Eigen::VectorXd { return goal; });

        for (search_planning::astar::AStar astar(
                 std::make_shared<search_planning::PlanningInterface>(
                     environment,           // environment
                     agent_pose.head<2>(),  // metric_start_coords
                     metric_goals_coords,   // metric_goal_coords
                     std::vector{
                         Eigen::VectorXd(setting->plan->goal_tolerance)}));  // goal_tolerance
             paths.size() < metric_goals.size();) {
            std::shared_ptr<search_planning::astar::Output> astar_output = astar.Plan();
            if (astar_output == nullptr || astar_output->goal_index < 0) { break; }
            if (!paths.empty() && std::get<0>(paths.back()) == astar_output->goal_index) {
                break;
            }  // avoid duplicate paths
            paths.emplace_back(astar_output->goal_index, astar_output->cost, astar_output->path);
        }

        // pick the best path
        if (paths.size() > 1) {
            switch (setting->plan->strategy) {
                case PlanStrategy::kMaxScore:   // all paths have the same score since all goals are
                                                // sampled from the same frontier, find the shortest
                                                // path
                case PlanStrategy::kMinCost: {  // find the shortest path
                    std::sort(paths.begin(), paths.end(), [](const auto &lhs, const auto &rhs) {
                        return std::get<1>(lhs) < std::get<1>(rhs);
                    });
                    break;
                }
                case PlanStrategy::kMaxScoreCostRatio: {
                    std::vector<double> ratios;
                    ratios.reserve(paths.size());
                    std::transform(
                        paths.begin(),
                        paths.end(),
                        std::back_inserter(ratios),
                        [&goal_scores](const auto &path) {
                            return goal_scores[std::get<0>(path)] / std::get<1>(path);
                        });
                    const int best_path_index = std::distance(
                        ratios.begin(),
                        std::max_element(ratios.begin(), ratios.end()));
                    std::swap(paths[0], paths[best_path_index]);
                    break;
                }
            }
        } else if (paths.empty()) {
            for (int retries = 0; retries < setting->plan->max_retries; ++retries) {
                const auto planning_interface =
                    std::make_shared<search_planning::PlanningInterface>(
                        environment,                                   // environment
                        agent_pose.head<2>(),                          // metric_start_coords
                        environment->SampleValidStates(1)[0]->metric,  // metric_goal_coords
                        setting->plan->goal_tolerance);                // goal_tolerance
                if (std::shared_ptr<search_planning::astar::Output> astar_output =
                        search_planning::astar::AStar(planning_interface).Plan();
                    astar_output != nullptr && astar_output->goal_index >= 0) {
                    paths.emplace_back(
                        astar_output->goal_index,
                        astar_output->cost,
                        astar_output->path);
                    random_goal = true;
                    break;
                }
            }
        }

        if (paths.empty()) { return false; }  // no path found

        // extract the best path
        const int goal_index = std::get<0>(paths[0]);
        path_cost = std::get<1>(paths[0]);
        goal_score = goal_scores[goal_index];
        goal.head<2>() = metric_goals[goal_index];
        goal[2] = 0.0;
        const Eigen::Matrix2Xd &path_2d = std::get<2>(paths[0]);
        path.resize(3, path_2d.cols());
        path(0, 0) = path_2d(0, 0);
        path(1, 0) = path_2d(1, 0);
        path(2, 0) = std::atan2(path_2d(1, 0) - agent_pose[1], path_2d(0, 0) - agent_pose[0]);
        for (long i = 1; i < path_2d.cols(); ++i) {
            path(0, i) = path_2d(0, i);
            path(1, i) = path_2d(1, i);
            path(2, i) =
                std::atan2(path_2d(1, i) - path_2d(1, i - 1), path_2d(0, i) - path_2d(0, i - 1));
        }
        return true;
    }

    std::vector<Eigen::Matrix2Xi>
    AgentFrontierSize2D::GetGridFrontiers(
        const std::shared_ptr<geometry::LogOddMap2D> &log_odd_map) const {
        const auto setting = std::static_pointer_cast<Setting>(m_setting_);
        const std::vector<Eigen::Matrix2Xi> frontiers = log_odd_map->GetFrontiers(
            setting->frontier->clean_at_first,
            setting->frontier->approx_iters);
        const std::size_t num_frontiers = frontiers.size();
        if (num_frontiers == 0) { return {}; }

        std::vector<Eigen::Matrix2Xi> grid_frontiers;
        grid_frontiers.reserve(setting->frontier->max_num_frontiers);

        // Filter out small frontiers and sort frontiers by size.
        struct Compare {
            bool
            operator()(
                const std::pair<std::size_t, long> &lhs,
                const std::pair<std::size_t, long> &rhs) const {
                return lhs.second < rhs.second;
            }
        };

        std::priority_queue<
            std::pair<std::size_t, long>,
            std::vector<std::pair<std::size_t, long>>,
            Compare>
            large_frontier_indices;
        for (std::size_t i = 0; i < num_frontiers; ++i) {
            large_frontier_indices.emplace(i, frontiers[i].cols());
        }

        while (grid_frontiers.size() < grid_frontiers.capacity() &&
               !large_frontier_indices.empty()) {
            grid_frontiers.push_back(frontiers[large_frontier_indices.top().first]);
            large_frontier_indices.pop();
            if (grid_frontiers.back().cols() < setting->frontier->min_size) { break; }
        }

        return grid_frontiers;
    }

    std::pair<std::vector<double>, std::vector<Eigen::Vector2d>>
    AgentFrontierSize2D::GetMetricGoals(
        const Eigen::Vector3d &agent_pose,
        const std::shared_ptr<geometry::LogOddMap2D> &log_odd_map,
        const std::vector<Eigen::Matrix2Xi> &grid_frontiers) const {
        if (grid_frontiers.empty()) { return {}; }

        const std::shared_ptr<const common::GridMapInfo2D> grid_map_info =
            log_odd_map->GetGridMapInfo();

        std::vector<double> goal_scores;
        std::vector<Eigen::Vector2d> metric_goals;
        goal_scores.reserve(
            m_setting_->frontier->max_num_goals_per_frontier * grid_frontiers.size());
        metric_goals.reserve(goal_scores.capacity());

        for (const Eigen::Matrix2Xi &grid_frontier: grid_frontiers) {
            if (m_setting_->frontier->sampling_goals) {
                const int num_goals = std::min(
                    static_cast<int>(grid_frontier.cols()),
                    std::min(
                        static_cast<int>(std::ceil(
                            static_cast<double>(grid_frontier.cols()) *
                            m_setting_->frontier->sampling_ratio)),
                        m_setting_->frontier->max_num_goals_per_frontier));
                std::vector<int> indices(grid_frontier.cols());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), common::g_random_engine);
                for (int i = 0; i < num_goals; ++i) {
                    Eigen::Vector2d metric_goal(
                        grid_map_info->GridToMeterForValue(grid_frontier(0, indices[i]), 0),
                        grid_map_info->GridToMeterForValue(grid_frontier(1, indices[i]), 1));
                    if (std::abs(metric_goal[0] - agent_pose[0]) -
                                m_setting_->plan->goal_tolerance[0] <
                            0 &&
                        std::abs(metric_goal[1] - agent_pose[1]) -
                                m_setting_->plan->goal_tolerance[1] <
                            0) {
                        continue;  // skip the goal if it is too close to the agent
                    }
                    goal_scores.emplace_back(static_cast<double>(grid_frontier.cols()));
                    metric_goals.push_back(std::move(metric_goal));
                }
            } else {
                const Eigen::Vector2i mean_grid =
                    grid_frontier.cast<double>().rowwise().mean().cast<int>();
                Eigen::Vector2d metric_goal(
                    grid_map_info->GridToMeterForValue(mean_grid[0], 0),
                    grid_map_info->GridToMeterForValue(mean_grid[1], 1));
                if (std::abs(metric_goal[0] - agent_pose[0]) - m_setting_->plan->goal_tolerance[0] <
                        0 &&
                    std::abs(metric_goal[1] - agent_pose[1]) - m_setting_->plan->goal_tolerance[1] <
                        0) {
                    continue;  // skip the goal if it is too close to the agent
                }
                goal_scores.emplace_back(static_cast<double>(grid_frontier.cols()));
                metric_goals.push_back(std::move(metric_goal));
            }
        }
        return {goal_scores, metric_goals};
    }

}  // namespace erl::active_mapping
