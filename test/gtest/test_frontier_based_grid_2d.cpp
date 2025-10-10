#include "erl_active_mapping/frontier_based_grid_2d.hpp"
#include "erl_common/plplot_fig.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"

template<typename Dtype>
class SimulatorFrontier2D {
public:
    using Agent = erl::active_mapping::frontier_based::AgentFrontierBasedGrid2D<Dtype>;
    using LogOddMap = typename Agent::LogOddMap;
    using GridMapInfo = typename Agent::GridMapInfo;
    using AgentState = typename Agent::State;
    using Path = typename Agent::Path;
    using SensorPose = Eigen::Matrix<Dtype, 2, 3>;  // [R|t] 2x3

    // given sensor pose, return a point cloud in the world frame.
    using SensorCallback = std::function<Eigen::Matrix2X<Dtype>(const SensorPose &)>;

private:
    std::shared_ptr<Agent> m_agent_ = nullptr;
    SensorPose m_sensor_pose_ = SensorPose::Identity();
    SensorCallback m_sensor_callback_{};
    Dtype m_max_observable_area_ = -1.0f;
    Dtype m_min_coverage_ratio_ = 0.9f;

    Dtype m_dist_ = 0;
    Dtype m_observed_area_ = 0;
    Dtype m_ratio_ = 0;
    AgentState m_agent_state_{};
    long m_step_ = 0;
    long m_wp_idx_ = 0;
    Path m_path_{};

public:
    explicit SimulatorFrontier2D(
        const std::shared_ptr<Agent> &agent,
        SensorPose init_sensor_pose,
        SensorCallback sensor_callback,
        Dtype max_observable_area,
        Dtype min_coverage_ratio)
        : m_agent_(agent),
          m_sensor_pose_(std::move(init_sensor_pose)),
          m_sensor_callback_(std::move(sensor_callback)),
          m_max_observable_area_(max_observable_area),
          m_min_coverage_ratio_(min_coverage_ratio) {}

    [[nodiscard]] const Path &
    GetPath() const {
        return m_path_;
    }

    [[nodiscard]] const SensorPose &
    GetSensorPose() const {
        return m_sensor_pose_;
    }

    [[nodiscard]] long
    GetStep() const {
        return m_step_;
    }

    [[nodiscard]] Dtype
    GetDistance() const {
        return m_dist_;
    }

    [[nodiscard]] Dtype
    GetObservedArea() const {
        return m_observed_area_;
    }

    [[nodiscard]] Dtype
    GetCoverageRatio() const {
        return m_ratio_;
    }

    /**
     *
     * @return if true, continue; otherwise stop.
     */
    bool
    Step() {
        if (m_step_++) {
            auto p = m_path_.col(m_wp_idx_++);
            Eigen::Vector2<Dtype> diff = p - m_sensor_pose_.col(2);
            m_dist_ += diff.norm();
            Dtype theta = std::atan2(diff[1], diff[0]);
            Dtype sin_theta = std::sin(theta);
            Dtype cos_theta = std::cos(theta);
            // clang-format off
            m_sensor_pose_ << cos_theta, -sin_theta, p[0],
                              sin_theta,  cos_theta, p[1];
            // clang-format on
            m_agent_state_.metric = p;
            Eigen::Matrix2X<Dtype> observation = m_sensor_callback_(m_sensor_pose_);
            m_agent_->Step(m_sensor_pose_, observation);
            if (m_wp_idx_ >= m_path_.cols()) {
                m_path_ = m_agent_->Plan(m_agent_state_);
                m_wp_idx_ = 0;
            }
        } else {
            m_agent_state_.metric = m_sensor_pose_.col(2);
            Eigen::Matrix2X<Dtype> observation = m_sensor_callback_(m_sensor_pose_);
            m_agent_->Step(m_sensor_pose_, observation);
            m_path_ = m_agent_->Plan(m_agent_state_);
            m_wp_idx_ = 0;
        }

        m_observed_area_ = ComputeObservedArea();

        if (m_wp_idx_ > 0 && m_agent_->ShouldReplan(m_agent_state_)) {
            // try re-plan only when the robot is moving.
            Path new_path = m_agent_->Plan(m_agent_state_);
            ERL_INFO("New path with {} waypoints at step {}.", new_path.cols(), m_step_);
            m_path_ = std::move(new_path);
            m_wp_idx_ = 0;
        }

        m_ratio_ = m_observed_area_ / m_max_observable_area_;

        return m_ratio_ < m_min_coverage_ratio_ && m_path_.cols() > 0;
    }

    [[nodiscard]] Dtype
    ComputeObservedArea() const {
        auto log_odd_map = m_agent_->GetLogOddMap();
        std::size_t num_free_cells = log_odd_map->GetNumFreeCells();
        Eigen::Vector2<Dtype> res = log_odd_map->GetGridMapInfo()->Resolution();
        Dtype area = static_cast<Dtype>(num_free_cells) * res[0] * res[1];
        return area;
    }
};

TEST(FrontierBased, Grid2D) {
    GTEST_PREPARE_OUTPUT_DIR();

    using Dtype = float;
    using Simulator = SimulatorFrontier2D<Dtype>;
    using Agent = Simulator::Agent;
    using GridMapInfo = Agent::GridMapInfo;
    using SensorPose = Simulator::SensorPose;
    using LidarSetting = erl::geometry::Lidar2D::Setting;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;

    constexpr Dtype wall_thickness = 0.2f;
    constexpr Dtype map_resolution = 0.05f;
    constexpr Dtype min_coverage_ratio = 0.95f;
    constexpr int canvas_size = 1000;
    constexpr bool save_figs = false;

    std::filesystem::path img_dir = test_output_dir / "images";
    std::filesystem::create_directories(img_dir);

    std::shared_ptr<Agent::Setting> agent_setting = std::make_shared<Agent::Setting>();
    std::string agent_setting_path = gtest_src_dir / "../../config/frontier_based_grid_2d.yaml";
    ASSERT_TRUE(agent_setting->FromYamlFile(agent_setting_path));

    erl::geometry::HouseExpoMap house_expo_map(
        gtest_src_dir / "../../data/house_expo_room_1451.json",
        wall_thickness);
    std::shared_ptr<LidarSetting> lidar_setting = std::make_shared<LidarSetting>();
    erl::geometry::Lidar2D lidar(lidar_setting, house_expo_map.GetMeterSpace());
    Eigen::Matrix2Xf ray_directions = lidar.GetRayDirectionsInFrame().cast<Dtype>();

    Eigen::Matrix3X<Dtype> traj = erl::common::LoadEigenMatrixFromTextFile<Dtype>(
        gtest_src_dir / "../../data/house_expo_room_1451.csv",
        erl::common::EigenTextFormat::kCsvFmt,
        true);

    Vector2 map_min = house_expo_map.GetMapMin().cast<Dtype>();
    Vector2 map_max = house_expo_map.GetMapMax().cast<Dtype>();
    std::shared_ptr<GridMapInfo> grid_map_info = std::make_shared<GridMapInfo>(
        map_min,
        map_max,
        Vector2(map_resolution, map_resolution),
        Eigen::Vector2i::Constant(10));  // 10 cells padding
    // map_min = grid_map_info->Min();
    // map_max = grid_map_info->Max();

    Eigen::MatrixX8U map_img = house_expo_map.GetMeterSpace()->GenerateMapImage(
        *grid_map_info->CastSharedPtr<double>(),
        true);
    cv::Mat cv_map_img;
    cv::eigen2cv(map_img, cv_map_img);

    Dtype scale = 1.0f;
    if (cv_map_img.rows != canvas_size) {
        scale = static_cast<Dtype>(canvas_size) / static_cast<Dtype>(cv_map_img.rows);
        cv::resize(cv_map_img, cv_map_img, cv::Size(), scale, scale, cv::INTER_NEAREST);
    }

    // cv::imshow("map_img", cv_map_img);
    // cv::waitKey(1);

    const Dtype max_observable_area = map_img.cast<Dtype>().sum() / 255.0f *
                                      grid_map_info->Resolution(0) * grid_map_info->Resolution(1);

    std::shared_ptr<Agent> agent = std::make_shared<Agent>(agent_setting, grid_map_info);

    Vector2 agent_pos(traj(0, 0), traj(1, 0));
    Dtype agent_heading = traj(2, 0);
    SensorPose init_sensor_pose;
    // set pose
    Dtype sin_theta = std::sin(agent_heading);
    Dtype cos_theta = std::cos(agent_heading);
    // clang-format off
    init_sensor_pose << cos_theta, -sin_theta, agent_pos[0],
                        sin_theta,  cos_theta, agent_pos[1];
    // clang-format on

    SimulatorFrontier2D<Dtype> simulator(
        agent,
        init_sensor_pose,
        [&](const SensorPose &pose) {
            const double rotation_angle = std::atan2(pose(1, 0), pose(0, 0));
            VectorX ranges =
                lidar.Scan(rotation_angle, pose.col(2).cast<double>(), true).cast<Dtype>();
            Matrix2X points(2, ranges.size());
            long idx = 0;
            for (long i = 0; i < ranges.size(); ++i) {
                if (!std::isfinite(ranges[i])) { continue; }
                points.col(idx++) =
                    pose.leftCols<2>() * (ray_directions.col(i) * ranges[i]) + pose.col(2);
            }
            points.conservativeResize(Eigen::NoChange, idx);
            return points;
        },
        max_observable_area,
        min_coverage_ratio);

    constexpr int fig_height = 300;
    cv::Mat canvas(cv_map_img.rows + fig_height, cv_map_img.cols, CV_8UC4);
    cv::Mat img;
    using namespace erl::common;
    PlplotFig fig(cv_map_img.cols, fig_height, true);
    PlplotFig::LegendOpt legend_opt(2, {"Observed Ratio", "Travel Distance"});
    legend_opt.SetTextColors({PlplotFig::Color0::Red, PlplotFig::Color0::Green})
        .SetStyles({PL_LEGEND_LINE, PL_LEGEND_LINE})
        .SetLineColors(legend_opt.text_colors)
        .SetLineStyles({1, 1})
        .SetLineWidths({1.0, 1.0})
        .SetPosition(PL_POSITION_LEFT | PL_POSITION_TOP)
        .SetBoxStyle(PL_LEGEND_BOUNDING_BOX | PL_LEGEND_BACKGROUND)
        .SetLegendBoxLineColor0(PlplotFig::Color0::Black)
        .SetBgColor0(PlplotFig::Color0::Gray)
        .SetTextScale(1.1);
    std::vector<std::vector<cv::Point2i>> traj_pixels(1);
    std::vector<std::vector<cv::Point2i>> robot_contour(1);
    std::vector<double> steps;
    std::vector<double> coverage_ratios;
    std::vector<double> travel_distances;
    while (simulator.Step()) {
        // visualize here
        img = 1.0f - agent->GetLogOddMap()->GetPossibilityMap();
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGRA);
        img.convertTo(img, CV_8UC4, 255.0f);

        if (const long best_frontier_idx = agent->GetBestFrontierIndex(); best_frontier_idx >= 0) {
            auto frontier = agent->GetFrontiers()[best_frontier_idx];
            for (const auto &p: frontier.points) {
                Eigen::Vector2i pixel = grid_map_info->MeterToGridForPoint(p.metric);
                cv::circle(img, cv::Point(pixel[1], pixel[0]), 1, cv::Scalar(0, 255, 0), -1);
            }
        }

        auto p = simulator.GetSensorPose().col(2);
        traj_pixels[0].emplace_back(
            grid_map_info->MeterToGridAtDim(p[1], 1),
            grid_map_info->MeterToGridAtDim(p[0], 0));

        cv::polylines(img, traj_pixels, false, cv::Scalar(0, 0, 255), 1);
        if (agent_setting->env->robot_metric_contour.cols() == 0) {
            cv::circle(img, traj_pixels[0].back(), 3, cv::Scalar(0, 128, 255), -1);
        } else {
            Matrix2X robot_metric_contour = agent_setting->env->robot_metric_contour.colwise() + p;
            robot_contour[0].reserve(robot_metric_contour.cols());
            robot_contour[0].clear();
            for (long i = 0; i < agent_setting->env->robot_metric_contour.cols(); ++i) {
                auto pp = robot_metric_contour.col(i);
                robot_contour[0].emplace_back(
                    grid_map_info->MeterToGridAtDim(pp[1], 1),
                    grid_map_info->MeterToGridAtDim(pp[0], 0));
            }
            cv::drawContours(img, robot_contour, 0, cv::Scalar(0, 128, 255), -1);
        }

        if (scale != 1.0f) { cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_NEAREST); }
        img = img.t();
        cv::flip(img, img, 0);

        const auto step = static_cast<double>(simulator.GetStep());
        constexpr double fig_t_span = 1000;
        steps.push_back(step);
        coverage_ratios.push_back(simulator.GetCoverageRatio());
        travel_distances.push_back(simulator.GetDistance());
        double *steps_ptr = steps.data();
        double *coverage_ratios_ptr = coverage_ratios.data();
        double *travel_distances_ptr = travel_distances.data();
        int n = static_cast<int>(steps.size());
        if (steps.size() > static_cast<std::size_t>(fig_t_span)) {
            const auto offset = static_cast<long>(static_cast<double>(steps.size()) - fig_t_span);
            steps_ptr += offset;
            coverage_ratios_ptr += offset;
            travel_distances_ptr += offset;
            n = static_cast<int>(fig_t_span);
        }
        fig.Clear()
            .SetMargin(0.1, 0.9, 0.2, 0.85)
            .SetAxisLimits(step - fig_t_span, step, 0, 1.1f)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .DrawAxesBox(
                PlplotFig::AxisOpt().DrawTopRightEdge(),
                PlplotFig::AxisOpt().DrawPerpendicularTickLabels())
            .SetAxisLabelX("step")
            .SetAxisLabelY("coverage ratio")
            .SetCurrentColor(PlplotFig::Color0::Red)
            .SetPenWidth(3)
            .DrawLine(n, steps_ptr, coverage_ratios_ptr)
            .SetPenWidth(1)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetAxisLimits(step - fig_t_span, step, 0, travel_distances.back() * 1.1f)
            .DrawAxesBox(
                PlplotFig::AxisOpt::Off(),
                PlplotFig::AxisOpt::Off()
                    .DrawTopRightEdge()
                    .DrawTickMajor()
                    .DrawTickMinor()
                    .DrawTopRightTickLabels()
                    .DrawPerpendicularTickLabels())
            .SetAxisLabelY("travel distance", true)
            .SetCurrentColor(PlplotFig::Color0::Green)
            .SetPenWidth(2)
            .DrawLine(n, steps_ptr, travel_distances_ptr)
            .SetPenWidth(1)
            .Legend(legend_opt)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetTitle(
                fmt::format(
                    "Observed area {:.2f}, ratio {:.2f}%, distance {:.2f} m, step {}",
                    simulator.GetObservedArea(),
                    coverage_ratios.back() * 100.0f,
                    travel_distances.back(),
                    simulator.GetStep())
                    .c_str());

        img.copyTo(canvas(cv::Rect(0, 0, img.cols, img.rows)));
        fig.ToCvMat().copyTo(canvas(cv::Rect(0, img.rows, fig.Width(), fig.Height())));

        cv::imshow("exploration", canvas);
        cv::waitKey(1);

        if (save_figs) {
            std::string img_path =
                (img_dir / fmt::format("frontier_grid_{:04d}.png", simulator.GetStep())).string();
            cv::imwrite(img_path, canvas);
        }
    }

    cv::waitKey();
}
