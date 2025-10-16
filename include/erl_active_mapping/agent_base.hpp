#pragma once

#include "erl_common/eigen.hpp"
#include "erl_env/environment_state.hpp"

namespace erl::active_mapping {

    template<typename Dtype, int Dim, typename Observation>
    struct AgentBase {

        using Observation_t = Observation;
        using State = env::EnvironmentState<Dtype, Dim>;
        using Pose = Eigen::Matrix<Dtype, Dim, Dim + 1>;
        using Path = std::vector<Pose>;

        virtual ~AgentBase() = default;

        virtual void
        Step(const Pose &pose, const Observation &observation) = 0;

        /**
         * @param pose the current pose of the agent/sensor.
         * @return a sequence of states representing the planned path for exploration.
         */
        virtual const Path &
        Plan(const Pose &pose) = 0;

        virtual const Path &
        RandomPlan(const Pose &pose) = 0;

        /**
         *
         * @return true if the agent should replan based on its internal criteria, false
         * otherwise.
         */
        [[nodiscard]] virtual bool
        ShouldReplan(const Pose &pose) = 0;
    };
}  // namespace erl::active_mapping
