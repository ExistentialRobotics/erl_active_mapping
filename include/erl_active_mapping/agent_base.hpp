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

        /**
         * Perform one step of the agent with the sensor pose and observation. This function is
         * responsible for updating the agent's internal state based on the new information received
         * from the environment.
         * @param pose the current pose of the agent/sensor.
         * @param observation the current observation of the agent/sensor.
         */
        virtual void
        Step(const Pose &pose, const Observation &observation) = 0;

        /**
         * Plan a path for exploration based on the current pose of the agent/sensor. This function
         * should return a sequence of states representing the planned path for exploration.
         * @param pose the current pose of the agent/sensor.
         * @return a sequence of states representing the planned path for exploration.
         */
        virtual const Path &
        Plan(const Pose &pose) = 0;

        virtual const Path &
        RandomPlan(const Pose &pose) = 0;

        /**
         * Determine if the agent should replan based on its internal criteria.
         * @param pose the current pose of the agent/sensor.
         * @return true if the agent should replan based on its internal criteria, false
         * otherwise.
         */
        [[nodiscard]] virtual bool
        ShouldReplan(const Pose &pose) = 0;
    };
}  // namespace erl::active_mapping
