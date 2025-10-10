#pragma once

#include "erl_common/eigen.hpp"
#include "erl_env/environment_state.hpp"

namespace erl::active_mapping {

    template<typename Dtype, int Dim, typename Observation>
    struct AgentBase {

        using State = env::EnvironmentState<Dtype, Dim>;
        using Pose = Eigen::Matrix<Dtype, Dim, Dim + 1>;
        using Path = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;

        virtual ~AgentBase() = default;

        virtual void
        Step(const Pose &pose, const Observation &observation) = 0;

        /**
         * @param state Current state of the agent (e.g., position, orientation).
         * @return a sequence of states representing the planned path for exploration.
         */
        virtual const Path &
        Plan(const State &state) = 0;

        virtual const Path &
        RandomPlan(const State &state) = 0;

        /**
         *
         * @return true if the agent should replan based on its internal criteria, false
         * otherwise.
         */
        [[nodiscard]] virtual bool
        ShouldReplan(const State &state) = 0;
    };
}  // namespace erl::active_mapping
