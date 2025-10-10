#include "erl_active_mapping/frontier_based_grid_2d.hpp"

namespace erl::active_mapping::frontier_based {
    template class AgentFrontierBasedGrid2D<float>;
    template class AgentFrontierBasedGrid2D<double>;
}  // namespace erl::active_mapping::frontier_based

YAML::Node
YAML::convert<erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting>::encode(
    const erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting &setting) {
    Node node;
    ERL_YAML_SAVE_ATTR(node, setting, clean_at_first);
    ERL_YAML_SAVE_ATTR(node, setting, approx_iters);
    ERL_YAML_SAVE_ATTR(node, setting, min_size);
    ERL_YAML_SAVE_ATTR(node, setting, max_num_frontiers);
    ERL_YAML_SAVE_ATTR(node, setting, sample_goals);
    ERL_YAML_SAVE_ATTR(node, setting, sampling_ratio);
    ERL_YAML_SAVE_ATTR(node, setting, max_num_goals_per_frontier);
    return node;
}

bool
YAML::convert<erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting>::decode(
    const YAML::Node &node,
    erl::active_mapping::frontier_based::grid_frontiers::GridFrontierSetting &setting) {
    if (!node.IsMap()) { return false; }
    ERL_YAML_LOAD_ATTR(node, setting, clean_at_first);
    ERL_YAML_LOAD_ATTR(node, setting, approx_iters);
    ERL_YAML_LOAD_ATTR(node, setting, min_size);
    ERL_YAML_LOAD_ATTR(node, setting, max_num_frontiers);
    ERL_YAML_LOAD_ATTR(node, setting, sample_goals);
    ERL_YAML_LOAD_ATTR(node, setting, sampling_ratio);
    ERL_YAML_LOAD_ATTR(node, setting, max_num_goals_per_frontier);
    return true;
}
