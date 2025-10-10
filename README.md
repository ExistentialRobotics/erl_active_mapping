# erl_active_mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS1](https://img.shields.io/badge/ROS1-noetic-blue)](http://wiki.ros.org/)
[![ROS2](https://img.shields.io/badge/ROS2-humble-blue)](https://docs.ros.org/)

`erl_active_mapping` provides algorithms and agents for active exploration and mapping. The package includes a small, reusable agent interface and a frontier-based 2D grid agent that integrates log-odd occupancy maps, environment / motion models, and A* planning to generate exploration goals and paths.

## Highlights

- Templated `AgentBase` interface for implementing exploration agents (`include/erl_active_mapping/agent_base.hpp`).
- Frontier-based 2D grid agent that extracts frontiers from a log-odd occupancy map and plans to frontier goals (`include/erl_active_mapping/frontier_based_grid_2d.hpp`).
- YAML-configurable settings for frontier extraction, planning strategies and environment parameters.
- Integrations with other erl packages: `erl_geometry::LogOddMap2D`, `erl_env::Environment2D`, and `erl_path_planning::astar`.
- Python bindings (pybind11) for easy experimentation and scripting (`python/`) (TODO).

## Public headers

- `include/erl_active_mapping/agent_base.hpp` — Templated base class describing the agent API (Step, Plan, RandomPlan, ShouldReplan).
- `include/erl_active_mapping/frontier_based_grid_2d.hpp` — Frontier-based grid agent implementation, settings and helper types.

## Getting started

Create a workspace and add the package to `src` (example follows the standard CMake/ROS layout used across the erl_* family):

```bash
cd <your_workspace>
mkdir -p src
# clone or copy erl_active_mapping into src
```

### Build with CMake

This package is a normal CMake project (C++17). From your workspace root you can use a CMake-based build if you prefer:

```cmake
cmake_minimum_required(VERSION 3.16)
project(<your_project_name>)
add_subdirectory(src/erl_cmake_tools)
add_subdirectory(src/erl_common)
add_subdirectory(src/erl_geometry)
add_subdirectory(src/erl_env)
add_subdirectory(src/erl_path_planning)
add_subdirectory(src/erl_active_mapping)
```

Then build:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Use as a ROS package

The package includes ROS package metadata (`package.xml`) so it can be built with standard ROS build tools.

For ROS 1 (catkin):

```bash
cd <your_workspace>
source /opt/ros/<ros_distro>/setup.bash
catkin build erl_active_mapping
source devel/setup.bash
```

For ROS 2 (colcon):

```bash
cd <your_workspace>
source /opt/ros/<ros_distro>/setup.bash
colcon build --packages-up-to erl_active_mapping
source install/setup.bash
```

### Python bindings

This package includes Python bindings (pybind11). There is a Python package layout under `python/` and packaging metadata (`setup.py`, `pyproject.toml`). To install into your active Python environment (pipenv, venv, or system):

```bash
cd <your_workspace>
cd src/erl_active_mapping
pip install . --user
# or using pipenv/venv: pip install .
```

After installation you should be able to import the module in Python:

```python
import erl_active_mapping
# The bindings expose the frontier-based agent and helper utilities for scripting
```

Refer to `python/binding/pybind11_erl_active_mapping.cpp` and `python/erl_active_mapping/__init__.py` for the exposed symbols and examples.

## Example configs & data

- `config/frontier_based_grid_2d.yaml` — example YAML settings for the frontier-based grid agent.
- `data/house_expo_room_1451.json` and `data/house_expo_room_1451.csv` — small example datasets used in experiments and tests.

Use the YAML config to customize planner behavior (plan strategy, replan strategy, goal tolerance, sampling parameters, etc.).

## API (quick overview)

- `AgentBase<Dtype, Dim, Observation>`: minimal interface for agents. Methods:
    - `Step(pose, observation)`: process a new observation at the given pose.
    - `Plan(state)`: generate a plan based on the current state.
    - `RandomPlan(state)`: generate a random plan based on the current state.
    - `ShouldReplan(state)`: determine if a replan is needed based on the current state.
- `frontier_based::AgentFrontierBasedGrid2D<Dtype, Observation>`: implements frontier extraction from a `LogOddMap2D`, goal sampling, and A*-based planning to frontiers. Exposes `Setting` and `Frontier` types for configuration and inspection.

## Contributing

Contributions welcome. Please follow the repository code style (files include `.clang-format`) and add tests where helpful. See `CMakeLists.txt` and Python packaging files for how targets and bindings are built.

## License

This project is licensed under the MIT License — see the `LICENSE` file in the package for details.
