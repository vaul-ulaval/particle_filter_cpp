# Particle Filter Localization (implementation in C++)

This project is a C++ version implementation of the particle filter algorithm. It realizes the same functionality as this reference [python version implementation](https://github.com/mit-racecar/particle_filter).

It depends on the library [RangeLibc](https://github.com/kctess5/range_libc) for fast 2D ray casting. For easy usage, we make slight modifcations and convert it into ROS package format, see this [fork](https://github.com/nanli42/range_libc). In this project, we use the `find_package(catkin REQUIRED COMPONENTS ... range_libc)` and `catkin_package(CATKIN_DEPENDS ... range_libc)` in `CMakeLists.txt` to include it.

The code style is inspired by the project [droemer7/localize](https://github.com/droemer7/localize) which is highly recommanded for the AMCL (Adaptive Monte Carlo Localization) problem.

# Limitation

In the original python implementation, there are different ray casting methods (range_method, which determines which RangeLibc algorithm to use) and different sensor model variants (rangelib_variant, high values are more optimal). For more details please refer to [this file](https://github.com/mit-racecar/particle_filter/blob/master/launch/localize.launch) and the python version [code](https://github.com/mit-racecar/particle_filter/blob/master/src/particle_filter.py).

For now, in this C++ implementation, we only offer the following options:
```
range_method = rmgpu (i.e. GPU is used)
rangelib_variant = 2 (i.e. VAR_REPEAT_ANGLES_EVAL_SENSOR)
```

The implementation of other options for range_method and ranglib_variant should be straightforward. Basically, you need replace these lines in `src/particle_filter.cpp` to the corresponding types and functions:
```
range_method_ = new RayMarchingGPU(...);
(dynamic_cast<RayMarchingGPU*> (range_method_))->set_sensor_model(...);
(dynamic_cast<RayMarchingGPU*> (range_method_))->numpy_calc_range_angles(...);
(dynamic_cast<RayMarchingGPU*> (range_method_))->eval_sensor_model(...);
(dynamic_cast<RayMarchingGPU*> (range_method_))->numpy_calc_range(...);
```

# Installation

To get started, clone the forked range_libc project into your ros ros workspace `ros_ws`. You may also need to adjust your GPU architecture (`-arch=sm_*`) in `CMakeList.txt` file.
```
cd ros_ws/src/
git clone https://https://github.com/nanli42/range_libc
```
Then download this project and compile it:
```
cd ros_ros/src/
git clone https://https://github.com/nanli42/particle_filter_cpp
cd ros_ros
catkin_make
```

# Usage

Similar to the python verison implementation, we provide a launch file `launch/localize_cpp.launch` to run the particle filter node with given parameters:

```
roslaunch particle_filter localize_cpp.launch
```
You may need to change parameters "odometry_topic", "scan_topic", and others to match your environment.

Once the particle filter is running, you can visualize the map and other particle filter visualization messages in RViz. Use the "2D Pose Estimate" tool from the RViz toolbar to initialize the particle locations.

# Run together with f1tenth_gym_ros

[f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros) is a simulator for the race car [f1tenth](https://f1tenth.org/), which is similar to mit-racecar. To run with this simulator, you need to modify the launch file `launch/localize_cpp.launch`:
1. comment the map_server related lines (since the simulator will start a map server on its own)
2. change the arguments `<arg name="odometry_topic" default="/ego_racecar/odom"/>` (or `<arg name="odometry_topic" default="/odom"/>`, depending on which version you use)

# Reference
1. 2D ray casting: [kctess5/range_libc](https://github.com/kctess5/range_libc)
2. Particle filter in python: [mit-racecar/particle_filter](https://github.com/mit-racecar/particle_filter)
3. An AMCL algorithm implementation using range_libc: [droemer7/localize](https://github.com/droemer7/localize)
