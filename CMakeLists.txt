cmake_minimum_required(VERSION 3.8)
project(particle_filter)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Add support for CUDA
if(DEFINED ENV{USE_CUDA})
  add_definitions(-DUSE_CUDA=1)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Uncomment to generate message interfaces if needed
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Detection.msg"
#   DEPENDENCIES sensor_msgs std_msgs
# )

# Include directories
include_directories(
  include
  /usr/local/include
)
link_directories(/usr/local/lib)

# Create the C++ node executable
add_executable(${PROJECT_NAME}_cpp_node
  src/particle_filter_cpp/particle_filter.cpp
)

# Specify dependencies
ament_target_dependencies(${PROJECT_NAME}_cpp_node
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  nav_msgs
  geometry_msgs
  tf2_geometry_msgs
)

# target_link_libraries(${PROJECT_NAME}_cpp_node
#   range_lib
# )

# Install executables
install(TARGETS
  ${PROJECT_NAME}_cpp_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files if you have any
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN ".svn" EXCLUDE
)

# Install include directories if needed
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# Generate package.xml dependencies
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  range_lib
  tf2
  tf2_ros
  nav_msgs
  geometry_msgs
)

ament_package()
