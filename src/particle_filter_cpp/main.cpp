#include "particle_filter.hpp"

int main(int argc, char **argv) {
  // Initialize ROS2 node
  rclcpp::init(argc, argv);

  // Create a Particle Filter node
  auto pf_node = std::make_shared<ParticleFilter>();

  // Run PF node until ROS2 is shutdown
  rclcpp::spin(pf_node);

  // Clean up
  rclcpp::shutdown();

  return 0;
}
