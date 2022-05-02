#include "particle_filter.h"

int main(int argc, char **argv) {

  // Initialize ROS node
  ros::init(argc, argv, "particle_filter_cpp");

  // Create a Particle Filter node
  ParticleFilter pf;

  // Run PF node until ROS is shutdown
  ros::waitForShutdown();

  return 0;
}
