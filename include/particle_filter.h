#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <ros/spinner.h>
#include <ros/service.h>

#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/GetMap.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/transform_broadcaster.h>

#include <iostream>
#include <mutex>

#include "range_libc/RangeLib.h"
using namespace ranges;

#define VAR_NO_EVAL_SENSOR_MODEL 0
#define VAR_CALC_RANGE_MANY_EVAL_SENSOR 1
#define VAR_REPEAT_ANGLES_EVAL_SENSOR 2
#define VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT 3
#define VAR_RADIAL_CDDT_OPTIMIZATIONS 4


typedef struct {
	double x;
	double y;
	double theta;
	double weight;
} ParticleState;

typedef std::recursive_mutex RecursiveMutex;
typedef std::lock_guard<std::recursive_mutex> RecursiveLock;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::_V2::system_clock::time_point Time;
typedef std::chrono::duration<double> Duration;

/*----------------------------------------------------------------------------//
  The following code about "getParam", "class RNG" and "durationMsec" is copied from the project:
  https://github.com/droemer7/localize
//----------------------------------------------------------------------------*/
// Retrieve the desired parameter value from the ROS parameter server
template <class T>
bool getParam(const ros::NodeHandle& nh,
              std::string name,
              T& val
             )
{
  bool result = true;

  if (!nh.getParam(name, val)) {
    ROS_FATAL("AMCL: Parameter '%s' not found", name.c_str());
    result = false;
  }
  return result;
}

// RNG (Random Number Generator) wrapper
class RNG
{
public:
  // Constructor
  RNG()
  {
    // Initialize random number generator
    // Source: https://stackoverflow.com/a/13446015
    std::random_device dev;
    std::chrono::_V2::system_clock::duration time = std::chrono::_V2::system_clock::now().time_since_epoch();
    std::mt19937::result_type time_seconds = std::chrono::duration_cast<std::chrono::seconds>(time).count();
    std::mt19937::result_type time_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
    std::mt19937::result_type seed = dev() ^ (time_seconds + time_microseconds);
    gen_.seed(seed);
  }

  // A reference to the random number engine
  std::mt19937& engine()
  { return gen_; }

private:
  std::mt19937 gen_;  // Generator for random numbers
};

inline double durationMsec(const Time& start, const Time& end)
  { return std::chrono::duration_cast<Duration>(end - start).count() * 1000.0; }
//----------------------------------------------------------------------------*/


class ParticleFilter {
public:
  ParticleFilter();

	// Load parameters
	void loadParam();
	// Load static map
	void loadMap();
	// Prepare the sensor model table
	void precomputeSensorModel();
	// Set up ROS interface
	void setUpROS();
	// Initialize particles over the permissible region of state space
	void initializeGlobalDistribution();
	// Initialize particles according to a pose msg
	void initializeParticlesPose(const geometry_msgs::PoseWithCovarianceStamped& msg);
	// Callback functions
  void lidarCB(const sensor_msgs::LaserScan& msg);
  void odomCB(const nav_msgs::Odometry& msg);
	void clickedPoseCB(const geometry_msgs::PoseWithCovarianceStamped& msg);
	// Fucntions for MCL algorithm
	void update();
	void sampling();
	void motionModel();
	void sensorModel();
	void expectedPose();
	void publishTfOdom();
	void visualize();

	// Utils
	std::vector<int> worldToMap(std::vector<double> position);
	std::vector<double> mapToWorld(std::vector<int> idx);

private:
  // ROS interface
  ros::Publisher pose_pub_;
	ros::Publisher odom_pub_;
  ros::Publisher particle_pub_;
  ros::Publisher fake_scan_pub_;

  ros::Subscriber laser_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber click_sub_;

	ros::NodeHandle pose_pub_nh_;
	ros::NodeHandle odom_pub_nh_;
  ros::NodeHandle particle_pub_nh_;
  ros::NodeHandle fake_scan_pub_nh_;
  ros::NodeHandle laser_sub_nh_;
  ros::NodeHandle odom_sub_nh_;
  ros::NodeHandle pose_sub_nh_;
	ros::NodeHandle click_sub_nh_;

  ros::CallbackQueue laser_sub_queue_;
  ros::CallbackQueue odom_sub_queue_;
  ros::CallbackQueue pose_sub_queue_;
	ros::CallbackQueue click_sub_queue_;

  ros::AsyncSpinner laser_sub_spinner_;
  ros::AsyncSpinner odom_sub_spinner_;
  ros::AsyncSpinner pose_sub_spinner_;
	ros::AsyncSpinner click_sub_spinner_;

  // Data containers used in MCL algorithm
  double max_range_px_;
	RangeMethod* range_method_;
	RecursiveMutex particles_mtx_;
	nav_msgs::OccupancyGrid loaded_map_;
	std::vector<ParticleState> particles_;
	ParticleState expected_pose_;

	bool map_initialized_;
	bool odom_initialized_;
	bool lidar_initialized_;
	ros::Time last_stamp_;
	std::vector<double> last_pose_;
	std::vector<double> odometry_data_;

	double angle_min_;
	double angle_increment_;
	std::vector<double> downsampled_laser_angles_;
	std::vector<double> downsampled_laser_ranges_;

	// Sampling related tools
	RNG rng_; // Random number generator
	std::uniform_real_distribution<double> x_dist_;   // Distribution of x locations in map frame [0, width)
	std::uniform_real_distribution<double> y_dist_;   // Distribution of y locations in map frame [0, height)
	std::uniform_real_distribution<double> th_dist_;  // Distribution of theta in map frame (-pi, pi]

  // Variables corresponding to parameters defined in launch file
  // topic parameters
  std::string scan_topic_;
  std::string odometry_topic_;

  // sensor model constants
  double z_short_;
  double z_max_;
  double z_rand_;
  double z_hit_;
  double sigma_hit_;

  // motion model dispersion constant
  double motion_dispersion_x_;
  double motion_dispersion_y_;
  double motion_dispersion_theta_;

  // options
  int viz_;
  int fine_timing_;
  int publish_odom_;

  // downsampling parameter and other parameters
  int angle_step_;
	int num_downsampled_angles_;
  int max_particles_num_;
  int max_viz_particles_;
  int rangelib_variant_;
	std::string which_range_method_;
  double theta_discretization_;
  double squash_factor_;
  double max_range_;

  // Timing related variables
  double sensor_model_calc_worst_time_;
  double motion_model_calc_worst_time_;
  
  // Set up array pointers 
  float *angles_;
  float *obs_;
  float *outs_;
  double *weights_;
  float *samples_;
  float *viz_queries_;
  float *viz_ranges_;
};

#endif
