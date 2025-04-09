#include "particle_filter.hpp"
#include <rclcpp/exceptions/exceptions.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>
#include "nav_msgs/msg/occupancy_grid.hpp"

ParticleFilter::ParticleFilter()
    : Node("particle_filter"), map_initialized_(false), odom_initialized_(false), lidar_initialized_(false), motion_model_calc_worst_time_(0.0), sensor_model_calc_worst_time_(0.0) {
    loadParam();
    RCLCPP_INFO(this->get_logger(), "MOM");
    precomputeSensorModel();
    RCLCPP_INFO(this->get_logger(), "MOM");
    initializeGlobalDistribution();

    // Initialize TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // Set up publishers
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 1);
    particle_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
    fake_scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>("/pf/viz/fake_scan", 1);

    // Set up subscribers
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    auto sensor_qos = rclcpp::SensorDataQoS();

    laser_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic_, sensor_qos, std::bind(&ParticleFilter::lidarCB, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(odometry_topic_, qos, std::bind(&ParticleFilter::odomCB, this, std::placeholders::_1));

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", qos, std::bind(&ParticleFilter::clickedPoseCB, this, std::placeholders::_1));

    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>("/map", qos, std::bind(&ParticleFilter::mapCB, this, std::placeholders::_1));

    // Set up map client
    map_client_ = create_client<nav_msgs::srv::GetMap>("static_map");
}

void ParticleFilter::loadParam() {
    // Declare and load all parameters
    scan_topic_ = declare_parameter<std::string>("scan_topic", "scan");
    odometry_topic_ = declare_parameter<std::string>("odometry_topic", "odom");
    angle_step_ = declare_parameter<int>("angle_step", 10);
    max_particles_num_ = declare_parameter<int>("max_particles", 4000);
    max_viz_particles_ = declare_parameter<int>("max_viz_particles", 50);
    which_range_method_ = declare_parameter<std::string>("range_method", "rmgpu");
    theta_discretization_ = declare_parameter<double>("theta_discretization", 112);
    squash_factor_ = declare_parameter<double>("squash_factor", 2.2);
    max_range_ = declare_parameter<double>("max_range", 10.0);
    rangelib_variant_ = declare_parameter<int>("rangelib_variant", 4);
    fine_timing_ = declare_parameter<int>("fine_timing", 0);
    publish_odom_ = declare_parameter<int>("publish_odom", 1);
    viz_ = declare_parameter<int>("viz", 1);
    z_short_ = declare_parameter<double>("z_short", 0.1);
    z_max_ = declare_parameter<double>("z_max", 0.1);
    z_rand_ = declare_parameter<double>("z_rand", 0.05);
    z_hit_ = declare_parameter<double>("z_hit", 0.75);
    sigma_hit_ = declare_parameter<double>("sigma_hit", 8.0);
    motion_dispersion_x_ = declare_parameter<double>("motion_dispersion_x", 0.05);
    motion_dispersion_y_ = declare_parameter<double>("motion_dispersion_y", 0.025);
    motion_dispersion_theta_ = declare_parameter<double>("motion_dispersion_theta", 0.25);

    RCLCPP_INFO(get_logger(), "Successfully loaded all parameters defined in launch file.");

    x_dist_ = std::uniform_real_distribution<double>();
    y_dist_ = std::uniform_real_distribution<double>();
    th_dist_ = std::uniform_real_distribution<double>();
}

void ParticleFilter::mapCB(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    loaded_map_ = msg;
    RCLCPP_INFO(get_logger(), "1");

    int rows = loaded_map_->info.height;
    int cols = loaded_map_->info.width;
    double mapResolution = loaded_map_->info.resolution;
    RCLCPP_INFO(get_logger(), "Received a %d X %d map @ %.3f m/px",
                loaded_map_->info.height,  // rows
                loaded_map_->info.width,   // cols
                loaded_map_->info.resolution);

    // max range in pixel
    max_range_px_ = (int)(max_range_ / mapResolution);

    // Transform loaded map into OMap format which is needed by range_libc
    // ref: originale range_libc project - range_libc/pywrapper/RangeLibc.pyx, line 146 USE_ROS_MAP
    OMap map = OMap(loaded_map_->info.height, loaded_map_->info.width);
    for (int i = 0; i < loaded_map_->info.height; i++) {
        for (int j = 0; j < loaded_map_->info.width; j++) {
            if (loaded_map_->data[i * loaded_map_->info.width + j] == 0)
                map.grid[i][j] = false;  // free space
            else
                map.grid[i][j] = true;  // occupied
        }
    }

    // Use tf2 for quaternion conversion in ROS 2
    tf2::Quaternion q;
    tf2::fromMsg(loaded_map_->info.origin.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    double angle = -1.0 * yaw;

    map.world_scale = loaded_map_->info.resolution;
    map.world_angle = angle;
    map.world_origin_x = loaded_map_->info.origin.position.x;
    map.world_origin_y = loaded_map_->info.origin.position.y;
    map.world_sin_angle = sin(angle);
    map.world_cos_angle = cos(angle);

    RCLCPP_INFO_STREAM(get_logger(), "Set range method: " << which_range_method_);
    if (which_range_method_ == "rmgpu") {
        range_method_ = new RayMarchingGPU(map, max_range_px_);
    } else {
        throw std::runtime_error(
            "Not yet implemented range_method. "
            "Please check this parameter in launch file. "
            "Or modified the code in ParticleFilter::loadMap().");
    }

    RCLCPP_INFO(get_logger(), "2");
    map_initialized_ = true;
}

void ParticleFilter::precomputeSensorModel() {
    RCLCPP_INFO(get_logger(), "precompute 1");
    if (rangelib_variant_ == 0) return;

    // Build a lookup table for sensor model with the given static map
    int table_width = max_range_px_ + 1;
    double* table = new double[table_width * table_width];

    RCLCPP_INFO(get_logger(), "precompute 2");
    // Calculate for each possible simulated LiDAR range value d and potential observed range value r
    for (int d = 0; d < table_width; d++) {
        double norm = 0.0;
        double sum_unkown = 0.0;
        for (int r = 0; r < table_width; r++) {
            double prob = 0.0;
            double z = (double)(r - d);
            prob += z_hit_ * exp(-(z * z) / (2.0 * sigma_hit_ * sigma_hit_)) / (sigma_hit_ * sqrt(M_PI));
            if (r < d) prob += 2.0 * z_short_ * (d - r) / (double)(d);
            if (r == max_range_px_) prob += z_max_;
            norm += prob;
            table[r * table_width + d] = prob;
        }
        for (int r = 0; r < table_width; r++) table[r * table_width + d] /= norm;
    }
    RCLCPP_INFO(get_logger(), "precompute 3");

    // Call for method provided in ray casting library range_libc
    (dynamic_cast<RayMarchingGPU*>(range_method_))->set_sensor_model(table, table_width);
}

void ParticleFilter::initializeGlobalDistribution() {
    RCLCPP_INFO(get_logger(), "GLOBAL INITIALIZATION");
    RecursiveLock lock(particles_mtx_);

    // Set particle distribution inside the map
    std::uniform_real_distribution<double> global_x_dist_(0, loaded_map_->info.width);
    std::uniform_real_distribution<double> global_y_dist_(0, loaded_map_->info.height);
    std::uniform_real_distribution<double> global_th_dist_(std::nextafter(-M_PI, std::numeric_limits<double>::max()), std::nextafter(+M_PI, std::numeric_limits<double>::max()));

    // Initialize all max_particles_num_ particles
    for (int i = 0; i < max_particles_num_; i++) {
        // Find a particle position which lays in permissible region
        bool occupied = true;
        int idx_x, idx_y;
        while (occupied) {
            idx_x = (int)(global_x_dist_(rng_.engine()));
            idx_y = (int)(global_y_dist_(rng_.engine()));
            occupied = (idx_x < 0 || idx_x > loaded_map_->info.width || idx_y < 0 || idx_y > loaded_map_->info.height || loaded_map_->data[idx_y * loaded_map_->info.width + idx_x]);
        }
        std::vector<int> idx = {idx_x, idx_y};
        std::vector<double> pos = mapToWorld(idx);
        ParticleState ps = {pos[0], pos[1], global_th_dist_(rng_.engine()), 1.0 / max_particles_num_};
        particles_.push_back(ps);
    }
}

void ParticleFilter::initializeParticlesPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "SETTING POSE");
    RecursiveLock lock(particles_mtx_);

    geometry_msgs::msg::Pose pose = msg->pose.pose;

    std::uniform_real_distribution<double> local_x_dist_(-0.5, 0.5);
    std::uniform_real_distribution<double> local_y_dist_(-0.5, 0.5);
    std::uniform_real_distribution<double> local_th_dist_(-0.4, 0.4);

    particles_.clear();
    // Initialize all max_particles_num_ particles
    for (int i = 0; i < max_particles_num_; i++) {
        // Find a particle position which lays in permissible region
        bool occupied = true;

        double dx, dy;
        int idx_x, idx_y;
        while (occupied) {
            dx = local_x_dist_(rng_.engine());
            dy = local_y_dist_(rng_.engine());
            std::vector<double> pos = {dx + pose.position.x, dy + pose.position.y};
            std::vector<int> idx = worldToMap(pos);
            idx_x = idx[0];
            idx_y = idx[1];
            occupied = (idx_x < 0 || idx_x > loaded_map_->info.width || idx_y < 0 || idx_y > loaded_map_->info.height || loaded_map_->data[idx_y * loaded_map_->info.width + idx_x]);
        }

        // Use tf2 for quaternion conversion in ROS 2
        tf2::Quaternion q;
        tf2::fromMsg(pose.orientation, q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        ParticleState ps = {dx + pose.position.x, dy + pose.position.y, local_th_dist_(rng_.engine()) + yaw, 1.0 / max_particles_num_};
        particles_.push_back(ps);
    }
}

void ParticleFilter::lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    if (downsampled_laser_angles_.empty()) {
        RCLCPP_INFO(get_logger(), "...Received first LiDAR message");
        angle_min_ = msg->angle_min;
        angle_increment_ = msg->angle_increment;
        for (size_t i = 0; i < msg->ranges.size(); i = i + angle_step_) {
            downsampled_laser_angles_.push_back(angle_min_ + angle_increment_ * i);
        }
        num_downsampled_angles_ = downsampled_laser_angles_.size();

        // allocate memory
        angles_ = new float[num_downsampled_angles_];
        obs_ = new float[num_downsampled_angles_];
        outs_ = new float[num_downsampled_angles_ * max_particles_num_];
        weights_ = new double[max_particles_num_];
        samples_ = new float[max_particles_num_ * 3];
        viz_queries_ = new float[num_downsampled_angles_ * 3];
        viz_ranges_ = new float[num_downsampled_angles_];

        lidar_initialized_ = true;
    }

    downsampled_laser_ranges_.clear();
    for (size_t i = 0; i < msg->ranges.size(); i = i + angle_step_) {
        downsampled_laser_ranges_.push_back(msg->ranges[i]);
    }
}

void ParticleFilter::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg) {
    if (last_pose_.empty()) {
        RCLCPP_INFO(get_logger(), "...Received first Odometry message");
    } else {
        double dx = msg->pose.pose.position.x - last_pose_[0];
        double dy = msg->pose.pose.position.y - last_pose_[1];

        // Use tf2 for quaternion conversion in ROS 2
        tf2::Quaternion q;
        tf2::fromMsg(msg->pose.pose.orientation, q);
        double roll, pitch, current_yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, current_yaw);

        double dtheta = current_yaw - last_pose_[2];
        double c = cos(-last_pose_[2]);
        double s = sin(-last_pose_[2]);
        double local_delta_x = dx * c + dy * -s;
        double local_delta_y = dy * s + dx * c;
        odometry_data_.clear();
        odometry_data_.push_back(local_delta_x);
        odometry_data_.push_back(local_delta_y);
        odometry_data_.push_back(dtheta);
        odom_initialized_ = true;
    }

    last_pose_.clear();
    last_pose_.push_back(msg->pose.pose.position.x);
    last_pose_.push_back(msg->pose.pose.position.y);

    // Use tf2 for quaternion conversion in ROS 2
    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.pose.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    last_pose_.push_back(yaw);

    last_stamp_ = msg->header.stamp;

    update();
}

void ParticleFilter::clickedPoseCB(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) { initializeParticlesPose(msg); }

void ParticleFilter::update() {
    // Execute update only when everything is ready
    if (!(lidar_initialized_ && odom_initialized_ && map_initialized_)) return;

    RecursiveLock lock(particles_mtx_);
    // MCL algorithm
    // Sampling
    sampling();
    // Motion model
    motionModel();
    // Sensor model
    sensorModel();
    // Calculate the average particle
    expectedPose();

    // Publish and visualize info only when expected pose is valid
    if (isnan(expected_pose_.x) || isnan(expected_pose_.y) || isnan(expected_pose_.theta)) return;

    // Publish tf and odom
    publishTfOdom();
    // Visualize pose, particles and scan
    visualize();
}

void ParticleFilter::sampling() {
    std::vector<int> samples;
    std::vector<double> proba;
    for (int i = 0; i < max_particles_num_; i++) {
        samples.push_back(i);
        proba.push_back(particles_[i].weight);
    }
    std::discrete_distribution<int> distribution(proba.begin(), proba.end());

    // ref: https://stackoverflow.com/questions/42926209/equivalent-function-to-numpy-random-choice-in-c
    std::vector<decltype(distribution)::result_type> indices;
    indices.reserve(max_particles_num_);  // reserve to prevent reallocation
    std::mt19937 generator = rng_.engine();
    // use a generator lambda to draw random indices based on distribution
    std::generate_n(back_inserter(indices), max_particles_num_,
                    [distribution = std::move(distribution),  // could also capture by reference (&) or construct in the capture list
                     generator]() mutable {                   // mutable required for generator
                        return distribution(generator);
                    });

    std::vector<ParticleState> new_particles_;
    for (auto const idx : indices) {
        new_particles_.push_back(particles_[idx]);
    }
    particles_ = new_particles_;
}

void ParticleFilter::motionModel() {
    Time t_start = Clock::now();

    std::normal_distribution<double> distribution1(0.0, motion_dispersion_x_);
    std::normal_distribution<double> distribution2(0.0, motion_dispersion_y_);
    std::normal_distribution<double> distribution3(0.0, motion_dispersion_theta_);
    std::mt19937 generator = rng_.engine();

    for (int i = 0; i < max_particles_num_; i++) {
        double cosine = cos(particles_[i].theta);
        double sine = sin(particles_[i].theta);

        double local_dx = cosine * odometry_data_[0] - sine * odometry_data_[1];
        double local_dy = sine * odometry_data_[0] + cosine * odometry_data_[1];
        double local_dtheta = odometry_data_[2];

        particles_[i].x += local_dx + distribution1(generator);
        particles_[i].y += local_dy + distribution2(generator);
        particles_[i].theta += local_dtheta + distribution3(generator);
    }

    Time t_end = Clock::now();

    double total_duration = durationMsec(t_start, t_end);
    RCLCPP_INFO_STREAM(get_logger(), "timing in motion_model: " << total_duration);
    motion_model_calc_worst_time_ = total_duration > motion_model_calc_worst_time_ ? total_duration : motion_model_calc_worst_time_;
    RCLCPP_INFO(get_logger(), "motion_model_calc_worst_time_: %.2f", motion_model_calc_worst_time_);
}

void ParticleFilter::sensorModel() {
    if (rangelib_variant_ == VAR_REPEAT_ANGLES_EVAL_SENSOR) {
        Time t_start = Clock::now();

        for (int i = 0; i < max_particles_num_; i++) {
            samples_[i * 3 + 0] = (float)particles_[i].x;
            samples_[i * 3 + 1] = (float)particles_[i].y;
            samples_[i * 3 + 2] = (float)particles_[i].theta;
        }

        for (int i = 0; i < num_downsampled_angles_; i++) {
            angles_[i] = (float)(downsampled_laser_angles_[i]);
            obs_[i] = (float)(downsampled_laser_ranges_[i]);
        }

        Time t_init = Clock::now();

        (dynamic_cast<RayMarchingGPU*>(range_method_))->numpy_calc_range_angles(samples_, angles_, outs_, max_particles_num_, num_downsampled_angles_);

        Time t_range = Clock::now();

        (dynamic_cast<RayMarchingGPU*>(range_method_))->eval_sensor_model(obs_, outs_, weights_, num_downsampled_angles_, max_particles_num_);

        Time t_eval = Clock::now();

        double inv_squash_factor = 1.0 / squash_factor_;
        double weight_sum = 0.0;
        for (int i = 0; i < max_particles_num_; i++) {
            weights_[i] = pow(weights_[i], inv_squash_factor);
            weight_sum += weights_[i];
        }
        for (int i = 0; i < max_particles_num_; i++) {
            particles_[i].weight = weights_[i] / weight_sum;
        }

        Time t_squash = Clock::now();

        double total_duration = durationMsec(t_init, t_squash);
        RCLCPP_INFO_STREAM(get_logger(), "timing in sensor_model: " << "\ninit: " << durationMsec(t_start, t_init) << ", range: " << durationMsec(t_init, t_range) << ", eval: "
                                                                    << durationMsec(t_range, t_eval) << ", squash: " << durationMsec(t_eval, t_squash) << " -- total: " << total_duration);
        sensor_model_calc_worst_time_ = total_duration > sensor_model_calc_worst_time_ ? total_duration : sensor_model_calc_worst_time_;
        RCLCPP_INFO(get_logger(), "sensor_model_calc_worst_time_: %.2f", sensor_model_calc_worst_time_);
    } else {
        throw std::runtime_error(
            "Not yet implemented rangelib_variant. "
            "Please check this parameter in launch file. "
            "Or modified the code in ParticleFilter::sensorModel().");
    }
}

void ParticleFilter::expectedPose() {
    // Expected pose for LiDAR
    expected_pose_.x = 0;
    expected_pose_.y = 0;
    expected_pose_.theta = 0;
    for (int i = 0; i < max_particles_num_; i++) {
        expected_pose_.x += particles_[i].weight * particles_[i].x;
        expected_pose_.y += particles_[i].weight * particles_[i].y;
        expected_pose_.theta += particles_[i].weight * particles_[i].theta;
    }
}

void ParticleFilter::publishTfOdom() {
    // Publish tf
    // Transform position of LiDAR to base_link
    double laser_base_link_offset = 0.265;  // for f1tenth_gym, it might be something like: 0.275 - 0.3302/2
    double tf_base_link_to_map_x = expected_pose_.x - laser_base_link_offset * cos(expected_pose_.theta);
    double tf_base_link_to_map_y = expected_pose_.y - laser_base_link_offset * sin(expected_pose_.theta);

    // Set up transform msg for tf2 in ROS 2
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = last_stamp_;
    transform_stamped.header.frame_id = "map";
    transform_stamped.child_frame_id = "base_link";
    transform_stamped.transform.translation.x = tf_base_link_to_map_x;
    transform_stamped.transform.translation.y = tf_base_link_to_map_y;
    transform_stamped.transform.translation.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, expected_pose_.theta);
    transform_stamped.transform.rotation = tf2::toMsg(q);

    // Broadcast transform
    tf_broadcaster_->sendTransform(transform_stamped);

    if (publish_odom_) {
        // Create quaternion for orientation
        geometry_msgs::msg::Quaternion q_msg = tf2::toMsg(q);

        // Publish odom
        auto odom = std::make_unique<nav_msgs::msg::Odometry>();
        odom->header.stamp = last_stamp_;
        odom->header.frame_id = "map";
        odom->pose.pose.position.x = expected_pose_.x;
        odom->pose.pose.position.y = expected_pose_.y;
        odom->pose.pose.orientation = q_msg;
        odom_pub_->publish(std::move(odom));
    }
}

void ParticleFilter::visualize() {
    if (!viz_) return;

    // Publish pose
    if (!(isnan(expected_pose_.x) || isnan(expected_pose_.y) || isnan(expected_pose_.theta)) && pose_pub_->get_subscription_count() > 0) {
        // Create quaternion for orientation
        tf2::Quaternion q;
        q.setRPY(0, 0, expected_pose_.theta);

        auto pose = std::make_unique<geometry_msgs::msg::PoseStamped>();
        pose->header.stamp = last_stamp_;
        pose->header.frame_id = "map";
        pose->pose.position.x = expected_pose_.x;
        pose->pose.position.y = expected_pose_.y;
        pose->pose.orientation = tf2::toMsg(q);
        pose_pub_->publish(std::move(pose));
    }

    // Visualize particles in rviz
    if (particle_pub_->get_subscription_count() > 0) {
        auto particles_ros = std::make_unique<geometry_msgs::msg::PoseArray>();
        particles_ros->header.stamp = this->now();
        particles_ros->header.frame_id = "map";
        particles_ros->poses.resize(max_particles_num_);

        for (int i = 0; i < max_particles_num_; i++) {
            geometry_msgs::msg::Pose pose_ros;
            pose_ros.position.x = particles_[i].x;
            pose_ros.position.y = particles_[i].y;
            pose_ros.position.z = 0.0;

            tf2::Quaternion q;
            q.setRPY(0, 0, particles_[i].theta);
            pose_ros.orientation = tf2::toMsg(q);

            particles_ros->poses[i] = pose_ros;
        }
        particle_pub_->publish(std::move(particles_ros));
    }

    // Publish simulated scan from the inferred position
    if (!(isnan(expected_pose_.x) || isnan(expected_pose_.y) || isnan(expected_pose_.theta)) && fake_scan_pub_->get_subscription_count() > 0) {
        double max_range = -1e+6;
        for (int i = 0; i < num_downsampled_angles_; i++) {
            viz_queries_[i * 3 + 0] = expected_pose_.x;
            viz_queries_[i * 3 + 1] = expected_pose_.y;
            viz_queries_[i * 3 + 2] = expected_pose_.theta + downsampled_laser_angles_[i];
            if (downsampled_laser_ranges_[i] > max_range) max_range = downsampled_laser_ranges_[i];
        }
        (dynamic_cast<RayMarchingGPU*>(range_method_))->numpy_calc_range(viz_queries_, viz_ranges_, num_downsampled_angles_);

        auto scan = std::make_unique<sensor_msgs::msg::LaserScan>();
        scan->header.stamp = last_stamp_;
        scan->header.frame_id = "laser";  // for f1tenth_gym, it might be something like "ego_racecar/laser"
        scan->angle_min = std::min(downsampled_laser_angles_[0], downsampled_laser_angles_[num_downsampled_angles_ - 1]);
        scan->angle_max = std::max(downsampled_laser_angles_[0], downsampled_laser_angles_[num_downsampled_angles_ - 1]);
        scan->angle_increment = abs(downsampled_laser_angles_[1] - downsampled_laser_angles_[0]);
        scan->range_min = 0.0;
        scan->range_max = max_range;
        scan->ranges.resize(num_downsampled_angles_);
        for (int i = 0; i < num_downsampled_angles_; i++) {
            scan->ranges[i] = viz_ranges_[i];
        }
        fake_scan_pub_->publish(std::move(scan));
    }
}

std::vector<int> ParticleFilter::worldToMap(std::vector<double> position) {
    std::vector<int> result;

    double x = position[0];
    double y = position[1];

    double scale = loaded_map_->info.resolution;

    tf2::Quaternion q;
    tf2::fromMsg(loaded_map_->info.origin.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    double angle = -1.0 * yaw;

    x -= loaded_map_->info.origin.position.x;
    y -= loaded_map_->info.origin.position.y;

    double c = cos(angle);
    double s = sin(angle);
    double temp = x;
    x = c * x - s * y;
    y = s * temp + c * y;

    result.push_back((int)(x / scale));
    result.push_back((int)(y / scale));

    if (position.size() == 3) {
        double theta = position[2];
        theta += angle;
        result.push_back((int)theta);
    }

    return result;
}

std::vector<double> ParticleFilter::mapToWorld(std::vector<int> idx) {
    std::vector<double> result;

    int map_x = idx[0];
    int map_y = idx[1];
    double x, y;

    double scale = loaded_map_->info.resolution;

    tf2::Quaternion q;
    tf2::fromMsg(loaded_map_->info.origin.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    double angle = yaw;
    double c = cos(angle);
    double s = sin(angle);
    double temp = map_x;
    x = c * map_x - s * map_y;
    y = s * temp + c * map_y;

    x = x * scale + loaded_map_->info.origin.position.x;
    y = y * scale + loaded_map_->info.origin.position.y;

    result.push_back(x);
    result.push_back(y);

    if (idx.size() == 3) {
        double theta = idx[2];
        theta += angle;
        result.push_back(theta);
    }

    return result;
}

int main(int argc, char** argv) {
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
