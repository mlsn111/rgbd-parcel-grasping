#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

class PlaneRansacNode : public rclcpp::Node
{
public:
  PlaneRansacNode()
  : Node("realsense_plane_ransac")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/cloud_voxel");

    plane_topic_ = declare_parameter<std::string>(
      "plane_topic", "/r10/cloud_plane");

    objects_topic_ = declare_parameter<std::string>(
      "objects_topic", "/r10/cloud_without_plane");

    distance_threshold_ = declare_parameter<double>("distance_threshold", 0.010);
    max_iterations_ = declare_parameter<int>("max_iterations", 150);
    min_inliers_ = declare_parameter<int>("min_inliers", 300);
    process_every_n_ = declare_parameter<int>("process_every_n", 3);
    log_every_n_ = declare_parameter<int>("log_every_n", 10);

    use_axis_constraint_ = declare_parameter<bool>("use_axis_constraint", true);
    axis_x_ = declare_parameter<double>("axis_x", 0.0);
    axis_y_ = declare_parameter<double>("axis_y", 0.0);
    axis_z_ = declare_parameter<double>("axis_z", 1.0);
    axis_tolerance_deg_ = declare_parameter<double>("axis_tolerance_deg", 25.0);

    if (distance_threshold_ <= 0.0 ||
        max_iterations_ <= 0 ||
        min_inliers_ <= 0 ||
        process_every_n_ <= 0) {
      throw std::runtime_error("Parametres RANSAC invalides");
    }

    Eigen::Vector3f axis(
      static_cast<float>(axis_x_),
      static_cast<float>(axis_y_),
      static_cast<float>(axis_z_));

    if (use_axis_constraint_ && axis.norm() < 1e-6f) {
      throw std::runtime_error("L'axe RANSAC ne peut pas etre nul");
    }

    rclcpp::QoS qos(rclcpp::KeepLast(2));
    qos.reliable();
    qos.durability_volatile();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&PlaneRansacNode::callback, this, std::placeholders::_1));

    plane_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      plane_topic_, qos);

    objects_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      objects_topic_, qos);

    RCLCPP_INFO(
      get_logger(),
      "R10 Plane RANSAC v2 actif\n"
      "input              : %s\n"
      "plane              : %s\n"
      "sans plan          : %s\n"
      "distance threshold : %.4f m\n"
      "max iterations     : %d\n"
      "min inliers        : %d\n"
      "process every N    : %d\n"
      "contrainte axe     : %s\n"
      "axe normal attendu : [%.3f, %.3f, %.3f]\n"
      "tolerance angle    : %.1f deg",
      input_topic_.c_str(),
      plane_topic_.c_str(),
      objects_topic_.c_str(),
      distance_threshold_,
      max_iterations_,
      min_inliers_,
      process_every_n_,
      use_axis_constraint_ ? "OUI" : "NON",
      axis_x_, axis_y_, axis_z_,
      axis_tolerance_deg_);
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    ++input_count_;
    if ((input_count_ - 1) % static_cast<std::uint64_t>(process_every_n_) != 0) {
      return;
    }

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->size() < static_cast<std::size_t>(min_inliers_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Nuage trop petit pour RANSAC: %zu points",
        cloud->size());
      return;
    }

    auto inliers = std::make_shared<pcl::PointIndices>();
    auto coefficients = std::make_shared<pcl::ModelCoefficients>();

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);

    if (use_axis_constraint_) {
      seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
      Eigen::Vector3f axis(
        static_cast<float>(axis_x_),
        static_cast<float>(axis_y_),
        static_cast<float>(axis_z_));
      axis.normalize();
      seg.setAxis(axis);
      constexpr double kPi = 3.14159265358979323846;
      seg.setEpsAngle(axis_tolerance_deg_ * kPi / 180.0);
    } else {
      seg.setModelType(pcl::SACMODEL_PLANE);
    }

    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations_);
    seg.setDistanceThreshold(distance_threshold_);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (static_cast<int>(inliers->indices.size()) < min_inliers_ ||
        coefficients->values.size() < 4)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Aucun plan valide: inliers=%zu (minimum=%d)",
        inliers->indices.size(),
        min_inliers_);
      return;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    auto plane_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto objects_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    extract.setNegative(false);
    extract.filter(*plane_cloud);

    extract.setNegative(true);
    extract.filter(*objects_cloud);

    sensor_msgs::msg::PointCloud2 plane_msg;
    sensor_msgs::msg::PointCloud2 objects_msg;

    pcl::toROSMsg(*plane_cloud, plane_msg);
    pcl::toROSMsg(*objects_cloud, objects_msg);
    plane_msg.header = msg->header;
    objects_msg.header = msg->header;

    plane_pub_->publish(plane_msg);
    objects_pub_->publish(objects_msg);

    ++processed_count_;

    const double a = coefficients->values[0];
    const double b = coefficients->values[1];
    const double c = coefficients->values[2];
    const double d = coefficients->values[3];
    const double norm = std::sqrt(a*a + b*b + c*c);
    const double distance_to_origin = norm > 0.0 ? std::abs(d) / norm : 0.0;

    const double inlier_percent =
      cloud->empty() ? 0.0 :
      100.0 * static_cast<double>(inliers->indices.size()) /
      static_cast<double>(cloud->size());

    double axis_angle_deg = -1.0;
    if (use_axis_constraint_ && norm > 0.0) {
      Eigen::Vector3d n(a / norm, b / norm, c / norm);
      Eigen::Vector3d axis(axis_x_, axis_y_, axis_z_);
      axis.normalize();
      const double dot = std::clamp(std::abs(n.dot(axis)), 0.0, 1.0);
      constexpr double kPi = 3.14159265358979323846;
      axis_angle_deg = std::acos(dot) * 180.0 / kPi;
    }

    if (log_every_n_ > 0 &&
        processed_count_ % static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      if (use_axis_constraint_) {
        RCLCPP_INFO(
          get_logger(),
          "RANSAC #%lu | input=%zu | plan=%zu (%.2f%%) | reste=%zu | "
          "n=[%.4f %.4f %.4f] | angle axe=%.2f deg | d=%.4f | "
          "distance origine=%.3f m",
          static_cast<unsigned long>(processed_count_),
          cloud->size(),
          plane_cloud->size(),
          inlier_percent,
          objects_cloud->size(),
          a, b, c,
          axis_angle_deg,
          d,
          distance_to_origin);
      } else {
        RCLCPP_INFO(
          get_logger(),
          "RANSAC #%lu | input=%zu | plan=%zu (%.2f%%) | reste=%zu | "
          "n=[%.4f %.4f %.4f] | d=%.4f | distance origine=%.3f m",
          static_cast<unsigned long>(processed_count_),
          cloud->size(),
          plane_cloud->size(),
          inlier_percent,
          objects_cloud->size(),
          a, b, c,
          d,
          distance_to_origin);
      }
    }
  }

  std::string input_topic_, plane_topic_, objects_topic_;
  double distance_threshold_;
  int max_iterations_, min_inliers_, process_every_n_, log_every_n_;
  bool use_axis_constraint_;
  double axis_x_, axis_y_, axis_z_, axis_tolerance_deg_;
  std::uint64_t input_count_{0}, processed_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr objects_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PlaneRansacNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("realsense_plane_ransac"),
      "Erreur fatale: %s",
      e.what());
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
