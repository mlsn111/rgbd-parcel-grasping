#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class MultiPlaneRansacNode : public rclcpp::Node
{
public:
  MultiPlaneRansacNode()
  : Node("realsense_multiplane_ransac")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/cloud_voxel");

    colored_topic_ = declare_parameter<std::string>(
      "colored_topic", "/r10/cloud_planes_colored");

    remainder_topic_ = declare_parameter<std::string>(
      "remainder_topic", "/r10/cloud_nonplanar");

    markers_topic_ = declare_parameter<std::string>(
      "markers_topic", "/r10/plane_markers");

    max_planes_ = declare_parameter<int>("max_planes", 5);
    distance_threshold_ = declare_parameter<double>("distance_threshold", 0.010);
    max_iterations_ = declare_parameter<int>("max_iterations", 400);
    min_plane_size_ = declare_parameter<int>("min_plane_size", 300);
    process_every_n_ = declare_parameter<int>("process_every_n", 10);
    log_every_n_ = declare_parameter<int>("log_every_n", 1);
    normal_arrow_length_ = declare_parameter<double>("normal_arrow_length", 0.10);

    if (max_planes_ <= 0 ||
        distance_threshold_ <= 0.0 ||
        max_iterations_ <= 0 ||
        min_plane_size_ <= 0 ||
        process_every_n_ <= 0 ||
        normal_arrow_length_ <= 0.0)
    {
      throw std::runtime_error("Parametres multi-plane invalides");
    }

    rclcpp::QoS qos(rclcpp::KeepLast(2));
    qos.reliable();
    qos.durability_volatile();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      qos,
      std::bind(&MultiPlaneRansacNode::callback, this, std::placeholders::_1));

    colored_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      colored_topic_, qos);

    remainder_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      remainder_topic_, qos);

    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      markers_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "R10 Multi-Plane RANSAC actif\n"
      "input              : %s\n"
      "planes colores     : %s\n"
      "reste non planaire : %s\n"
      "markers            : %s\n"
      "max planes         : %d\n"
      "distance threshold : %.3f m\n"
      "max iterations     : %d\n"
      "min plane size     : %d\n"
      "process every N    : %d",
      input_topic_.c_str(),
      colored_topic_.c_str(),
      remainder_topic_.c_str(),
      markers_topic_.c_str(),
      max_planes_,
      distance_threshold_,
      max_iterations_,
      min_plane_size_,
      process_every_n_);
  }

private:
  struct PlaneInfo
  {
    std::size_t id;
    std::size_t size;
    Eigen::Vector4f centroid;
    Eigen::Vector3f normal;
    float d;
    float distance_to_origin;
  };

  static std::array<std::uint8_t, 3> colorForIndex(std::size_t i)
  {
    static constexpr std::array<std::array<std::uint8_t, 3>, 10> palette {{
      {{255,  80,  80}},
      {{ 80, 255,  80}},
      {{ 80, 140, 255}},
      {{255, 220,  80}},
      {{255,  80, 220}},
      {{ 80, 255, 230}},
      {{255, 150,  60}},
      {{160,  90, 255}},
      {{120, 255, 120}},
      {{220, 220, 220}},
    }};
    return palette[i % palette.size()];
  }

  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    ++input_count_;

    if ((input_count_ - 1) %
        static_cast<std::uint64_t>(process_every_n_) != 0)
    {
      return;
    }

    auto original = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *original);

    if (original->size() < static_cast<std::size_t>(min_plane_size_)) {
      return;
    }

    auto remaining =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*original);

    auto colored =
      std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    colored->reserve(original->size());

    std::vector<PlaneInfo> planes;
    planes.reserve(static_cast<std::size_t>(max_planes_));

    for (int plane_id = 0; plane_id < max_planes_; ++plane_id)
    {
      if (remaining->size() < static_cast<std::size_t>(min_plane_size_)) {
        break;
      }

      auto inliers = std::make_shared<pcl::PointIndices>();
      auto coefficients = std::make_shared<pcl::ModelCoefficients>();

      pcl::SACSegmentation<pcl::PointXYZ> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setMaxIterations(max_iterations_);
      seg.setDistanceThreshold(distance_threshold_);
      seg.setInputCloud(remaining);
      seg.segment(*inliers, *coefficients);

      if (coefficients->values.size() < 4 ||
          inliers->indices.size() < static_cast<std::size_t>(min_plane_size_))
      {
        break;
      }

      auto plane_indices =
        std::make_shared<pcl::PointIndices>(*inliers);

      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(remaining);
      extract.setIndices(plane_indices);

      auto plane_cloud =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      extract.setNegative(false);
      extract.filter(*plane_cloud);

      auto next_remaining =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      extract.setNegative(true);
      extract.filter(*next_remaining);

      const auto rgb = colorForIndex(static_cast<std::size_t>(plane_id));

      for (const auto & p : *plane_cloud) {
        pcl::PointXYZRGB q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.r = rgb[0];
        q.g = rgb[1];
        q.b = rgb[2];
        colored->push_back(q);
      }

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*plane_cloud, centroid);

      Eigen::Vector3f normal(
        coefficients->values[0],
        coefficients->values[1],
        coefficients->values[2]);

      float d = coefficients->values[3];

      const float norm = normal.norm();
      if (norm > 1e-6f) {
        normal /= norm;
        d /= norm;
      }

      // Plane coefficient sign is arbitrary. Keep normals pointing roughly
      // toward +Z when possible, otherwise preserve orientation.
      if (normal.z() < 0.0f) {
        normal *= -1.0f;
        d *= -1.0f;
      }

      planes.push_back(PlaneInfo{
        static_cast<std::size_t>(plane_id),
        plane_cloud->size(),
        centroid,
        normal,
        d,
        std::abs(d)
      });

      remaining = next_remaining;
    }

    colored->width = static_cast<std::uint32_t>(colored->size());
    colored->height = 1;
    colored->is_dense = true;

    sensor_msgs::msg::PointCloud2 colored_msg;
    pcl::toROSMsg(*colored, colored_msg);
    colored_msg.header = msg->header;
    colored_pub_->publish(colored_msg);

    sensor_msgs::msg::PointCloud2 remainder_msg;
    pcl::toROSMsg(*remaining, remainder_msg);
    remainder_msg.header = msg->header;
    remainder_pub_->publish(remainder_msg);

    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker delete_all;
    delete_all.header = msg->header;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_all);

    for (const auto & plane : planes)
    {
      const auto rgb = colorForIndex(plane.id);

      visualization_msgs::msg::Marker sphere;
      sphere.header = msg->header;
      sphere.ns = "plane_centroids";
      sphere.id = static_cast<int>(plane.id * 3);
      sphere.type = visualization_msgs::msg::Marker::SPHERE;
      sphere.action = visualization_msgs::msg::Marker::ADD;
      sphere.pose.position.x = plane.centroid[0];
      sphere.pose.position.y = plane.centroid[1];
      sphere.pose.position.z = plane.centroid[2];
      sphere.pose.orientation.w = 1.0;
      sphere.scale.x = 0.025;
      sphere.scale.y = 0.025;
      sphere.scale.z = 0.025;
      sphere.color.r = static_cast<float>(rgb[0]) / 255.0f;
      sphere.color.g = static_cast<float>(rgb[1]) / 255.0f;
      sphere.color.b = static_cast<float>(rgb[2]) / 255.0f;
      sphere.color.a = 1.0f;
      marker_array.markers.push_back(sphere);

      visualization_msgs::msg::Marker text;
      text.header = msg->header;
      text.ns = "plane_labels";
      text.id = static_cast<int>(plane.id * 3 + 1);
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = plane.centroid[0];
      text.pose.position.y = plane.centroid[1] - 0.04;
      text.pose.position.z = plane.centroid[2];
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.035;
      text.color.r = static_cast<float>(rgb[0]) / 255.0f;
      text.color.g = static_cast<float>(rgb[1]) / 255.0f;
      text.color.b = static_cast<float>(rgb[2]) / 255.0f;
      text.color.a = 1.0f;
      text.text =
        "plane_" + std::to_string(plane.id) +
        " (" + std::to_string(plane.size) + " pts)";
      marker_array.markers.push_back(text);

      visualization_msgs::msg::Marker arrow;
      arrow.header = msg->header;
      arrow.ns = "plane_normals";
      arrow.id = static_cast<int>(plane.id * 3 + 2);
      arrow.type = visualization_msgs::msg::Marker::ARROW;
      arrow.action = visualization_msgs::msg::Marker::ADD;
      arrow.scale.x = 0.008;
      arrow.scale.y = 0.016;
      arrow.scale.z = 0.025;
      arrow.color.r = static_cast<float>(rgb[0]) / 255.0f;
      arrow.color.g = static_cast<float>(rgb[1]) / 255.0f;
      arrow.color.b = static_cast<float>(rgb[2]) / 255.0f;
      arrow.color.a = 1.0f;

      geometry_msgs::msg::Point p0;
      p0.x = plane.centroid[0];
      p0.y = plane.centroid[1];
      p0.z = plane.centroid[2];

      geometry_msgs::msg::Point p1;
      p1.x = plane.centroid[0] + plane.normal.x() * normal_arrow_length_;
      p1.y = plane.centroid[1] + plane.normal.y() * normal_arrow_length_;
      p1.z = plane.centroid[2] + plane.normal.z() * normal_arrow_length_;

      arrow.points.push_back(p0);
      arrow.points.push_back(p1);
      marker_array.markers.push_back(arrow);
    }

    markers_pub_->publish(marker_array);

    ++processed_count_;

    if (log_every_n_ > 0 &&
        processed_count_ % static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "MULTIPLANE #%lu | input=%zu | planes=%zu | planar=%zu | reste=%zu",
        static_cast<unsigned long>(processed_count_),
        original->size(),
        planes.size(),
        colored->size(),
        remaining->size());

      for (const auto & plane : planes) {
        RCLCPP_INFO(
          get_logger(),
          "  plane_%zu | pts=%zu | centroid=[%.3f %.3f %.3f] | "
          "n=[%.4f %.4f %.4f] | d=%.4f | dist=%.3f m",
          plane.id,
          plane.size,
          plane.centroid[0],
          plane.centroid[1],
          plane.centroid[2],
          plane.normal.x(),
          plane.normal.y(),
          plane.normal.z(),
          plane.d,
          plane.distance_to_origin);
      }
    }
  }

  std::string input_topic_;
  std::string colored_topic_;
  std::string remainder_topic_;
  std::string markers_topic_;

  int max_planes_;
  double distance_threshold_;
  int max_iterations_;
  int min_plane_size_;
  int process_every_n_;
  int log_every_n_;
  double normal_arrow_length_;

  std::uint64_t input_count_{0};
  std::uint64_t processed_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr remainder_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<MultiPlaneRansacNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("realsense_multiplane_ransac"),
      "Erreur fatale: %s",
      e.what());
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}
