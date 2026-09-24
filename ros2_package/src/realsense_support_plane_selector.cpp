#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/common/centroid.h>
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
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class SupportPlaneSelectorNode : public rclcpp::Node
{
public:
  SupportPlaneSelectorNode()
  : Node("realsense_support_plane_selector")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/cloud_voxel");

    clicked_point_topic_ = declare_parameter<std::string>(
      "clicked_point_topic", "/clicked_point");

    support_topic_ = declare_parameter<std::string>(
      "support_topic", "/r10/cloud_support_plane");

    without_support_topic_ = declare_parameter<std::string>(
      "without_support_topic", "/r10/cloud_without_support");

    markers_topic_ = declare_parameter<std::string>(
      "markers_topic", "/r10/support_plane_marker");

    max_planes_ = declare_parameter<int>("max_planes", 5);

    // Legacy fallback only. Default -1 means: select by RViz Publish Point.
    calibration_plane_id_ = declare_parameter<int>(
      "calibration_plane_id", -1);

    distance_threshold_ = declare_parameter<double>(
      "distance_threshold", 0.010);

    max_iterations_ = declare_parameter<int>(
      "max_iterations", 400);

    min_plane_size_ = declare_parameter<int>(
      "min_plane_size", 300);

    process_every_n_ = declare_parameter<int>(
      "process_every_n", 10);

    max_normal_angle_deg_ = declare_parameter<double>(
      "max_normal_angle_deg", 15.0);

    max_plane_offset_m_ = declare_parameter<double>(
      "max_plane_offset_m", 0.030);

    click_max_distance_m_ = declare_parameter<double>(
      "click_max_distance_m", 0.080);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 1);

    if (max_planes_ <= 0 ||
        distance_threshold_ <= 0.0 ||
        max_iterations_ <= 0 ||
        min_plane_size_ <= 0 ||
        process_every_n_ <= 0 ||
        max_normal_angle_deg_ <= 0.0 ||
        max_plane_offset_m_ <= 0.0 ||
        click_max_distance_m_ <= 0.0)
    {
      throw std::runtime_error("Parametres support-plane invalides");
    }

    rclcpp::QoS cloud_qos(rclcpp::KeepLast(2));
    cloud_qos.reliable();
    cloud_qos.durability_volatile();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      cloud_qos,
      std::bind(&SupportPlaneSelectorNode::cloudCallback,
                this, std::placeholders::_1));

    clicked_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      clicked_point_topic_,
      10,
      std::bind(&SupportPlaneSelectorNode::clickedPointCallback,
                this, std::placeholders::_1));

    support_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      support_topic_, cloud_qos);

    without_support_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      without_support_topic_, cloud_qos);

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      markers_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "R10 Support Plane Selector v2 actif\n"
      "input                 : %s\n"
      "clic RViz             : %s\n"
      "support               : %s\n"
      "sans support          : %s\n"
      "legacy plane id       : %d\n"
      "max planes            : %d\n"
      "distance threshold    : %.3f m\n"
      "normal tolerance      : %.1f deg\n"
      "offset tolerance      : %.3f m\n"
      "click max distance    : %.3f m\n"
      "etat                  : %s",
      input_topic_.c_str(),
      clicked_point_topic_.c_str(),
      support_topic_.c_str(),
      without_support_topic_.c_str(),
      calibration_plane_id_,
      max_planes_,
      distance_threshold_,
      max_normal_angle_deg_,
      max_plane_offset_m_,
      click_max_distance_m_,
      calibration_plane_id_ >= 0
        ? "legacy plane-id calibration disponible"
        : "EN ATTENTE D'UN CLIC RVIZ SUR LA TABLE");
  }

private:
  struct PlaneCandidate
  {
    int id{-1};
    std::size_t size{0};
    Eigen::Vector4f centroid{0.f, 0.f, 0.f, 0.f};
    Eigen::Vector3f normal{0.f, 0.f, 1.f};
    float d{0.f};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{
      new pcl::PointCloud<pcl::PointXYZ>()};
  };

  static void canonicalize(Eigen::Vector3f & n, float & d)
  {
    const float ax = std::abs(n.x());
    const float ay = std::abs(n.y());
    const float az = std::abs(n.z());

    float dominant = n.x();
    if (ay > ax && ay >= az) {
      dominant = n.y();
    } else if (az > ax && az > ay) {
      dominant = n.z();
    }

    if (dominant < 0.0f) {
      n *= -1.0f;
      d *= -1.0f;
    }
  }

  static double angleDeg(
    const Eigen::Vector3f & a,
    const Eigen::Vector3f & b)
  {
    const double dot = std::clamp(
      static_cast<double>(std::abs(a.dot(b))),
      0.0,
      1.0);

    constexpr double kPi = 3.14159265358979323846;
    return std::acos(dot) * 180.0 / kPi;
  }

  std::vector<PlaneCandidate> extractPlanes(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & input)
  {
    std::vector<PlaneCandidate> planes;
    planes.reserve(static_cast<std::size_t>(max_planes_));

    auto remaining =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*input);

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

      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(remaining);
      extract.setIndices(inliers);

      auto plane_cloud =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      extract.setNegative(false);
      extract.filter(*plane_cloud);

      auto next_remaining =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      extract.setNegative(true);
      extract.filter(*next_remaining);

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*plane_cloud, centroid);

      Eigen::Vector3f normal(
        coefficients->values[0],
        coefficients->values[1],
        coefficients->values[2]);

      float d = coefficients->values[3];

      const float norm = normal.norm();
      if (norm <= 1e-6f) {
        remaining = next_remaining;
        continue;
      }

      normal /= norm;
      d /= norm;
      canonicalize(normal, d);

      PlaneCandidate candidate;
      candidate.id = plane_id;
      candidate.size = plane_cloud->size();
      candidate.centroid = centroid;
      candidate.normal = normal;
      candidate.d = d;
      candidate.cloud = plane_cloud;
      planes.push_back(candidate);

      remaining = next_remaining;
    }

    return planes;
  }

  static double nearestPointDistance(
    const PlaneCandidate & plane,
    const Eigen::Vector3f & click)
  {
    double best_sq = std::numeric_limits<double>::infinity();

    for (const auto & p : *plane.cloud) {
      const double dx = static_cast<double>(p.x) - click.x();
      const double dy = static_cast<double>(p.y) - click.y();
      const double dz = static_cast<double>(p.z) - click.z();
      const double sq = dx*dx + dy*dy + dz*dz;
      if (sq < best_sq) {
        best_sq = sq;
      }
    }

    return std::sqrt(best_sq);
  }

  void clickedPointCallback(
    const geometry_msgs::msg::PointStamped::ConstSharedPtr msg)
  {
    pending_click_ = *msg;

    RCLCPP_WARN(
      get_logger(),
      "CLIC RVIZ recu | frame=%s | p=[%.3f %.3f %.3f] m | "
      "selection au prochain nuage...",
      msg->header.frame_id.c_str(),
      msg->point.x,
      msg->point.y,
      msg->point.z);
  }

  void lockReferenceFromCandidate(
    const PlaneCandidate & candidate,
    const char * source)
  {
    ref_normal_ = candidate.normal;
    ref_d_ = candidate.d;
    ref_centroid_ = candidate.centroid;
    reference_locked_ = true;

    RCLCPP_WARN(
      get_logger(),
      "REFERENCE SUPPORT VERROUILLEE (%s) | candidate plane_%d | "
      "pts=%zu | centroid=[%.3f %.3f %.3f] | "
      "n=[%.4f %.4f %.4f] | d=%.4f",
      source,
      candidate.id,
      candidate.size,
      candidate.centroid[0],
      candidate.centroid[1],
      candidate.centroid[2],
      candidate.normal.x(),
      candidate.normal.y(),
      candidate.normal.z(),
      candidate.d);
  }

  void tryClickSelection(
    const std_msgs::msg::Header & cloud_header,
    const std::vector<PlaneCandidate> & planes)
  {
    if (!pending_click_.has_value()) {
      return;
    }

    const auto click_msg = *pending_click_;
    pending_click_.reset();

    if (click_msg.header.frame_id != cloud_header.frame_id) {
      RCLCPP_ERROR(
        get_logger(),
        "Clic rejete: frame du clic='%s' mais cloud='%s'. "
        "Dans RViz, mets Fixed Frame=%s puis reclique.",
        click_msg.header.frame_id.c_str(),
        cloud_header.frame_id.c_str(),
        cloud_header.frame_id.c_str());
      return;
    }

    const Eigen::Vector3f click(
      static_cast<float>(click_msg.point.x),
      static_cast<float>(click_msg.point.y),
      static_cast<float>(click_msg.point.z));

    int best_index = -1;
    double best_distance = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < planes.size(); ++i) {
      const double distance = nearestPointDistance(planes[i], click);
      if (distance < best_distance) {
        best_distance = distance;
        best_index = static_cast<int>(i);
      }
    }

    if (best_index < 0 || best_distance > click_max_distance_m_) {
      RCLCPP_ERROR(
        get_logger(),
        "Clic trop loin de tous les plans candidats: nearest=%.3f m "
        "(max=%.3f m). Reclique directement sur la surface coloree de la table.",
        best_distance,
        click_max_distance_m_);
      return;
    }

    const auto & selected = planes[static_cast<std::size_t>(best_index)];

    RCLCPP_WARN(
      get_logger(),
      "CLIC associe a plane_%d | nearest point=%.3f m",
      selected.id,
      best_distance);

    lockReferenceFromCandidate(selected, "clic RViz");
  }

  void publishMarker(
    const std_msgs::msg::Header & header,
    const PlaneCandidate & selected,
    double angle_deg,
    double offset_diff)
  {
    visualization_msgs::msg::MarkerArray array;

    visualization_msgs::msg::Marker clear;
    clear.header = header;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);

    visualization_msgs::msg::Marker sphere;
    sphere.header = header;
    sphere.ns = "support_plane";
    sphere.id = 0;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = selected.centroid[0];
    sphere.pose.position.y = selected.centroid[1];
    sphere.pose.position.z = selected.centroid[2];
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = 0.035;
    sphere.scale.y = 0.035;
    sphere.scale.z = 0.035;
    sphere.color.r = 1.0f;
    sphere.color.g = 1.0f;
    sphere.color.b = 1.0f;
    sphere.color.a = 1.0f;
    array.markers.push_back(sphere);

    visualization_msgs::msg::Marker text;
    text.header = header;
    text.ns = "support_plane";
    text.id = 1;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = selected.centroid[0];
    text.pose.position.y = selected.centroid[1] - 0.05;
    text.pose.position.z = selected.centroid[2];
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.04;
    text.color.r = 1.0f;
    text.color.g = 1.0f;
    text.color.b = 1.0f;
    text.color.a = 1.0f;
    text.text =
      "SUPPORT | angle=" + std::to_string(angle_deg).substr(0, 5) +
      " deg | dd=" + std::to_string(offset_diff).substr(0, 5) + " m";
    array.markers.push_back(text);

    marker_pub_->publish(array);
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    ++input_count_;

    if ((input_count_ - 1) %
        static_cast<std::uint64_t>(process_every_n_) != 0)
    {
      return;
    }

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->size() < static_cast<std::size_t>(min_plane_size_)) {
      return;
    }

    auto planes = extractPlanes(cloud);

    if (planes.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Aucun plan candidat.");
      return;
    }

    // Preferred selection path: click directly on the support in RViz.
    if (pending_click_.has_value()) {
      tryClickSelection(msg->header, planes);
    }

    // Legacy fallback: plane id, only if no click reference exists yet.
    if (!reference_locked_ && calibration_plane_id_ >= 0) {
      auto it = std::find_if(
        planes.begin(),
        planes.end(),
        [this](const PlaneCandidate & p) {
          return p.id == calibration_plane_id_;
        });

      if (it != planes.end()) {
        lockReferenceFromCandidate(*it, "legacy plane id");
      }
    }

    if (!reference_locked_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "EN ATTENTE: dans RViz, Fixed Frame='%s', puis clique "
        "'Publish Point' et clique directement sur la table.",
        msg->header.frame_id.c_str());
      return;
    }

    int best_index = -1;
    double best_score = std::numeric_limits<double>::infinity();
    double best_angle = 0.0;
    double best_offset = 0.0;

    for (std::size_t i = 0; i < planes.size(); ++i)
    {
      const double angle = angleDeg(planes[i].normal, ref_normal_);
      const double offset =
        std::abs(static_cast<double>(planes[i].d - ref_d_));

      if (angle > max_normal_angle_deg_ ||
          offset > max_plane_offset_m_)
      {
        continue;
      }

      const double cx =
        static_cast<double>(planes[i].centroid[0] - ref_centroid_[0]);
      const double cy =
        static_cast<double>(planes[i].centroid[1] - ref_centroid_[1]);
      const double cz =
        static_cast<double>(planes[i].centroid[2] - ref_centroid_[2]);
      const double centroid_distance =
        std::sqrt(cx*cx + cy*cy + cz*cz);

      // Main gates: orientation + plane offset.
      // Centroid is only a weak tie-breaker because visible table area can
      // change with occlusions.
      const double score =
        angle / max_normal_angle_deg_ +
        offset / max_plane_offset_m_ +
        0.15 * centroid_distance;

      if (score < best_score) {
        best_score = score;
        best_index = static_cast<int>(i);
        best_angle = angle;
        best_offset = offset;
      }
    }

    if (best_index < 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "Support non retrouve | angle<=%.1f deg, |dd|<=%.3f m. "
        "Tu peux recliquer sur la table pour recalibrer.",
        max_normal_angle_deg_,
        max_plane_offset_m_);
      return;
    }

    const auto & selected = planes[static_cast<std::size_t>(best_index)];

    auto support_cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto without_support =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    support_cloud->reserve(cloud->size());
    without_support->reserve(cloud->size());

    for (const auto & p : *cloud)
    {
      const float signed_distance =
        selected.normal.x() * p.x +
        selected.normal.y() * p.y +
        selected.normal.z() * p.z +
        selected.d;

      if (std::abs(signed_distance) <=
          static_cast<float>(distance_threshold_))
      {
        support_cloud->push_back(p);
      } else {
        without_support->push_back(p);
      }
    }

    support_cloud->width =
      static_cast<std::uint32_t>(support_cloud->size());
    support_cloud->height = 1;
    support_cloud->is_dense = true;

    without_support->width =
      static_cast<std::uint32_t>(without_support->size());
    without_support->height = 1;
    without_support->is_dense = true;

    sensor_msgs::msg::PointCloud2 support_msg;
    pcl::toROSMsg(*support_cloud, support_msg);
    support_msg.header = msg->header;
    support_pub_->publish(support_msg);

    sensor_msgs::msg::PointCloud2 without_msg;
    pcl::toROSMsg(*without_support, without_msg);
    without_msg.header = msg->header;
    without_support_pub_->publish(without_msg);

    publishMarker(msg->header, selected, best_angle, best_offset);

    ++processed_count_;

    if (log_every_n_ > 0 &&
        processed_count_ %
          static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "SUPPORT #%lu | current_plane=%d | candidate_pts=%zu | "
        "support_removed=%zu | reste=%zu | angle_ref=%.2f deg | "
        "offset_ref=%.4f m | score=%.3f | "
        "centroid=[%.3f %.3f %.3f] | n=[%.4f %.4f %.4f] | d=%.4f",
        static_cast<unsigned long>(processed_count_),
        selected.id,
        selected.size,
        support_cloud->size(),
        without_support->size(),
        best_angle,
        best_offset,
        best_score,
        selected.centroid[0],
        selected.centroid[1],
        selected.centroid[2],
        selected.normal.x(),
        selected.normal.y(),
        selected.normal.z(),
        selected.d);
    }
  }

  std::string input_topic_;
  std::string clicked_point_topic_;
  std::string support_topic_;
  std::string without_support_topic_;
  std::string markers_topic_;

  int max_planes_;
  int calibration_plane_id_;
  double distance_threshold_;
  int max_iterations_;
  int min_plane_size_;
  int process_every_n_;
  double max_normal_angle_deg_;
  double max_plane_offset_m_;
  double click_max_distance_m_;
  int log_every_n_;

  bool reference_locked_{false};
  Eigen::Vector3f ref_normal_{0.f, 0.f, 1.f};
  Eigen::Vector4f ref_centroid_{0.f, 0.f, 0.f, 0.f};
  float ref_d_{0.f};

  std::optional<geometry_msgs::msg::PointStamped> pending_click_;

  std::uint64_t input_count_{0};
  std::uint64_t processed_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr clicked_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr support_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr without_support_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<SupportPlaneSelectorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("realsense_support_plane_selector"),
      "Erreur fatale: %s",
      e.what());
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}
