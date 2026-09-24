#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class GraspCandidateFilter : public rclcpp::Node
{
public:
  GraspCandidateFilter()
  : Node("realsense_grasp_candidate_filter")
  {
    objects_topic_ = declare_parameter<std::string>(
      "objects_topic", "/r10/tracked_object_markers");

    support_topic_ = declare_parameter<std::string>(
      "support_topic", "/r10/cloud_support_plane");

    marker_topic_ = declare_parameter<std::string>(
      "marker_topic", "/r10/grasp_filtered_markers");

    grasp_pose_topic_ = declare_parameter<std::string>(
      "grasp_pose_topic", "/r10/grasp_filtered_poses");

    pregrasp_pose_topic_ = declare_parameter<std::string>(
      "pregrasp_pose_topic", "/r10/pregrasp_filtered_poses");

    max_gripper_opening_m_ = declare_parameter<double>(
      "max_gripper_opening_m", 0.080);

    finger_clearance_m_ = declare_parameter<double>(
      "finger_clearance_m", 0.005);

    pregrasp_distance_m_ = declare_parameter<double>(
      "pregrasp_distance_m", 0.10);

    min_table_clearance_m_ = declare_parameter<double>(
      "min_table_clearance_m", 0.015);

    top_k_per_object_ = declare_parameter<int>(
      "top_k_per_object", 3);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 10);

    rclcpp::QoS cloud_qos(rclcpp::KeepLast(2));
    cloud_qos.reliable();
    cloud_qos.durability_volatile();

    support_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      support_topic_,
      cloud_qos,
      std::bind(
        &GraspCandidateFilter::supportCallback,
        this,
        std::placeholders::_1));

    objects_sub_ =
      create_subscription<visualization_msgs::msg::MarkerArray>(
        objects_topic_,
        10,
        std::bind(
          &GraspCandidateFilter::objectsCallback,
          this,
          std::placeholders::_1));

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
        marker_topic_, 10);

    grasp_pose_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>(
        grasp_pose_topic_, 10);

    pregrasp_pose_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>(
        pregrasp_pose_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "R10-5B Grasp Filter actif\n"
      "objects                : %s\n"
      "support plane          : %s\n"
      "max gripper opening    : %.3f m\n"
      "finger clearance       : %.3f m\n"
      "pregrasp distance      : %.3f m\n"
      "table clearance        : %.3f m\n"
      "top K/object           : %d",
      objects_topic_.c_str(),
      support_topic_.c_str(),
      max_gripper_opening_m_,
      finger_clearance_m_,
      pregrasp_distance_m_,
      min_table_clearance_m_,
      top_k_per_object_);
  }

private:
  struct TrackedObb
  {
    int id{-1};
    std_msgs::msg::Header header;
    Eigen::Vector3d center{0,0,0};
    Eigen::Vector3d dims{0,0,0};
    Eigen::Quaterniond q{1,0,0,0};
  };

  struct Candidate
  {
    int object_id{-1};
    int closing_axis{-1};
    int approach_axis{-1};
    int approach_sign{1};

    std_msgs::msg::Header header;

    Eigen::Vector3d center{0,0,0};
    Eigen::Vector3d pregrasp{0,0,0};
    Eigen::Vector3d closing{0,1,0};
    Eigen::Vector3d approach{1,0,0};
    Eigen::Vector3d contact_plus{0,0,0};
    Eigen::Vector3d contact_minus{0,0,0};

    Eigen::Quaterniond q{1,0,0,0};

    double required_opening{0.0};
    double pregrasp_clearance{0.0};
    double contact_clearance{0.0};
    double score{0.0};
  };

  struct SupportPlane
  {
    bool valid{false};
    std_msgs::msg::Header header;
    Eigen::Vector3d centroid{0,0,0};
    Eigen::Vector3d normal{0,0,1};
    double d{0.0};
  };

  static geometry_msgs::msg::Point toPointMsg(
    const Eigen::Vector3d & p)
  {
    geometry_msgs::msg::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
  }

  static geometry_msgs::msg::Quaternion toQuatMsg(
    const Eigen::Quaterniond & q)
  {
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
  }

  static double clamp01(double x)
  {
    return std::clamp(x, 0.0, 1.0);
  }

  double signedDistance(
    const Eigen::Vector3d & p,
    const Eigen::Vector3d & n,
    double d) const
  {
    return n.dot(p) + d;
  }

  void supportCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    auto cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->size() < 50) {
      return;
    }

    Eigen::Vector4f centroid4;
    pcl::compute3DCentroid(*cloud, centroid4);

    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(
      *cloud, centroid4, covariance);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);

    if (solver.info() != Eigen::Success) {
      return;
    }

    // Smallest-variance PCA axis = plane normal.
    Eigen::Vector3d n =
      solver.eigenvectors().col(0).cast<double>().normalized();

    const Eigen::Vector3d c =
      centroid4.head<3>().cast<double>();

    double d = -n.dot(c);

    support_plane_.valid = true;
    support_plane_.header = msg->header;
    support_plane_.centroid = c;
    support_plane_.normal = n;
    support_plane_.d = d;
  }

  std::vector<TrackedObb> parseObjects(
    const visualization_msgs::msg::MarkerArray & array) const
  {
    std::vector<TrackedObb> objects;

    for (const auto & marker : array.markers)
    {
      if (marker.action != visualization_msgs::msg::Marker::ADD ||
          marker.ns != "tracked_obb" ||
          marker.type != visualization_msgs::msg::Marker::CUBE)
      {
        continue;
      }

      TrackedObb obj;
      obj.id = marker.id;
      obj.header = marker.header;

      obj.center = Eigen::Vector3d(
        marker.pose.position.x,
        marker.pose.position.y,
        marker.pose.position.z);

      obj.dims = Eigen::Vector3d(
        marker.scale.x,
        marker.scale.y,
        marker.scale.z);

      obj.q = Eigen::Quaterniond(
        marker.pose.orientation.w,
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z);

      if (obj.q.norm() < 1e-9) {
        obj.q = Eigen::Quaterniond::Identity();
      } else {
        obj.q.normalize();
      }

      if (obj.dims.minCoeff() > 0.0) {
        objects.push_back(obj);
      }
    }

    return objects;
  }

  std::vector<Candidate> generateRaw(
    const TrackedObb & obj,
    Eigen::Vector3d table_n,
    double table_d) const
  {
    // Orient support normal toward the object side.
    if (signedDistance(obj.center, table_n, table_d) < 0.0) {
      table_n *= -1.0;
      table_d *= -1.0;
    }

    const Eigen::Matrix3d R =
      obj.q.toRotationMatrix();

    std::vector<Candidate> out;
    out.reserve(12);

    for (int close_axis = 0; close_axis < 3; ++close_axis)
    {
      const Eigen::Vector3d close =
        R.col(close_axis).normalized();

      const double object_width =
        obj.dims[close_axis];

      const double required_opening =
        object_width + 2.0 * finger_clearance_m_;

      for (int approach_axis = 0;
           approach_axis < 3;
           ++approach_axis)
      {
        if (approach_axis == close_axis) {
          continue;
        }

        for (int sign : {-1, 1})
        {
          Eigen::Vector3d approach =
            static_cast<double>(sign) *
            R.col(approach_axis).normalized();

          Eigen::Vector3d closing = close;
          Eigen::Vector3d binormal =
            approach.cross(closing).normalized();

          closing =
            binormal.cross(approach).normalized();

          Eigen::Matrix3d Rg;
          Rg.col(0) = approach;
          Rg.col(1) = closing;
          Rg.col(2) = binormal;

          Candidate c;
          c.object_id = obj.id;
          c.closing_axis = close_axis;
          c.approach_axis = approach_axis;
          c.approach_sign = sign;
          c.header = obj.header;

          c.center = obj.center;
          c.pregrasp =
            obj.center - pregrasp_distance_m_ * approach;

          c.closing = closing;
          c.approach = approach;

          c.contact_plus =
            obj.center + 0.5 * object_width * closing;
          c.contact_minus =
            obj.center - 0.5 * object_width * closing;

          c.q = Eigen::Quaterniond(Rg);
          c.q.normalize();

          c.required_opening = required_opening;

          c.pregrasp_clearance =
            signedDistance(
              c.pregrasp,
              table_n,
              table_d);

          const double cp =
            signedDistance(
              c.contact_plus,
              table_n,
              table_d);

          const double cm =
            signedDistance(
              c.contact_minus,
              table_n,
              table_d);

          c.contact_clearance =
            std::min(cp, cm);

          // Hard filters:
          // 1) gripper physically cannot open enough
          if (required_opening >
              max_gripper_opening_m_)
          {
            continue;
          }

          // 2) pregrasp lies too close to/below support
          if (c.pregrasp_clearance <
              min_table_clearance_m_)
          {
            continue;
          }

          // 3) ideal finger contacts would collide with/support on table
          if (c.contact_clearance <
              min_table_clearance_m_)
          {
            continue;
          }

          // Soft scores.
          const double width_score =
            clamp01(
              (max_gripper_opening_m_ -
               required_opening) /
              max_gripper_opening_m_);

          // Top-down approach means approach ~= -table_normal.
          // side approach -> 0.5, top-down -> 1, bottom-up -> 0.
          const double approach_score =
            clamp01(
              0.5 * (1.0 - approach.dot(table_n)));

          const double clearance_score =
            clamp01(
              std::min(
                c.pregrasp_clearance,
                c.contact_clearance) / 0.050);

          c.score =
            0.45 * width_score +
            0.35 * approach_score +
            0.20 * clearance_score;

          out.push_back(c);
        }
      }
    }

    std::sort(
      out.begin(),
      out.end(),
      [](const Candidate & a, const Candidate & b) {
        return a.score > b.score;
      });

    if (top_k_per_object_ > 0 &&
        out.size() >
          static_cast<std::size_t>(top_k_per_object_))
    {
      out.resize(
        static_cast<std::size_t>(top_k_per_object_));
    }

    return out;
  }

  visualization_msgs::msg::Marker makeArrow(
    const Candidate & c,
    int id) const
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "filtered_grasp_approach";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.points.push_back(toPointMsg(c.pregrasp));
    m.points.push_back(toPointMsg(c.center));

    m.scale.x = 0.007;
    m.scale.y = 0.014;
    m.scale.z = 0.020;

    m.color.r = 0.15f;
    m.color.g = 1.0f;
    m.color.b = 0.25f;
    m.color.a = 1.0f;
    return m;
  }

  visualization_msgs::msg::Marker makeContactLine(
    const Candidate & c,
    int id) const
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "filtered_grasp_contacts";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::LINE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.points.push_back(toPointMsg(c.contact_minus));
    m.points.push_back(toPointMsg(c.contact_plus));

    m.scale.x = 0.006;

    m.color.r = 1.0f;
    m.color.g = 0.85f;
    m.color.b = 0.10f;
    m.color.a = 1.0f;
    return m;
  }

  visualization_msgs::msg::Marker makeLabel(
    const Candidate & c,
    int id,
    int rank) const
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "filtered_grasp_labels";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.pose.position =
      toPointMsg(
        c.pregrasp +
        Eigen::Vector3d(0.0, -0.025, 0.0));

    m.pose.orientation.w = 1.0;
    m.scale.z = 0.026;

    m.color.r = 1.0f;
    m.color.g = 1.0f;
    m.color.b = 1.0f;
    m.color.a = 1.0f;

    const int opening_mm =
      static_cast<int>(
        std::lround(
          c.required_opening * 1000.0));

    const int score_pct =
      static_cast<int>(
        std::lround(c.score * 100.0));

    m.text =
      "obj" +
      std::to_string(c.object_id) +
      " #" +
      std::to_string(rank) +
      " score=" +
      std::to_string(score_pct) +
      "% open=" +
      std::to_string(opening_mm) +
      "mm";

    return m;
  }

  void objectsCallback(
    const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
  {
    ++frame_count_;

    if (!support_plane_.valid) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        3000,
        "En attente de /r10/cloud_support_plane. "
        "Clique Publish Point sur la table.");
      return;
    }

    const auto objects =
      parseObjects(*msg);

    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker clear;
    clear.action =
      visualization_msgs::msg::Marker::DELETEALL;

    if (!objects.empty()) {
      clear.header = objects.front().header;
    }

    markers.markers.push_back(clear);

    geometry_msgs::msg::PoseArray grasp_poses;
    geometry_msgs::msg::PoseArray pregrasp_poses;

    if (!objects.empty()) {
      grasp_poses.header =
        objects.front().header;
      pregrasp_poses.header =
        objects.front().header;
    }

    std::size_t total_accepted = 0;

    for (const auto & obj : objects)
    {
      Eigen::Vector3d n =
        support_plane_.normal;
      double d =
        support_plane_.d;

      const auto accepted =
        generateRaw(obj, n, d);

      for (std::size_t rank = 0;
           rank < accepted.size();
           ++rank)
      {
        const auto & c =
          accepted[rank];

        const int base =
          c.object_id * 1000 +
          static_cast<int>(rank) * 10;

        markers.markers.push_back(
          makeArrow(c, base));

        markers.markers.push_back(
          makeContactLine(c, base + 1));

        markers.markers.push_back(
          makeLabel(
            c,
            base + 2,
            static_cast<int>(rank + 1)));

        geometry_msgs::msg::Pose gp;
        gp.position =
          toPointMsg(c.center);
        gp.orientation =
          toQuatMsg(c.q);

        geometry_msgs::msg::Pose pp;
        pp.position =
          toPointMsg(c.pregrasp);
        pp.orientation =
          toQuatMsg(c.q);

        grasp_poses.poses.push_back(gp);
        pregrasp_poses.poses.push_back(pp);

        ++total_accepted;
      }
    }

    marker_pub_->publish(markers);

    if (!objects.empty()) {
      grasp_pose_pub_->publish(grasp_poses);
      pregrasp_pose_pub_->publish(pregrasp_poses);
    }

    if (log_every_n_ > 0 &&
        frame_count_ %
          static_cast<std::uint64_t>(
            log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "GRASP FILTER #%lu | objects=%zu | accepted=%zu",
        static_cast<unsigned long>(frame_count_),
        objects.size(),
        total_accepted);
    }
  }

  std::string objects_topic_;
  std::string support_topic_;
  std::string marker_topic_;
  std::string grasp_pose_topic_;
  std::string pregrasp_pose_topic_;

  double max_gripper_opening_m_;
  double finger_clearance_m_;
  double pregrasp_distance_m_;
  double min_table_clearance_m_;
  int top_k_per_object_;
  int log_every_n_;

  std::uint64_t frame_count_{0};
  SupportPlane support_plane_;

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    support_sub_;

  rclcpp::Subscription<
    visualization_msgs::msg::MarkerArray>::SharedPtr
    objects_sub_;

  rclcpp::Publisher<
    visualization_msgs::msg::MarkerArray>::SharedPtr
    marker_pub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseArray>::SharedPtr
    grasp_pose_pub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseArray>::SharedPtr
    pregrasp_pose_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<GraspCandidateFilter>());
  rclcpp::shutdown();
  return 0;
}
