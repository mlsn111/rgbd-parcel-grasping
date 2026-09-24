#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class GraspCandidateGenerator : public rclcpp::Node
{
public:
  GraspCandidateGenerator()
  : Node("realsense_grasp_candidate_generator")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/tracked_object_markers");

    marker_topic_ = declare_parameter<std::string>(
      "marker_topic", "/r10/grasp_candidate_markers");

    grasp_pose_topic_ = declare_parameter<std::string>(
      "grasp_pose_topic", "/r10/grasp_candidate_poses");

    pregrasp_pose_topic_ = declare_parameter<std::string>(
      "pregrasp_pose_topic", "/r10/pregrasp_candidate_poses");

    pregrasp_distance_ = declare_parameter<double>(
      "pregrasp_distance", 0.10);

    finger_clearance_ = declare_parameter<double>(
      "finger_clearance", 0.005);

    contact_marker_radius_ = declare_parameter<double>(
      "contact_marker_radius", 0.008);

    max_objects_ = declare_parameter<int>(
      "max_objects", 10);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 10);

    sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      input_topic_,
      10,
      std::bind(
        &GraspCandidateGenerator::callback,
        this,
        std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic_, 10);

    grasp_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      grasp_pose_topic_, 10);

    pregrasp_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      pregrasp_pose_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "R10-5A Grasp Candidate Generator actif\n"
      "input              : %s\n"
      "markers            : %s\n"
      "grasp poses        : %s\n"
      "pregrasp poses     : %s\n"
      "pregrasp distance  : %.3f m\n"
      "finger clearance   : %.3f m",
      input_topic_.c_str(),
      marker_topic_.c_str(),
      grasp_pose_topic_.c_str(),
      pregrasp_pose_topic_.c_str(),
      pregrasp_distance_,
      finger_clearance_);
  }

private:
  struct TrackedObb
  {
    int object_id{-1};
    std_msgs::msg::Header header;
    Eigen::Vector3d center{0.0, 0.0, 0.0};
    Eigen::Vector3d dims{0.0, 0.0, 0.0};
    Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};
  };

  struct Candidate
  {
    int object_id{-1};
    int closing_axis{-1};
    int approach_axis{-1};
    int approach_sign{1};

    std_msgs::msg::Header header;

    Eigen::Vector3d grasp_center{0.0, 0.0, 0.0};
    Eigen::Vector3d pregrasp_center{0.0, 0.0, 0.0};

    Eigen::Vector3d closing{0.0, 1.0, 0.0};
    Eigen::Vector3d approach{1.0, 0.0, 0.0};
    Eigen::Vector3d binormal{0.0, 0.0, 1.0};

    Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};

    Eigen::Vector3d contact_plus{0.0, 0.0, 0.0};
    Eigen::Vector3d contact_minus{0.0, 0.0, 0.0};

    double object_width{0.0};
    double required_opening{0.0};
  };

  static geometry_msgs::msg::Quaternion toQuaternionMsg(
    const Eigen::Quaterniond & q)
  {
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
  }

  static geometry_msgs::msg::Point toPointMsg(
    const Eigen::Vector3d & p)
  {
    geometry_msgs::msg::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
  }

  static std::array<float, 3> colorForClosingAxis(int axis)
  {
    if (axis == 0) return {1.0f, 0.25f, 0.25f};
    if (axis == 1) return {0.25f, 1.0f, 0.25f};
    return {0.25f, 0.45f, 1.0f};
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
      obj.object_id = marker.id;
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

      if (obj.dims.minCoeff() <= 0.0) {
        continue;
      }

      objects.push_back(obj);

      if (static_cast<int>(objects.size()) >= max_objects_) {
        break;
      }
    }

    return objects;
  }

  std::vector<Candidate> generateCandidates(
    const TrackedObb & obj) const
  {
    std::vector<Candidate> candidates;
    candidates.reserve(12);

    const Eigen::Matrix3d R = obj.q.toRotationMatrix();

    // 3 possible closing axes.
    for (int closing_axis = 0; closing_axis < 3; ++closing_axis)
    {
      Eigen::Vector3d closing =
        R.col(closing_axis).normalized();

      const double object_width = obj.dims[closing_axis];

      // The approach must be perpendicular to the jaw-closing axis.
      // For each closing axis there are 2 remaining OBB axes,
      // and each can be approached from + or - side: 2*2 = 4.
      for (int approach_axis = 0; approach_axis < 3; ++approach_axis)
      {
        if (approach_axis == closing_axis) {
          continue;
        }

        for (int sign : {-1, 1})
        {
          Eigen::Vector3d approach =
            static_cast<double>(sign) *
            R.col(approach_axis).normalized();

          // Gripper-frame convention used only for candidate visualization:
          //   +X_g = approach direction (pregrasp -> grasp)
          //   +Y_g = jaw closing axis
          //   +Z_g = X_g x Y_g
          Eigen::Vector3d binormal =
            approach.cross(closing).normalized();

          // Numerical re-orthogonalization.
          closing = binormal.cross(approach).normalized();

          Eigen::Matrix3d Rg;
          Rg.col(0) = approach;
          Rg.col(1) = closing;
          Rg.col(2) = binormal;

          Eigen::Quaterniond qg(Rg);
          qg.normalize();

          Candidate c;
          c.object_id = obj.object_id;
          c.closing_axis = closing_axis;
          c.approach_axis = approach_axis;
          c.approach_sign = sign;
          c.header = obj.header;
          c.grasp_center = obj.center;
          c.pregrasp_center =
            obj.center - pregrasp_distance_ * approach;
          c.closing = closing;
          c.approach = approach;
          c.binormal = binormal;
          c.q = qg;
          c.object_width = object_width;
          c.required_opening =
            object_width + 2.0 * finger_clearance_;

          // Idealized opposing contacts on the two OBB faces.
          c.contact_plus =
            obj.center + 0.5 * object_width * closing;
          c.contact_minus =
            obj.center - 0.5 * object_width * closing;

          candidates.push_back(c);
        }
      }
    }

    return candidates;
  }

  static visualization_msgs::msg::Marker makeArrow(
    const Candidate & c,
    int id,
    const std::array<float, 3> & color)
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "grasp_approach";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.points.push_back(toPointMsg(c.pregrasp_center));
    m.points.push_back(toPointMsg(c.grasp_center));

    m.scale.x = 0.006;
    m.scale.y = 0.012;
    m.scale.z = 0.018;

    m.color.r = color[0];
    m.color.g = color[1];
    m.color.b = color[2];
    m.color.a = 1.0f;
    return m;
  }

  visualization_msgs::msg::Marker makeContactLine(
    const Candidate & c,
    int id,
    const std::array<float, 3> & color) const
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "grasp_contacts";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::LINE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.points.push_back(toPointMsg(c.contact_minus));
    m.points.push_back(toPointMsg(c.contact_plus));

    m.scale.x = 0.005;

    m.color.r = color[0];
    m.color.g = color[1];
    m.color.b = color[2];
    m.color.a = 1.0f;
    return m;
  }

  visualization_msgs::msg::Marker makeContactSphere(
    const Candidate & c,
    const Eigen::Vector3d & p,
    int id,
    const std::array<float, 3> & color) const
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "grasp_contact_points";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position = toPointMsg(p);
    m.pose.orientation.w = 1.0;
    m.scale.x = contact_marker_radius_ * 2.0;
    m.scale.y = contact_marker_radius_ * 2.0;
    m.scale.z = contact_marker_radius_ * 2.0;
    m.color.r = color[0];
    m.color.g = color[1];
    m.color.b = color[2];
    m.color.a = 0.9f;
    return m;
  }

  static visualization_msgs::msg::Marker makeLabel(
    const Candidate & c,
    int id,
    const std::array<float, 3> & color)
  {
    visualization_msgs::msg::Marker m;
    m.header = c.header;
    m.ns = "grasp_labels";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.pose.position = toPointMsg(
      c.pregrasp_center + Eigen::Vector3d(0.0, -0.02, 0.0));
    m.pose.orientation.w = 1.0;

    m.scale.z = 0.024;

    m.color.r = color[0];
    m.color.g = color[1];
    m.color.b = color[2];
    m.color.a = 1.0f;

    const int width_mm =
      static_cast<int>(std::lround(c.required_opening * 1000.0));

    m.text =
      "obj" + std::to_string(c.object_id) +
      " close=" + std::to_string(c.closing_axis) +
      " appr=" + std::to_string(c.approach_axis) +
      (c.approach_sign > 0 ? "+" : "-") +
      " open=" + std::to_string(width_mm) + "mm";

    return m;
  }

  void callback(
    const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
  {
    const auto objects = parseObjects(*msg);

    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    if (!objects.empty()) {
      clear.header = objects.front().header;
    }
    markers.markers.push_back(clear);

    geometry_msgs::msg::PoseArray grasp_poses;
    geometry_msgs::msg::PoseArray pregrasp_poses;

    if (!objects.empty()) {
      grasp_poses.header = objects.front().header;
      pregrasp_poses.header = objects.front().header;
    }

    std::size_t total_candidates = 0;

    for (const auto & obj : objects)
    {
      const auto candidates = generateCandidates(obj);

      for (std::size_t k = 0; k < candidates.size(); ++k)
      {
        const auto & c = candidates[k];

        // Stable visualization IDs:
        // each tracked object owns a block of 1000 IDs.
        const int base =
          obj.object_id * 1000 + static_cast<int>(k) * 10;

        const auto color =
          colorForClosingAxis(c.closing_axis);

        markers.markers.push_back(
          makeArrow(c, base + 0, color));

        markers.markers.push_back(
          makeContactLine(c, base + 1, color));

        markers.markers.push_back(
          makeContactSphere(
            c, c.contact_minus, base + 2, color));

        markers.markers.push_back(
          makeContactSphere(
            c, c.contact_plus, base + 3, color));

        markers.markers.push_back(
          makeLabel(c, base + 4, color));

        geometry_msgs::msg::Pose grasp_pose;
        grasp_pose.position = toPointMsg(c.grasp_center);
        grasp_pose.orientation = toQuaternionMsg(c.q);
        grasp_poses.poses.push_back(grasp_pose);

        geometry_msgs::msg::Pose pregrasp_pose;
        pregrasp_pose.position = toPointMsg(c.pregrasp_center);
        pregrasp_pose.orientation = toQuaternionMsg(c.q);
        pregrasp_poses.poses.push_back(pregrasp_pose);

        ++total_candidates;
      }
    }

    marker_pub_->publish(markers);

    if (!objects.empty()) {
      grasp_pose_pub_->publish(grasp_poses);
      pregrasp_pose_pub_->publish(pregrasp_poses);
    }

    ++frame_count_;

    if (log_every_n_ > 0 &&
        frame_count_ % static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "GRASP CANDIDATES #%lu | objects=%zu | candidates=%zu",
        static_cast<unsigned long>(frame_count_),
        objects.size(),
        total_candidates);
    }
  }

  std::string input_topic_;
  std::string marker_topic_;
  std::string grasp_pose_topic_;
  std::string pregrasp_pose_topic_;

  double pregrasp_distance_;
  double finger_clearance_;
  double contact_marker_radius_;
  int max_objects_;
  int log_every_n_;

  std::uint64_t frame_count_{0};

  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr grasp_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pregrasp_pose_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GraspCandidateGenerator>());
  rclcpp::shutdown();
  return 0;
}
