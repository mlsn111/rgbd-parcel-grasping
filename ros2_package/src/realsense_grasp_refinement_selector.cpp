#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

class GraspRefinementSelector : public rclcpp::Node
{
public:
  GraspRefinementSelector()
  : Node("realsense_grasp_refinement_selector")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/grasp_antipodal_markers");

    marker_topic_ = declare_parameter<std::string>(
      "marker_topic", "/r10/grasp_selected_markers");

    grasp_pose_topic_ = declare_parameter<std::string>(
      "grasp_pose_topic", "/r10/grasp_selected_poses");

    pregrasp_pose_topic_ = declare_parameter<std::string>(
      "pregrasp_pose_topic", "/r10/pregrasp_selected_poses");

    max_gripper_opening_m_ = declare_parameter<double>(
      "max_gripper_opening_m", 0.080);

    finger_clearance_m_ = declare_parameter<double>(
      "finger_clearance_m", 0.005);

    pregrasp_distance_m_ = declare_parameter<double>(
      "pregrasp_distance_m", 0.10);

    min_antipodal_score_ = declare_parameter<double>(
      "min_antipodal_score", 0.55);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 10);

    sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      input_topic_,
      10,
      std::bind(
        &GraspRefinementSelector::callback,
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
      "R10-5D Grasp Refinement + Selection actif\n"
      "input                 : %s\n"
      "selected markers      : %s\n"
      "max gripper opening   : %.3f m\n"
      "finger clearance      : %.3f m\n"
      "pregrasp distance     : %.3f m\n"
      "min antipodal score   : %.2f",
      input_topic_.c_str(),
      marker_topic_.c_str(),
      max_gripper_opening_m_,
      finger_clearance_m_,
      pregrasp_distance_m_,
      min_antipodal_score_);
  }

private:
  struct Candidate
  {
    int anti_base{-1};
    int object_id{-1};
    int rank{-1};

    std_msgs::msg::Header header;

    bool accepted{false};
    bool has_approach{false};
    bool has_contacts{false};
    bool has_label{false};

    double antipodal_score{0.0};

    Eigen::Vector3d old_pregrasp{0,0,0};
    Eigen::Vector3d old_center{0,0,0};

    Eigen::Vector3d observed_contact0{0,0,0};
    Eigen::Vector3d observed_contact1{0,0,0};
  };

  struct Refined
  {
    Candidate source;

    Eigen::Vector3d center{0,0,0};
    Eigen::Vector3d pregrasp{0,0,0};

    Eigen::Vector3d approach{1,0,0};
    Eigen::Vector3d closing{0,1,0};
    Eigen::Vector3d binormal{0,0,1};

    Eigen::Quaterniond q{1,0,0,0};

    double observed_width{0.0};
    double required_opening{0.0};
  };

  static Eigen::Vector3d pointToEigen(
    const geometry_msgs::msg::Point & p)
  {
    return Eigen::Vector3d(p.x, p.y, p.z);
  }

  static geometry_msgs::msg::Point eigenToPoint(
    const Eigen::Vector3d & p)
  {
    geometry_msgs::msg::Point out;
    out.x = p.x();
    out.y = p.y();
    out.z = p.z();
    return out;
  }

  static geometry_msgs::msg::Quaternion quatToMsg(
    const Eigen::Quaterniond & q)
  {
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
  }

  static int recoverObjectId(int anti_base)
  {
    return anti_base / 10000;
  }

  static int recoverRank(int anti_base)
  {
    return ((anti_base % 10000) / 100) + 1;
  }

  static std::optional<double> parseAcceptedScore(
    const std::string & text)
  {
    if (text.find("ANTIPODAL") == std::string::npos) {
      return std::nullopt;
    }

    static const std::regex re(
      R"(ANTIPODAL\s+([0-9]+)%)");

    std::smatch match;

    if (!std::regex_search(text, match, re) ||
        match.size() < 2)
    {
      return std::nullopt;
    }

    const double pct =
      std::stod(match[1].str());

    return std::clamp(pct / 100.0, 0.0, 1.0);
  }

  std::vector<Candidate> parseCandidates(
    const visualization_msgs::msg::MarkerArray & array) const
  {
    std::map<int, Candidate> by_base;

    for (const auto & marker : array.markers)
    {
      if (marker.action !=
          visualization_msgs::msg::Marker::ADD)
      {
        continue;
      }

      if (marker.ns == "antipodal_approach" &&
          marker.type ==
            visualization_msgs::msg::Marker::ARROW &&
          marker.points.size() >= 2)
      {
        const int base = marker.id;
        auto & c = by_base[base];

        c.anti_base = base;
        c.object_id = recoverObjectId(base);
        c.rank = recoverRank(base);
        c.header = marker.header;

        c.old_pregrasp =
          pointToEigen(marker.points[0]);

        c.old_center =
          pointToEigen(marker.points[1]);

        c.has_approach = true;
      }
      else if (
        marker.ns == "antipodal_contacts" &&
        marker.type ==
          visualization_msgs::msg::Marker::LINE_LIST &&
        marker.points.size() >= 2)
      {
        const int base = marker.id - 4;
        auto & c = by_base[base];

        c.anti_base = base;
        c.object_id = recoverObjectId(base);
        c.rank = recoverRank(base);
        c.header = marker.header;

        c.observed_contact0 =
          pointToEigen(marker.points[0]);

        c.observed_contact1 =
          pointToEigen(marker.points[1]);

        c.has_contacts = true;
      }
      else if (
        marker.ns == "antipodal_labels" &&
        marker.type ==
          visualization_msgs::msg::Marker::TEXT_VIEW_FACING)
      {
        const int base = marker.id - 1;
        auto & c = by_base[base];

        c.anti_base = base;
        c.object_id = recoverObjectId(base);
        c.rank = recoverRank(base);
        c.header = marker.header;
        c.has_label = true;

        const auto score =
          parseAcceptedScore(marker.text);

        if (score.has_value()) {
          c.accepted = true;
          c.antipodal_score = *score;
        }
      }
    }

    std::vector<Candidate> out;

    for (const auto & kv : by_base)
    {
      const Candidate & c = kv.second;

      if (c.accepted &&
          c.has_approach &&
          c.has_contacts &&
          c.has_label &&
          c.antipodal_score >= min_antipodal_score_)
      {
        out.push_back(c);
      }
    }

    return out;
  }

  std::optional<Refined> refine(
    const Candidate & c) const
  {
    Refined r;
    r.source = c;

    const Eigen::Vector3d delta =
      c.observed_contact1 -
      c.observed_contact0;

    r.observed_width = delta.norm();

    if (r.observed_width < 1e-6) {
      return std::nullopt;
    }

    r.required_opening =
      r.observed_width +
      2.0 * finger_clearance_m_;

    if (r.required_opening >
        max_gripper_opening_m_)
    {
      return std::nullopt;
    }

    r.center =
      0.5 * (
        c.observed_contact0 +
        c.observed_contact1);

    r.closing =
      delta.normalized();

    Eigen::Vector3d old_approach =
      c.old_center -
      c.old_pregrasp;

    if (old_approach.norm() < 1e-8) {
      return std::nullopt;
    }

    old_approach.normalize();

    // Project the old approach onto the plane orthogonal
    // to the refined closing axis.
    Eigen::Vector3d projected =
      old_approach -
      old_approach.dot(r.closing) *
      r.closing;

    if (projected.norm() < 1e-8) {
      return std::nullopt;
    }

    r.approach =
      projected.normalized();

    r.binormal =
      r.approach.cross(r.closing);

    if (r.binormal.norm() < 1e-8) {
      return std::nullopt;
    }

    r.binormal.normalize();

    // Re-orthogonalize closing to guarantee an exact
    // right-handed gripper frame.
    r.closing =
      r.binormal.cross(r.approach).normalized();

    Eigen::Matrix3d Rg;
    Rg.col(0) = r.approach;
    Rg.col(1) = r.closing;
    Rg.col(2) = r.binormal;

    r.q =
      Eigen::Quaterniond(Rg).normalized();

    r.pregrasp =
      r.center -
      pregrasp_distance_m_ *
      r.approach;

    return r;
  }

  static visualization_msgs::msg::Marker makeApproach(
    const Refined & r,
    int id)
  {
    visualization_msgs::msg::Marker m;
    m.header = r.source.header;
    m.ns = "selected_grasp_approach";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::ARROW;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.points.push_back(
      eigenToPoint(r.pregrasp));

    m.points.push_back(
      eigenToPoint(r.center));

    m.scale.x = 0.009;
    m.scale.y = 0.018;
    m.scale.z = 0.026;

    m.color.r = 0.15f;
    m.color.g = 1.00f;
    m.color.b = 0.20f;
    m.color.a = 1.0f;

    return m;
  }

  static visualization_msgs::msg::Marker makeContactLine(
    const Refined & r,
    int id)
  {
    visualization_msgs::msg::Marker m;
    m.header = r.source.header;
    m.ns = "selected_grasp_contacts";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::LINE_LIST;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.points.push_back(
      eigenToPoint(
        r.source.observed_contact0));

    m.points.push_back(
      eigenToPoint(
        r.source.observed_contact1));

    m.scale.x = 0.009;

    m.color.r = 1.00f;
    m.color.g = 0.85f;
    m.color.b = 0.10f;
    m.color.a = 1.0f;

    return m;
  }

  static visualization_msgs::msg::Marker makeCenterSphere(
    const Refined & r,
    int id)
  {
    visualization_msgs::msg::Marker m;
    m.header = r.source.header;
    m.ns = "selected_grasp_center";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::SPHERE;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.pose.position =
      eigenToPoint(r.center);

    m.pose.orientation.w = 1.0;

    m.scale.x = 0.018;
    m.scale.y = 0.018;
    m.scale.z = 0.018;

    m.color.r = 1.00f;
    m.color.g = 1.00f;
    m.color.b = 1.00f;
    m.color.a = 1.0f;

    return m;
  }

  static visualization_msgs::msg::Marker makeLabel(
    const Refined & r,
    int id)
  {
    visualization_msgs::msg::Marker m;
    m.header = r.source.header;
    m.ns = "selected_grasp_labels";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.pose.position =
      eigenToPoint(
        r.pregrasp +
        Eigen::Vector3d(0.0, -0.025, 0.0));

    m.pose.orientation.w = 1.0;
    m.scale.z = 0.030;

    m.color.r = 1.0f;
    m.color.g = 1.0f;
    m.color.b = 1.0f;
    m.color.a = 1.0f;

    const int score_pct =
      static_cast<int>(
        std::lround(
          r.source.antipodal_score *
          100.0));

    const int width_mm =
      static_cast<int>(
        std::lround(
          r.observed_width *
          1000.0));

    const int opening_mm =
      static_cast<int>(
        std::lround(
          r.required_opening *
          1000.0));

    m.text =
      "SELECTED obj" +
      std::to_string(
        r.source.object_id) +
      " anti=" +
      std::to_string(score_pct) +
      "% width=" +
      std::to_string(width_mm) +
      "mm open=" +
      std::to_string(opening_mm) +
      "mm";

    return m;
  }

  void callback(
    const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
  {
    ++frame_count_;

    const auto candidates =
      parseCandidates(*msg);

    std::map<int, Refined> best_by_object;

    for (const auto & c : candidates)
    {
      const auto refined =
        refine(c);

      if (!refined.has_value()) {
        continue;
      }

      auto it =
        best_by_object.find(
          c.object_id);

      const bool better =
        it == best_by_object.end() ||
        refined->source.antipodal_score >
          it->second.source.antipodal_score ||
        (
          std::abs(
            refined->source.antipodal_score -
            it->second.source.antipodal_score) <
            1e-9 &&
          c.rank <
            it->second.source.rank
        );

      if (better) {
        best_by_object[
          c.object_id] = *refined;
      }
    }

    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker clear;
    clear.action =
      visualization_msgs::msg::Marker::DELETEALL;

    if (!best_by_object.empty()) {
      clear.header =
        best_by_object.begin()->second.source.header;
    }

    markers.markers.push_back(clear);

    geometry_msgs::msg::PoseArray grasps;
    geometry_msgs::msg::PoseArray pregrasps;

    if (!best_by_object.empty()) {
      grasps.header =
        best_by_object.begin()->second.source.header;
      pregrasps.header =
        best_by_object.begin()->second.source.header;
    }

    for (const auto & kv : best_by_object)
    {
      const int object_id =
        kv.first;

      const Refined & r =
        kv.second;

      const int base =
        object_id * 100;

      markers.markers.push_back(
        makeApproach(
          r,
          base + 0));

      markers.markers.push_back(
        makeContactLine(
          r,
          base + 1));

      markers.markers.push_back(
        makeCenterSphere(
          r,
          base + 2));

      markers.markers.push_back(
        makeLabel(
          r,
          base + 3));

      geometry_msgs::msg::Pose gp;
      gp.position =
        eigenToPoint(r.center);
      gp.orientation =
        quatToMsg(r.q);

      geometry_msgs::msg::Pose pp;
      pp.position =
        eigenToPoint(r.pregrasp);
      pp.orientation =
        quatToMsg(r.q);

      grasps.poses.push_back(gp);
      pregrasps.poses.push_back(pp);
    }

    marker_pub_->publish(markers);

    if (!best_by_object.empty()) {
      grasp_pose_pub_->publish(grasps);
      pregrasp_pose_pub_->publish(pregrasps);
    }

    if (log_every_n_ > 0 &&
        frame_count_ %
          static_cast<std::uint64_t>(
            log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "GRASP SELECT #%lu | accepted_input=%zu | selected_objects=%zu",
        static_cast<unsigned long>(
          frame_count_),
        candidates.size(),
        best_by_object.size());

      for (const auto & kv : best_by_object)
      {
        const auto & r =
          kv.second;

        RCLCPP_INFO(
          get_logger(),
          "  object_%d | source_rank=%d | anti=%.2f | "
          "observed_width=%.3f m | opening=%.3f m | "
          "center=[%.3f %.3f %.3f]",
          kv.first,
          r.source.rank,
          r.source.antipodal_score,
          r.observed_width,
          r.required_opening,
          r.center.x(),
          r.center.y(),
          r.center.z());
      }
    }
  }

  std::string input_topic_;
  std::string marker_topic_;
  std::string grasp_pose_topic_;
  std::string pregrasp_pose_topic_;

  double max_gripper_opening_m_;
  double finger_clearance_m_;
  double pregrasp_distance_m_;
  double min_antipodal_score_;
  int log_every_n_;

  std::uint64_t frame_count_{0};

  rclcpp::Subscription<
    visualization_msgs::msg::MarkerArray>::SharedPtr
    sub_;

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
    std::make_shared<
      GraspRefinementSelector>());
  rclcpp::shutdown();
  return 0;
}
