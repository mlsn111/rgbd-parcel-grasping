#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class TemporalObbTracker : public rclcpp::Node
{
public:
  TemporalObbTracker()
  : Node("realsense_temporal_obb_tracker")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/cluster_geometry_markers");

    markers_topic_ = declare_parameter<std::string>(
      "markers_topic", "/r10/tracked_object_markers");

    poses_topic_ = declare_parameter<std::string>(
      "poses_topic", "/r10/tracked_obb_poses");

    max_position_distance_ = declare_parameter<double>(
      "max_position_distance", 0.15);

    max_relative_dimension_error_ = declare_parameter<double>(
      "max_relative_dimension_error", 0.65);

    position_alpha_ = declare_parameter<double>(
      "position_alpha", 0.35);

    dimension_alpha_ = declare_parameter<double>(
      "dimension_alpha", 0.25);

    orientation_alpha_ = declare_parameter<double>(
      "orientation_alpha", 0.25);

    max_missed_frames_ = declare_parameter<int>(
      "max_missed_frames", 12);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 10);

    sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      input_topic_,
      10,
      std::bind(&TemporalObbTracker::callback, this, std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      markers_topic_, 10);

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      poses_topic_, 10);

    buildCuboidSymmetries();

    RCLCPP_INFO(
      get_logger(),
      "R10-4H Temporal OBB Tracker actif | input=%s | output=%s",
      input_topic_.c_str(),
      markers_topic_.c_str());
  }

private:
  struct Detection
  {
    int ephemeral_cluster_id{-1};
    std_msgs::msg::Header header;
    Eigen::Vector3d center{0.0, 0.0, 0.0};
    Eigen::Vector3d dims{0.0, 0.0, 0.0};
    Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};
  };

  struct Track
  {
    int id{-1};
    std_msgs::msg::Header header;
    Eigen::Vector3d center{0.0, 0.0, 0.0};
    Eigen::Vector3d dims{0.0, 0.0, 0.0};
    Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};
    int age{1};
    int missed{0};
    bool visible{true};
  };

  struct MatchCandidate
  {
    std::size_t track_index{0};
    std::size_t detection_index{0};
    double cost{0.0};
  };

  static Eigen::Vector3d sortedDims(const Eigen::Vector3d & d)
  {
    std::array<double, 3> a {d.x(), d.y(), d.z()};
    std::sort(a.begin(), a.end());
    return Eigen::Vector3d(a[0], a[1], a[2]);
  }

  static double relativeDimensionError(
    const Eigen::Vector3d & a,
    const Eigen::Vector3d & b)
  {
    const auto sa = sortedDims(a);
    const auto sb = sortedDims(b);
    return (sa - sb).norm() / std::max(1e-6, sa.norm());
  }

  static double rotationAngle(
    const Eigen::Matrix3d & a,
    const Eigen::Matrix3d & b)
  {
    const Eigen::Matrix3d delta = a.transpose() * b;
    const double c = std::clamp((delta.trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(c);
  }

  void buildCuboidSymmetries()
  {
    std::array<int, 3> perm {0, 1, 2};
    do {
      for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
          for (int sz : {-1, 1}) {
            Eigen::Matrix3d p = Eigen::Matrix3d::Zero();
            p(perm[0], 0) = sx;
            p(perm[1], 1) = sy;
            p(perm[2], 2) = sz;
            if (p.determinant() > 0.5) {
              cuboid_symmetries_.push_back(p);
            }
          }
        }
      }
    } while (std::next_permutation(perm.begin(), perm.end()));
  }

  std::pair<Eigen::Quaterniond, Eigen::Vector3d> alignCuboidRepresentation(
    const Track & track,
    const Detection & det) const
  {
    const auto r_prev = track.q.toRotationMatrix();
    const auto r_det = det.q.toRotationMatrix();

    double best_score = std::numeric_limits<double>::infinity();
    Eigen::Quaterniond best_q = det.q;
    Eigen::Vector3d best_dims = det.dims;

    for (const auto & symmetry : cuboid_symmetries_)
    {
      const Eigen::Matrix3d r_candidate = r_det * symmetry;
      const Eigen::Vector3d candidate_dims =
        symmetry.cwiseAbs().transpose() * det.dims;

      const double angle = rotationAngle(r_prev, r_candidate);
      const double dim_error =
        (candidate_dims - track.dims).norm() /
        std::max(1e-6, track.dims.norm());

      const double score = angle + 0.75 * dim_error;

      if (score < best_score) {
        best_score = score;
        best_q = Eigen::Quaterniond(r_candidate);
        best_q.normalize();
        best_dims = candidate_dims;
      }
    }

    if (track.q.dot(best_q) < 0.0) {
      best_q.coeffs() *= -1.0;
    }

    return {best_q, best_dims};
  }

  std::vector<Detection> parseDetections(
    const visualization_msgs::msg::MarkerArray & array)
  {
    std::vector<Detection> detections;

    for (const auto & marker : array.markers)
    {
      if (marker.action != visualization_msgs::msg::Marker::ADD ||
          marker.ns != "obb" ||
          marker.type != visualization_msgs::msg::Marker::CUBE)
      {
        continue;
      }

      Detection d;
      d.ephemeral_cluster_id = marker.id;
      d.header = marker.header;
      d.center = Eigen::Vector3d(
        marker.pose.position.x,
        marker.pose.position.y,
        marker.pose.position.z);
      d.dims = Eigen::Vector3d(
        marker.scale.x,
        marker.scale.y,
        marker.scale.z);
      d.q = Eigen::Quaterniond(
        marker.pose.orientation.w,
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z);
      if (d.q.norm() < 1e-8) {
        d.q = Eigen::Quaterniond::Identity();
      } else {
        d.q.normalize();
      }
      detections.push_back(d);
    }
    return detections;
  }

  double associationCost(const Track & track, const Detection & det) const
  {
    const double dp = (track.center - det.center).norm();
    const double dd = relativeDimensionError(track.dims, det.dims);

    if (dp > max_position_distance_ || dd > max_relative_dimension_error_) {
      return std::numeric_limits<double>::infinity();
    }

    return dp / max_position_distance_
      + 0.5 * dd / max_relative_dimension_error_;
  }

  void updateTrack(Track & track, const Detection & det)
  {
    const auto [aligned_q, aligned_dims] =
      alignCuboidRepresentation(track, det);

    track.center =
      (1.0 - position_alpha_) * track.center +
      position_alpha_ * det.center;

    track.dims =
      (1.0 - dimension_alpha_) * track.dims +
      dimension_alpha_ * aligned_dims;

    track.q = track.q.slerp(orientation_alpha_, aligned_q);
    track.q.normalize();

    track.header = det.header;
    track.age++;
    track.missed = 0;
    track.visible = true;
  }

  void createTrack(const Detection & det)
  {
    Track t;
    t.id = next_track_id_++;
    t.header = det.header;
    t.center = det.center;
    t.dims = det.dims;
    t.q = det.q;
    tracks_.push_back(t);

    RCLCPP_INFO(
      get_logger(),
      "NOUVEL OBJET object_%d depuis cluster_%d",
      t.id, det.ephemeral_cluster_id);
  }

  static geometry_msgs::msg::Quaternion toMsg(
    const Eigen::Quaterniond & q)
  {
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
  }

  void publishTracks()
  {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    if (!tracks_.empty()) clear.header = tracks_.front().header;
    array.markers.push_back(clear);

    geometry_msgs::msg::PoseArray poses;
    bool header_set = false;

    for (const auto & t : tracks_)
    {
      if (!t.visible) continue;

      if (!header_set) {
        poses.header = t.header;
        header_set = true;
      }

      visualization_msgs::msg::Marker cube;
      cube.header = t.header;
      cube.ns = "tracked_obb";
      cube.id = t.id;
      cube.type = visualization_msgs::msg::Marker::CUBE;
      cube.action = visualization_msgs::msg::Marker::ADD;
      cube.pose.position.x = t.center.x();
      cube.pose.position.y = t.center.y();
      cube.pose.position.z = t.center.z();
      cube.pose.orientation = toMsg(t.q);
      cube.scale.x = t.dims.x();
      cube.scale.y = t.dims.y();
      cube.scale.z = t.dims.z();
      cube.color.r = 0.2f;
      cube.color.g = 0.8f;
      cube.color.b = 1.0f;
      cube.color.a = 0.30f;
      array.markers.push_back(cube);

      visualization_msgs::msg::Marker text;
      text.header = t.header;
      text.ns = "tracked_labels";
      text.id = t.id;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = t.center.x();
      text.pose.position.y = t.center.y() - 0.06;
      text.pose.position.z = t.center.z();
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.03;
      text.color.r = 1.0f;
      text.color.g = 1.0f;
      text.color.b = 1.0f;
      text.color.a = 1.0f;
      text.text = "object_" + std::to_string(t.id);
      array.markers.push_back(text);

      geometry_msgs::msg::Pose pose;
      pose.position = cube.pose.position;
      pose.orientation = cube.pose.orientation;
      poses.poses.push_back(pose);
    }

    marker_pub_->publish(array);
    if (header_set) pose_pub_->publish(poses);
  }

  void callback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
  {
    const auto detections = parseDetections(*msg);
    frame_count_++;

    for (auto & t : tracks_) t.visible = false;

    std::vector<MatchCandidate> candidates;
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      for (std::size_t di = 0; di < detections.size(); ++di) {
        const double c = associationCost(tracks_[ti], detections[di]);
        if (std::isfinite(c)) candidates.push_back({ti, di, c});
      }
    }

    std::sort(
      candidates.begin(), candidates.end(),
      [](const MatchCandidate & a, const MatchCandidate & b) {
        return a.cost < b.cost;
      });

    std::vector<bool> used_t(tracks_.size(), false);
    std::vector<bool> used_d(detections.size(), false);

    for (const auto & c : candidates) {
      if (used_t[c.track_index] || used_d[c.detection_index]) continue;
      updateTrack(tracks_[c.track_index], detections[c.detection_index]);
      used_t[c.track_index] = true;
      used_d[c.detection_index] = true;
    }

    for (std::size_t i = 0; i < tracks_.size(); ++i) {
      if (!used_t[i]) {
        tracks_[i].missed++;
        tracks_[i].visible = false;
      }
    }

    tracks_.erase(
      std::remove_if(
        tracks_.begin(), tracks_.end(),
        [this](const Track & t) {
          return t.missed > max_missed_frames_;
        }),
      tracks_.end());

    for (std::size_t di = 0; di < detections.size(); ++di) {
      if (!used_d[di]) createTrack(detections[di]);
    }

    publishTracks();

    if (log_every_n_ > 0 &&
        frame_count_ % static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "TRACKING #%lu | detections=%zu | tracks=%zu",
        static_cast<unsigned long>(frame_count_),
        detections.size(),
        tracks_.size());
    }
  }

  std::string input_topic_;
  std::string markers_topic_;
  std::string poses_topic_;

  double max_position_distance_;
  double max_relative_dimension_error_;
  double position_alpha_;
  double dimension_alpha_;
  double orientation_alpha_;
  int max_missed_frames_;
  int log_every_n_;

  int next_track_id_{0};
  std::uint64_t frame_count_{0};

  std::vector<Eigen::Matrix3d> cuboid_symmetries_;
  std::vector<Track> tracks_;

  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TemporalObbTracker>());
  rclcpp::shutdown();
  return 0;
}
