#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

class AntipodalGraspValidator : public rclcpp::Node
{
public:
  AntipodalGraspValidator()
  : Node("realsense_antipodal_grasp_validator")
  {
    cloud_topic_ = declare_parameter<std::string>(
      "cloud_topic", "/r10/cloud_without_support");

    candidates_topic_ = declare_parameter<std::string>(
      "candidates_topic", "/r10/grasp_filtered_markers");

    markers_topic_ = declare_parameter<std::string>(
      "markers_topic", "/r10/grasp_antipodal_markers");

    accepted_pose_topic_ = declare_parameter<std::string>(
      "accepted_pose_topic", "/r10/grasp_antipodal_poses");

    accepted_pregrasp_topic_ = declare_parameter<std::string>(
      "accepted_pregrasp_topic", "/r10/pregrasp_antipodal_poses");

    max_contact_distance_m_ = declare_parameter<double>(
      "max_contact_distance_m", 0.020);

    normal_radius_m_ = declare_parameter<double>(
      "normal_radius_m", 0.025);

    min_neighbors_ = declare_parameter<int>(
      "min_neighbors", 8);

    max_normal_alignment_deg_ = declare_parameter<double>(
      "max_normal_alignment_deg", 30.0);

    max_normal_opposition_deg_ = declare_parameter<double>(
      "max_normal_opposition_deg", 30.0);

    max_curvature_ = declare_parameter<double>(
      "max_curvature", 0.12);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 10);

    rclcpp::QoS cloud_qos(rclcpp::KeepLast(2));
    cloud_qos.reliable();
    cloud_qos.durability_volatile();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_,
      cloud_qos,
      std::bind(
        &AntipodalGraspValidator::cloudCallback,
        this,
        std::placeholders::_1));

    candidate_sub_ =
      create_subscription<visualization_msgs::msg::MarkerArray>(
        candidates_topic_,
        10,
        std::bind(
          &AntipodalGraspValidator::candidateCallback,
          this,
          std::placeholders::_1));

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
        markers_topic_, 10);

    accepted_pose_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>(
        accepted_pose_topic_, 10);

    accepted_pregrasp_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>(
        accepted_pregrasp_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "R10-5C Antipodal Grasp Validator actif\n"
      "cloud                    : %s\n"
      "candidates               : %s\n"
      "markers                  : %s\n"
      "max contact distance     : %.3f m\n"
      "normal radius            : %.3f m\n"
      "min neighbors            : %d\n"
      "normal alignment tol     : %.1f deg\n"
      "normal opposition tol    : %.1f deg\n"
      "max curvature            : %.3f",
      cloud_topic_.c_str(),
      candidates_topic_.c_str(),
      markers_topic_.c_str(),
      max_contact_distance_m_,
      normal_radius_m_,
      min_neighbors_,
      max_normal_alignment_deg_,
      max_normal_opposition_deg_,
      max_curvature_);
  }

private:
  enum class Status
  {
    ACCEPTED,
    REJECTED,
    UNVERIFIED
  };

  struct PartialCandidate
  {
    int base_id{-1};
    int object_id{-1};
    int rank{-1};
    std_msgs::msg::Header header;

    bool has_approach{false};
    bool has_contacts{false};

    Eigen::Vector3d pregrasp{0,0,0};
    Eigen::Vector3d center{0,0,0};

    Eigen::Vector3d p0{0,0,0};
    Eigen::Vector3d p1{0,0,0};
  };

  struct NormalEstimate
  {
    bool observed{false};
    Eigen::Vector3d nearest{0,0,0};
    Eigen::Vector3d normal{0,0,1};
    double nearest_distance{0.0};
    double curvature{1.0};
    int neighbors{0};
  };

  struct Result
  {
    Status status{Status::UNVERIFIED};
    PartialCandidate c;

    NormalEstimate n0;
    NormalEstimate n1;

    Eigen::Vector3d closing{0,1,0};
    Eigen::Vector3d approach{1,0,0};
    Eigen::Quaterniond q{1,0,0,0};

    double align0_deg{180.0};
    double align1_deg{180.0};
    double opposition_deg{180.0};
    double score{0.0};

    std::string reason;
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

  static double angleDeg(
    const Eigen::Vector3d & a,
    const Eigen::Vector3d & b)
  {
    const double dot = std::clamp(
      a.normalized().dot(b.normalized()),
      -1.0,
      1.0);

    constexpr double kPi =
      3.14159265358979323846;

    return std::acos(dot) * 180.0 / kPi;
  }

  static double clamp01(double x)
  {
    return std::clamp(x, 0.0, 1.0);
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    auto cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
      return;
    }

    latest_cloud_ = cloud;
    latest_cloud_header_ = msg->header;

    latest_tree_ =
      std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();

    latest_tree_->setInputCloud(latest_cloud_);
  }

  std::vector<PartialCandidate> parseCandidates(
    const visualization_msgs::msg::MarkerArray & array) const
  {
    std::map<int, PartialCandidate> by_base;

    for (const auto & marker : array.markers)
    {
      if (marker.action !=
          visualization_msgs::msg::Marker::ADD)
      {
        continue;
      }

      if (marker.ns == "filtered_grasp_approach" &&
          marker.type ==
            visualization_msgs::msg::Marker::ARROW &&
          marker.points.size() >= 2)
      {
        const int base = marker.id;
        auto & c = by_base[base];

        c.base_id = base;
        c.object_id = base / 1000;
        c.rank = (base % 1000) / 10 + 1;
        c.header = marker.header;
        c.pregrasp = pointToEigen(marker.points[0]);
        c.center = pointToEigen(marker.points[1]);
        c.has_approach = true;
      }
      else if (
        marker.ns == "filtered_grasp_contacts" &&
        marker.type ==
          visualization_msgs::msg::Marker::LINE_LIST &&
        marker.points.size() >= 2)
      {
        const int base = marker.id - 1;
        auto & c = by_base[base];

        c.base_id = base;
        c.object_id = base / 1000;
        c.rank = (base % 1000) / 10 + 1;
        c.header = marker.header;
        c.p0 = pointToEigen(marker.points[0]);
        c.p1 = pointToEigen(marker.points[1]);
        c.has_contacts = true;
      }
    }

    std::vector<PartialCandidate> out;
    out.reserve(by_base.size());

    for (const auto & kv : by_base) {
      if (kv.second.has_approach &&
          kv.second.has_contacts)
      {
        out.push_back(kv.second);
      }
    }

    return out;
  }

  NormalEstimate estimateNormal(
    const Eigen::Vector3d & query,
    const Eigen::Vector3d & outward) const
  {
    NormalEstimate result;

    if (!latest_cloud_ ||
        !latest_tree_ ||
        latest_cloud_->empty())
    {
      return result;
    }

    pcl::PointXYZ q;
    q.x = static_cast<float>(query.x());
    q.y = static_cast<float>(query.y());
    q.z = static_cast<float>(query.z());

    std::vector<int> nearest_idx(1);
    std::vector<float> nearest_dist_sq(1);

    const int found =
      latest_tree_->nearestKSearch(
        q,
        1,
        nearest_idx,
        nearest_dist_sq);

    if (found < 1) {
      return result;
    }

    result.nearest_distance =
      std::sqrt(
        static_cast<double>(nearest_dist_sq[0]));

    if (result.nearest_distance >
        max_contact_distance_m_)
    {
      return result;
    }

    const auto & nearest_p =
      (*latest_cloud_)[
        static_cast<std::size_t>(nearest_idx[0])];

    result.nearest =
      Eigen::Vector3d(
        nearest_p.x,
        nearest_p.y,
        nearest_p.z);

    pcl::PointXYZ radius_center;
    radius_center.x = nearest_p.x;
    radius_center.y = nearest_p.y;
    radius_center.z = nearest_p.z;

    std::vector<int> indices;
    std::vector<float> distances_sq;

    latest_tree_->radiusSearch(
      radius_center,
      normal_radius_m_,
      indices,
      distances_sq);

    result.neighbors =
      static_cast<int>(indices.size());

    if (result.neighbors < min_neighbors_) {
      return result;
    }

    Eigen::Vector3d centroid =
      Eigen::Vector3d::Zero();

    for (const int idx : indices)
    {
      const auto & p =
        (*latest_cloud_)[static_cast<std::size_t>(idx)];

      centroid +=
        Eigen::Vector3d(p.x, p.y, p.z);
    }

    centroid /=
      static_cast<double>(indices.size());

    Eigen::Matrix3d covariance =
      Eigen::Matrix3d::Zero();

    for (const int idx : indices)
    {
      const auto & p =
        (*latest_cloud_)[static_cast<std::size_t>(idx)];

      const Eigen::Vector3d x(
        p.x, p.y, p.z);

      const Eigen::Vector3d d =
        x - centroid;

      covariance += d * d.transpose();
    }

    covariance /=
      static_cast<double>(indices.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
      solver(covariance);

    if (solver.info() != Eigen::Success) {
      return result;
    }

    // Ascending eigenvalues. Smallest-variance axis is
    // the normal of a locally planar surface.
    const Eigen::Vector3d evals =
      solver.eigenvalues();

    Eigen::Vector3d normal =
      solver.eigenvectors().col(0).normalized();

    // PCA normal has arbitrary sign. Orient it outward
    // using the ideal OBB contact direction.
    if (normal.dot(outward) < 0.0) {
      normal *= -1.0;
    }

    const double sum =
      std::max(
        1e-12,
        evals.x() + evals.y() + evals.z());

    result.curvature =
      evals.x() / sum;

    result.normal = normal;
    result.observed = true;

    return result;
  }

  Result evaluate(
    const PartialCandidate & c) const
  {
    Result r;
    r.c = c;

    const Eigen::Vector3d closing_raw =
      c.p1 - c.p0;

    const Eigen::Vector3d approach_raw =
      c.center - c.pregrasp;

    if (closing_raw.norm() < 1e-8 ||
        approach_raw.norm() < 1e-8)
    {
      r.reason = "degenerate geometry";
      r.status = Status::UNVERIFIED;
      return r;
    }

    r.closing = closing_raw.normalized();
    r.approach = approach_raw.normalized();

    const Eigen::Vector3d midpoint =
      0.5 * (c.p0 + c.p1);

    const Eigen::Vector3d outward0 =
      (c.p0 - midpoint).normalized();

    const Eigen::Vector3d outward1 =
      (c.p1 - midpoint).normalized();

    r.n0 = estimateNormal(c.p0, outward0);
    r.n1 = estimateNormal(c.p1, outward1);

    if (!r.n0.observed || !r.n1.observed)
    {
      r.status = Status::UNVERIFIED;
      r.reason = "one/both contacts not observed";
      return r;
    }

    r.align0_deg =
      angleDeg(r.n0.normal, outward0);

    r.align1_deg =
      angleDeg(r.n1.normal, outward1);

    // Antipodal: n0 should be approximately -n1.
    r.opposition_deg =
      angleDeg(r.n0.normal, -r.n1.normal);

    if (r.n0.curvature > max_curvature_ ||
        r.n1.curvature > max_curvature_)
    {
      r.status = Status::REJECTED;
      r.reason = "surface too non-planar";
      return r;
    }

    if (r.align0_deg >
          max_normal_alignment_deg_ ||
        r.align1_deg >
          max_normal_alignment_deg_)
    {
      r.status = Status::REJECTED;
      r.reason = "normals not aligned with jaw axis";
      return r;
    }

    if (r.opposition_deg >
        max_normal_opposition_deg_)
    {
      r.status = Status::REJECTED;
      r.reason = "normals not antipodal";
      return r;
    }

    // Build a right-handed grasp frame from the observed
    // approach and jaw-closing direction.
    Eigen::Vector3d binormal =
      r.approach.cross(r.closing);

    if (binormal.norm() < 1e-8)
    {
      r.status = Status::REJECTED;
      r.reason = "approach parallel to closing";
      return r;
    }

    binormal.normalize();

    r.closing =
      binormal.cross(r.approach).normalized();

    Eigen::Matrix3d Rg;
    Rg.col(0) = r.approach;
    Rg.col(1) = r.closing;
    Rg.col(2) = binormal;

    r.q =
      Eigen::Quaterniond(Rg).normalized();

    const double align_score =
      0.5 * (
        std::cos(
          r.align0_deg *
          3.14159265358979323846 / 180.0) +
        std::cos(
          r.align1_deg *
          3.14159265358979323846 / 180.0));

    const double opposition_score =
      std::cos(
        r.opposition_deg *
        3.14159265358979323846 / 180.0);

    const double curvature_score =
      1.0 -
      clamp01(
        std::max(
          r.n0.curvature,
          r.n1.curvature) /
        max_curvature_);

    r.score =
      clamp01(
        0.45 * align_score +
        0.35 * opposition_score +
        0.20 * curvature_score);

    r.status = Status::ACCEPTED;
    r.reason = "antipodal";
    return r;
  }

  visualization_msgs::msg::Marker makeApproach(
    const Result & r,
    int id,
    float cr,
    float cg,
    float cb) const
  {
    visualization_msgs::msg::Marker m;
    m.header = r.c.header;
    m.ns = "antipodal_approach";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.points.push_back(
      eigenToPoint(r.c.pregrasp));

    m.points.push_back(
      eigenToPoint(r.c.center));

    m.scale.x = 0.007;
    m.scale.y = 0.014;
    m.scale.z = 0.020;

    m.color.r = cr;
    m.color.g = cg;
    m.color.b = cb;
    m.color.a = 1.0f;

    return m;
  }

  visualization_msgs::msg::Marker makeNormalArrow(
    const Result & r,
    const NormalEstimate & n,
    int id,
    float cr,
    float cg,
    float cb) const
  {
    visualization_msgs::msg::Marker m;
    m.header = r.c.header;
    m.ns = "contact_normals";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.points.push_back(
      eigenToPoint(n.nearest));

    m.points.push_back(
      eigenToPoint(
        n.nearest +
        0.045 * n.normal));

    m.scale.x = 0.004;
    m.scale.y = 0.009;
    m.scale.z = 0.014;

    m.color.r = cr;
    m.color.g = cg;
    m.color.b = cb;
    m.color.a = 1.0f;

    return m;
  }

  visualization_msgs::msg::Marker makeContactLine(
    const Result & r,
    int id,
    float cr,
    float cg,
    float cb) const
  {
    visualization_msgs::msg::Marker m;
    m.header = r.c.header;
    m.ns = "antipodal_contacts";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::LINE_LIST;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.points.push_back(
      eigenToPoint(r.n0.nearest));

    m.points.push_back(
      eigenToPoint(r.n1.nearest));

    m.scale.x = 0.006;

    m.color.r = cr;
    m.color.g = cg;
    m.color.b = cb;
    m.color.a = 1.0f;

    return m;
  }

  visualization_msgs::msg::Marker makeLabel(
    const Result & r,
    int id) const
  {
    visualization_msgs::msg::Marker m;
    m.header = r.c.header;
    m.ns = "antipodal_labels";
    m.id = id;
    m.type =
      visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action =
      visualization_msgs::msg::Marker::ADD;

    m.pose.position =
      eigenToPoint(
        r.c.pregrasp +
        Eigen::Vector3d(0.0, -0.025, 0.0));

    m.pose.orientation.w = 1.0;
    m.scale.z = 0.025;

    if (r.status == Status::ACCEPTED) {
      m.color.r = 0.20f;
      m.color.g = 1.00f;
      m.color.b = 0.25f;
    } else if (r.status == Status::REJECTED) {
      m.color.r = 1.00f;
      m.color.g = 0.20f;
      m.color.b = 0.20f;
    } else {
      m.color.r = 1.00f;
      m.color.g = 0.65f;
      m.color.b = 0.10f;
    }

    m.color.a = 1.0f;

    if (r.status == Status::ACCEPTED)
    {
      const int score_pct =
        static_cast<int>(
          std::lround(r.score * 100.0));

      const int a0 =
        static_cast<int>(
          std::lround(r.align0_deg));

      const int a1 =
        static_cast<int>(
          std::lround(r.align1_deg));

      const int opp =
        static_cast<int>(
          std::lround(r.opposition_deg));

      m.text =
        "obj" +
        std::to_string(r.c.object_id) +
        " #" +
        std::to_string(r.c.rank) +
        " ANTIPODAL " +
        std::to_string(score_pct) +
        "% a=" +
        std::to_string(a0) +
        "/" +
        std::to_string(a1) +
        " opp=" +
        std::to_string(opp);
    }
    else if (r.status == Status::REJECTED)
    {
      m.text =
        "obj" +
        std::to_string(r.c.object_id) +
        " #" +
        std::to_string(r.c.rank) +
        " REJECT: " +
        r.reason;
    }
    else
    {
      m.text =
        "obj" +
        std::to_string(r.c.object_id) +
        " #" +
        std::to_string(r.c.rank) +
        " UNVERIFIED";
    }

    return m;
  }

  void candidateCallback(
    const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
  {
    ++frame_count_;

    if (!latest_cloud_ ||
        !latest_tree_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        3000,
        "En attente de /r10/cloud_without_support.");
      return;
    }

    const auto candidates =
      parseCandidates(*msg);

    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker clear;
    clear.action =
      visualization_msgs::msg::Marker::DELETEALL;

    if (!candidates.empty()) {
      clear.header =
        candidates.front().header;
    }

    markers.markers.push_back(clear);

    geometry_msgs::msg::PoseArray accepted_poses;
    geometry_msgs::msg::PoseArray accepted_pregrasp;

    if (!candidates.empty()) {
      accepted_poses.header =
        candidates.front().header;
      accepted_pregrasp.header =
        candidates.front().header;
    }

    int accepted = 0;
    int rejected = 0;
    int unverified = 0;

    for (const auto & c : candidates)
    {
      const Result r =
        evaluate(c);

      const int base =
        c.base_id * 10;

      float cr = 1.0f;
      float cg = 0.65f;
      float cb = 0.10f;

      if (r.status == Status::ACCEPTED) {
        cr = 0.20f;
        cg = 1.00f;
        cb = 0.25f;
        ++accepted;
      }
      else if (r.status == Status::REJECTED) {
        cr = 1.00f;
        cg = 0.20f;
        cb = 0.20f;
        ++rejected;
      }
      else {
        ++unverified;
      }

      markers.markers.push_back(
        makeApproach(
          r,
          base + 0,
          cr, cg, cb));

      markers.markers.push_back(
        makeLabel(
          r,
          base + 1));

      if (r.n0.observed) {
        markers.markers.push_back(
          makeNormalArrow(
            r,
            r.n0,
            base + 2,
            0.10f, 0.80f, 1.00f));
      }

      if (r.n1.observed) {
        markers.markers.push_back(
          makeNormalArrow(
            r,
            r.n1,
            base + 3,
            0.10f, 0.80f, 1.00f));
      }

      if (r.n0.observed &&
          r.n1.observed)
      {
        markers.markers.push_back(
          makeContactLine(
            r,
            base + 4,
            cr, cg, cb));
      }

      if (r.status == Status::ACCEPTED)
      {
        geometry_msgs::msg::Pose gp;
        gp.position =
          eigenToPoint(c.center);
        gp.orientation =
          quatToMsg(r.q);

        geometry_msgs::msg::Pose pp;
        pp.position =
          eigenToPoint(c.pregrasp);
        pp.orientation =
          quatToMsg(r.q);

        accepted_poses.poses.push_back(gp);
        accepted_pregrasp.poses.push_back(pp);
      }
    }

    marker_pub_->publish(markers);

    if (!candidates.empty()) {
      accepted_pose_pub_->publish(
        accepted_poses);
      accepted_pregrasp_pub_->publish(
        accepted_pregrasp);
    }

    if (log_every_n_ > 0 &&
        frame_count_ %
          static_cast<std::uint64_t>(
            log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "ANTIPODAL #%lu | candidates=%zu | "
        "accepted=%d | rejected=%d | unverified=%d",
        static_cast<unsigned long>(frame_count_),
        candidates.size(),
        accepted,
        rejected,
        unverified);
    }
  }

  std::string cloud_topic_;
  std::string candidates_topic_;
  std::string markers_topic_;
  std::string accepted_pose_topic_;
  std::string accepted_pregrasp_topic_;

  double max_contact_distance_m_;
  double normal_radius_m_;
  int min_neighbors_;
  double max_normal_alignment_deg_;
  double max_normal_opposition_deg_;
  double max_curvature_;
  int log_every_n_;

  std::uint64_t frame_count_{0};

  pcl::PointCloud<pcl::PointXYZ>::Ptr
    latest_cloud_;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr
    latest_tree_;

  std_msgs::msg::Header
    latest_cloud_header_;

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    cloud_sub_;

  rclcpp::Subscription<
    visualization_msgs::msg::MarkerArray>::SharedPtr
    candidate_sub_;

  rclcpp::Publisher<
    visualization_msgs::msg::MarkerArray>::SharedPtr
    marker_pub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseArray>::SharedPtr
    accepted_pose_pub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseArray>::SharedPtr
    accepted_pregrasp_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<AntipodalGraspValidator>());
  rclcpp::shutdown();
  return 0;
}
