#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
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
#include <stdexcept>
#include <string>
#include <vector>

class EuclideanClusterNode : public rclcpp::Node
{
public:
  EuclideanClusterNode()
  : Node("realsense_euclidean_cluster")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/r10/cloud_without_support");

    largest_topic_ = declare_parameter<std::string>(
      "largest_topic", "/r10/cloud_largest_cluster");

    centroid_topic_ = declare_parameter<std::string>(
      "centroid_topic", "/r10/largest_cluster_centroid");

    colored_topic_ = declare_parameter<std::string>(
      "colored_topic", "/r10/cloud_clusters_colored");

    markers_topic_ = declare_parameter<std::string>(
      "markers_topic", "/r10/cluster_markers");

    geometry_markers_topic_ = declare_parameter<std::string>(
      "geometry_markers_topic", "/r10/cluster_geometry_markers");

    obb_poses_topic_ = declare_parameter<std::string>(
      "obb_poses_topic", "/r10/cluster_obb_poses");

    cluster_tolerance_ = declare_parameter<double>(
      "cluster_tolerance", 0.020);

    min_cluster_size_ = declare_parameter<int>(
      "min_cluster_size", 100);

    max_cluster_size_ = declare_parameter<int>(
      "max_cluster_size", 30000);

    process_every_n_ = declare_parameter<int>(
      "process_every_n", 1);

    log_every_n_ = declare_parameter<int>(
      "log_every_n", 10);

    marker_scale_ = declare_parameter<double>(
      "marker_scale", 0.025);

    text_height_ = declare_parameter<double>(
      "text_height", 0.035);

    geometry_max_clusters_ = declare_parameter<int>(
      "geometry_max_clusters", 10);

    if (cluster_tolerance_ <= 0.0 ||
        min_cluster_size_ <= 0 ||
        max_cluster_size_ < min_cluster_size_ ||
        process_every_n_ <= 0 ||
        marker_scale_ <= 0.0 ||
        text_height_ <= 0.0 ||
        geometry_max_clusters_ <= 0)
    {
      throw std::runtime_error("Parametres de clustering/geometrie invalides");
    }

    rclcpp::QoS cloud_qos(rclcpp::KeepLast(2));
    cloud_qos.reliable();
    cloud_qos.durability_volatile();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      cloud_qos,
      std::bind(&EuclideanClusterNode::callback, this, std::placeholders::_1));

    largest_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      largest_topic_, cloud_qos);

    colored_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      colored_topic_, cloud_qos);

    centroid_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      centroid_topic_, 10);

    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      markers_topic_, 10);

    geometry_markers_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
        geometry_markers_topic_, 10);

    obb_poses_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      obb_poses_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "R10 Euclidean Clustering + Geometry actif\n"
      "input              : %s\n"
      "clusters colores   : %s\n"
      "cluster markers    : %s\n"
      "geometry markers   : %s\n"
      "OBB poses          : %s\n"
      "cluster tolerance  : %.3f m\n"
      "cluster size       : [%d, %d]\n"
      "geometry top N     : %d",
      input_topic_.c_str(),
      colored_topic_.c_str(),
      markers_topic_.c_str(),
      geometry_markers_topic_.c_str(),
      obb_poses_topic_.c_str(),
      cluster_tolerance_,
      min_cluster_size_,
      max_cluster_size_,
      geometry_max_clusters_);
  }

private:
  struct Geometry
  {
    Eigen::Vector3f centroid;
    Eigen::Vector3f aabb_center;
    Eigen::Vector3f aabb_dims;

    Eigen::Vector3f obb_center;
    Eigen::Vector3f obb_dims;
    Eigen::Matrix3f obb_rotation;
    Eigen::Quaternionf obb_quaternion;

    Eigen::Vector3f eigenvalues;
  };

  static std::array<std::uint8_t, 3> colorForIndex(std::size_t i)
  {
    static constexpr std::array<std::array<std::uint8_t, 3>, 12> palette {{
      {{255,  80,  80}},
      {{ 80, 255,  80}},
      {{ 80, 140, 255}},
      {{255, 220,  80}},
      {{255,  80, 220}},
      {{ 80, 255, 230}},
      {{255, 150,  60}},
      {{160,  90, 255}},
      {{120, 255, 120}},
      {{255, 120, 170}},
      {{120, 210, 255}},
      {{220, 220, 220}},
    }};
    return palette[i % palette.size()];
  }

  static Geometry computeGeometry(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
  {
    Geometry g;

    Eigen::Vector4f centroid4;
    pcl::compute3DCentroid(*cloud, centroid4);
    g.centroid = centroid4.head<3>();

    pcl::PointXYZ min_pt;
    pcl::PointXYZ max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    const Eigen::Vector3f aabb_min(min_pt.x, min_pt.y, min_pt.z);
    const Eigen::Vector3f aabb_max(max_pt.x, max_pt.y, max_pt.z);

    g.aabb_center = 0.5f * (aabb_min + aabb_max);
    g.aabb_dims = aabb_max - aabb_min;

    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid4, covariance);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Echec decomposition PCA");
    }

    // SelfAdjointEigenSolver returns eigenvalues/eigenvectors in ascending order.
    const Eigen::Vector3f evals_asc = solver.eigenvalues();
    const Eigen::Matrix3f evecs_asc = solver.eigenvectors();

    Eigen::Matrix3f rotation;
    rotation.col(0) = evecs_asc.col(2).normalized();  // principal axis
    rotation.col(1) = evecs_asc.col(1).normalized();
    rotation.col(2) = rotation.col(0).cross(rotation.col(1)).normalized();

    // Re-orthogonalize axis 1 to guarantee a right-handed orthonormal basis.
    rotation.col(1) = rotation.col(2).cross(rotation.col(0)).normalized();

    g.eigenvalues = Eigen::Vector3f(
      evals_asc[2], evals_asc[1], evals_asc[0]);

    Eigen::Vector3f local_min(
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity());

    Eigen::Vector3f local_max(
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity());

    for (const auto & p : *cloud)
    {
      const Eigen::Vector3f world(p.x, p.y, p.z);
      const Eigen::Vector3f local =
        rotation.transpose() * (world - g.centroid);

      local_min = local_min.cwiseMin(local);
      local_max = local_max.cwiseMax(local);
    }

    const Eigen::Vector3f local_center =
      0.5f * (local_min + local_max);

    g.obb_dims = local_max - local_min;
    g.obb_center = g.centroid + rotation * local_center;
    g.obb_rotation = rotation;
    g.obb_quaternion = Eigen::Quaternionf(rotation);
    g.obb_quaternion.normalize();

    return g;
  }

  static geometry_msgs::msg::Quaternion toQuaternionMsg(
    const Eigen::Quaternionf & q)
  {
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
  }

  static visualization_msgs::msg::Marker makeCubeMarker(
    const std_msgs::msg::Header & header,
    const std::string & ns,
    int id,
    const Eigen::Vector3f & center,
    const Eigen::Vector3f & dims,
    const Eigen::Quaternionf & q,
    const std::array<std::uint8_t, 3> & rgb,
    float alpha)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = center.x();
    marker.pose.position.y = center.y();
    marker.pose.position.z = center.z();
    marker.pose.orientation = toQuaternionMsg(q);

    marker.scale.x = std::max(0.001f, dims.x());
    marker.scale.y = std::max(0.001f, dims.y());
    marker.scale.z = std::max(0.001f, dims.z());

    marker.color.r = static_cast<float>(rgb[0]) / 255.0f;
    marker.color.g = static_cast<float>(rgb[1]) / 255.0f;
    marker.color.b = static_cast<float>(rgb[2]) / 255.0f;
    marker.color.a = alpha;

    return marker;
  }

  static visualization_msgs::msg::Marker makeAxisArrow(
    const std_msgs::msg::Header & header,
    int id,
    const Eigen::Vector3f & origin,
    const Eigen::Vector3f & axis,
    float length,
    float r, float g, float b)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "pca_axes";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p0;
    p0.x = origin.x();
    p0.y = origin.y();
    p0.z = origin.z();

    geometry_msgs::msg::Point p1;
    const Eigen::Vector3f end = origin + axis * length;
    p1.x = end.x();
    p1.y = end.y();
    p1.z = end.z();

    marker.points.push_back(p0);
    marker.points.push_back(p1);

    marker.scale.x = 0.005;
    marker.scale.y = 0.010;
    marker.scale.z = 0.015;

    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0f;

    return marker;
  }

  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    ++input_count_;

    if ((input_count_ - 1) %
        static_cast<std::uint64_t>(process_every_n_) != 0)
    {
      return;
    }

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
      return;
    }

    auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    ++processed_count_;

    if (cluster_indices.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "Aucun cluster valide | input=%zu",
        cloud->size());
      return;
    }

    std::sort(
      cluster_indices.begin(),
      cluster_indices.end(),
      [](const pcl::PointIndices & a, const pcl::PointIndices & b) {
        return a.indices.size() > b.indices.size();
      });

    auto colored_cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    std::size_t total_cluster_points = 0;
    for (const auto & cluster : cluster_indices) {
      total_cluster_points += cluster.indices.size();
    }
    colored_cloud->reserve(total_cluster_points);

    visualization_msgs::msg::MarkerArray cluster_markers;
    visualization_msgs::msg::MarkerArray geometry_markers;
    geometry_msgs::msg::PoseArray obb_poses;
    obb_poses.header = msg->header;

    visualization_msgs::msg::Marker clear_clusters;
    clear_clusters.header = msg->header;
    clear_clusters.action = visualization_msgs::msg::Marker::DELETEALL;
    cluster_markers.markers.push_back(clear_clusters);

    visualization_msgs::msg::Marker clear_geometry;
    clear_geometry.header = msg->header;
    clear_geometry.action = visualization_msgs::msg::Marker::DELETEALL;
    geometry_markers.markers.push_back(clear_geometry);

    const std::size_t geometry_count = std::min(
      cluster_indices.size(),
      static_cast<std::size_t>(geometry_max_clusters_));

    std::vector<Geometry> geometries;
    geometries.reserve(geometry_count);

    pcl::PointCloud<pcl::PointXYZ>::Ptr largest_cloud;

    for (std::size_t cluster_id = 0;
         cluster_id < cluster_indices.size();
         ++cluster_id)
    {
      const auto & indices = cluster_indices[cluster_id];
      const auto rgb = colorForIndex(cluster_id);

      auto cluster_cloud =
        std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      cluster_cloud->reserve(indices.indices.size());

      for (const int index : indices.indices) {
        if (index < 0 ||
            static_cast<std::size_t>(index) >= cloud->size())
        {
          continue;
        }

        const auto & p = (*cloud)[static_cast<std::size_t>(index)];
        cluster_cloud->push_back(p);

        pcl::PointXYZRGB q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.r = rgb[0];
        q.g = rgb[1];
        q.b = rgb[2];
        colored_cloud->push_back(q);
      }

      cluster_cloud->width =
        static_cast<std::uint32_t>(cluster_cloud->size());
      cluster_cloud->height = 1;
      cluster_cloud->is_dense = true;

      if (cluster_id == 0) {
        largest_cloud = cluster_cloud;
      }

      if (cluster_cloud->empty()) {
        continue;
      }

      Eigen::Vector4f centroid4;
      pcl::compute3DCentroid(*cluster_cloud, centroid4);

      visualization_msgs::msg::Marker sphere;
      sphere.header = msg->header;
      sphere.ns = "cluster_centroids";
      sphere.id = static_cast<int>(cluster_id * 2);
      sphere.type = visualization_msgs::msg::Marker::SPHERE;
      sphere.action = visualization_msgs::msg::Marker::ADD;
      sphere.pose.position.x = centroid4[0];
      sphere.pose.position.y = centroid4[1];
      sphere.pose.position.z = centroid4[2];
      sphere.pose.orientation.w = 1.0;
      sphere.scale.x = marker_scale_;
      sphere.scale.y = marker_scale_;
      sphere.scale.z = marker_scale_;
      sphere.color.r = static_cast<float>(rgb[0]) / 255.0f;
      sphere.color.g = static_cast<float>(rgb[1]) / 255.0f;
      sphere.color.b = static_cast<float>(rgb[2]) / 255.0f;
      sphere.color.a = 1.0f;
      cluster_markers.markers.push_back(sphere);

      visualization_msgs::msg::Marker text;
      text.header = msg->header;
      text.ns = "cluster_labels";
      text.id = static_cast<int>(cluster_id * 2 + 1);
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = centroid4[0];
      text.pose.position.y = centroid4[1] - 0.04;
      text.pose.position.z = centroid4[2];
      text.pose.orientation.w = 1.0;
      text.scale.z = text_height_;
      text.color.r = static_cast<float>(rgb[0]) / 255.0f;
      text.color.g = static_cast<float>(rgb[1]) / 255.0f;
      text.color.b = static_cast<float>(rgb[2]) / 255.0f;
      text.color.a = 1.0f;
      text.text =
        "cluster_" + std::to_string(cluster_id) +
        " (" + std::to_string(cluster_cloud->size()) + " pts)";
      cluster_markers.markers.push_back(text);

      if (cluster_id < geometry_count)
      {
        const Geometry g = computeGeometry(cluster_cloud);
        geometries.push_back(g);

        // AABB is axis-aligned with the camera optical frame.
        const Eigen::Quaternionf identity =
          Eigen::Quaternionf::Identity();

        geometry_markers.markers.push_back(
          makeCubeMarker(
            msg->header,
            "aabb",
            static_cast<int>(cluster_id),
            g.aabb_center,
            g.aabb_dims,
            identity,
            rgb,
            0.12f));

        // OBB is oriented by PCA.
        geometry_markers.markers.push_back(
          makeCubeMarker(
            msg->header,
            "obb",
            static_cast<int>(cluster_id),
            g.obb_center,
            g.obb_dims,
            g.obb_quaternion,
            rgb,
            0.30f));

        const float max_dim =
          std::max({g.obb_dims.x(), g.obb_dims.y(), g.obb_dims.z()});
        const float axis_len =
          std::clamp(0.6f * max_dim, 0.04f, 0.20f);

        geometry_markers.markers.push_back(
          makeAxisArrow(
            msg->header,
            static_cast<int>(cluster_id * 3),
            g.obb_center,
            g.obb_rotation.col(0),
            axis_len,
            1.0f, 0.15f, 0.15f));

        geometry_markers.markers.push_back(
          makeAxisArrow(
            msg->header,
            static_cast<int>(cluster_id * 3 + 1),
            g.obb_center,
            g.obb_rotation.col(1),
            axis_len,
            0.15f, 1.0f, 0.15f));

        geometry_markers.markers.push_back(
          makeAxisArrow(
            msg->header,
            static_cast<int>(cluster_id * 3 + 2),
            g.obb_center,
            g.obb_rotation.col(2),
            axis_len,
            0.15f, 0.35f, 1.0f));

        visualization_msgs::msg::Marker geom_text;
        geom_text.header = msg->header;
        geom_text.ns = "geometry_labels";
        geom_text.id = static_cast<int>(cluster_id);
        geom_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        geom_text.action = visualization_msgs::msg::Marker::ADD;
        geom_text.pose.position.x = g.obb_center.x();
        geom_text.pose.position.y = g.obb_center.y() - 0.075;
        geom_text.pose.position.z = g.obb_center.z();
        geom_text.pose.orientation.w = 1.0;
        geom_text.scale.z = 0.028;
        geom_text.color.r = 1.0f;
        geom_text.color.g = 1.0f;
        geom_text.color.b = 1.0f;
        geom_text.color.a = 1.0f;

        const int d0_cm =
          static_cast<int>(std::lround(g.obb_dims.x() * 100.0f));
        const int d1_cm =
          static_cast<int>(std::lround(g.obb_dims.y() * 100.0f));
        const int d2_cm =
          static_cast<int>(std::lround(g.obb_dims.z() * 100.0f));

        geom_text.text =
          "c" + std::to_string(cluster_id) +
          " OBB=" +
          std::to_string(d0_cm) + "x" +
          std::to_string(d1_cm) + "x" +
          std::to_string(d2_cm) + " cm";

        geometry_markers.markers.push_back(geom_text);

        geometry_msgs::msg::Pose pose;
        pose.position.x = g.obb_center.x();
        pose.position.y = g.obb_center.y();
        pose.position.z = g.obb_center.z();
        pose.orientation = toQuaternionMsg(g.obb_quaternion);
        obb_poses.poses.push_back(pose);
      }
    }

    colored_cloud->width =
      static_cast<std::uint32_t>(colored_cloud->size());
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;

    sensor_msgs::msg::PointCloud2 colored_msg;
    pcl::toROSMsg(*colored_cloud, colored_msg);
    colored_msg.header = msg->header;
    colored_pub_->publish(colored_msg);

    markers_pub_->publish(cluster_markers);
    geometry_markers_pub_->publish(geometry_markers);
    obb_poses_pub_->publish(obb_poses);

    if (largest_cloud && !largest_cloud->empty())
    {
      sensor_msgs::msg::PointCloud2 largest_msg;
      pcl::toROSMsg(*largest_cloud, largest_msg);
      largest_msg.header = msg->header;
      largest_pub_->publish(largest_msg);

      Eigen::Vector4f largest_centroid;
      pcl::compute3DCentroid(*largest_cloud, largest_centroid);

      geometry_msgs::msg::PointStamped centroid_msg;
      centroid_msg.header = msg->header;
      centroid_msg.point.x = largest_centroid[0];
      centroid_msg.point.y = largest_centroid[1];
      centroid_msg.point.z = largest_centroid[2];
      centroid_pub_->publish(centroid_msg);
    }

    if (log_every_n_ > 0 &&
        processed_count_ %
          static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      RCLCPP_INFO(
        get_logger(),
        "GEOMETRY #%lu | input=%zu | clusters=%zu | geometry=%zu",
        static_cast<unsigned long>(processed_count_),
        cloud->size(),
        cluster_indices.size(),
        geometries.size());

      for (std::size_t i = 0; i < geometries.size(); ++i)
      {
        const auto & g = geometries[i];
        RCLCPP_INFO(
          get_logger(),
          "  cluster_%zu | pts=%zu | C=[%.3f %.3f %.3f] m | "
          "AABB=[%.3f %.3f %.3f] m | "
          "OBB=[%.3f %.3f %.3f] m | "
          "eig=[%.6f %.6f %.6f]",
          i,
          cluster_indices[i].indices.size(),
          g.centroid.x(),
          g.centroid.y(),
          g.centroid.z(),
          g.aabb_dims.x(),
          g.aabb_dims.y(),
          g.aabb_dims.z(),
          g.obb_dims.x(),
          g.obb_dims.y(),
          g.obb_dims.z(),
          g.eigenvalues.x(),
          g.eigenvalues.y(),
          g.eigenvalues.z());
      }
    }
  }

  std::string input_topic_;
  std::string largest_topic_;
  std::string centroid_topic_;
  std::string colored_topic_;
  std::string markers_topic_;
  std::string geometry_markers_topic_;
  std::string obb_poses_topic_;

  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  int process_every_n_;
  int log_every_n_;
  double marker_scale_;
  double text_height_;
  int geometry_max_clusters_;

  std::uint64_t input_count_{0};
  std::uint64_t processed_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr largest_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr centroid_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    geometry_markers_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obb_poses_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<EuclideanClusterNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("realsense_euclidean_cluster"),
      "Erreur fatale: %s",
      e.what());
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}
