#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

class VoxelFilterNode : public rclcpp::Node
{
public:
  VoxelFilterNode()
  : Node("realsense_voxel_filter")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic",
      "/r10/cloud_filtered");

    output_topic_ = declare_parameter<std::string>(
      "output_topic",
      "/r10/cloud_voxel");

    leaf_size_ = declare_parameter<double>("leaf_size", 0.005);
    log_every_n_ = declare_parameter<int>("log_every_n", 30);

    if (leaf_size_ <= 0.0) {
      throw std::runtime_error("leaf_size doit etre > 0");
    }

    rclcpp::QoS qos(rclcpp::KeepLast(2));
    qos.reliable();
    qos.durability_volatile();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      qos,
      std::bind(&VoxelFilterNode::callback, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_,
      qos);

    RCLCPP_INFO(
      get_logger(),
      "R10 Voxel Filter actif\n"
      "input       : %s\n"
      "output      : %s\n"
      "leaf size   : %.4f m (%.1f mm)\n"
      "QoS         : RELIABLE / VOLATILE / KEEP_LAST(2)",
      input_topic_.c_str(),
      output_topic_.c_str(),
      leaf_size_,
      leaf_size_ * 1000.0);
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    auto cloud_in = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cloud_out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::fromROSMsg(*msg, *cloud_in);

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud_in);

    const float leaf = static_cast<float>(leaf_size_);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*cloud_out);

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*cloud_out, out_msg);
    out_msg.header = msg->header;

    pub_->publish(out_msg);

    ++frame_count_;

    if (log_every_n_ > 0 &&
        frame_count_ % static_cast<std::uint64_t>(log_every_n_) == 0)
    {
      const std::size_t n_in = cloud_in->size();
      const std::size_t n_out = cloud_out->size();

      const double kept_percent =
        n_in > 0
          ? 100.0 * static_cast<double>(n_out) /
              static_cast<double>(n_in)
          : 0.0;

      RCLCPP_INFO(
        get_logger(),
        "cloud #%lu | frame=%s | entree=%zu pts | voxel=%zu pts | "
        "conserves=%.2f %% | leaf=%.1f mm",
        static_cast<unsigned long>(frame_count_),
        msg->header.frame_id.c_str(),
        n_in,
        n_out,
        kept_percent,
        leaf_size_ * 1000.0);
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  double leaf_size_;
  int log_every_n_;
  std::uint64_t frame_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<VoxelFilterNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("realsense_voxel_filter"),
      "Erreur fatale: %s",
      e.what());
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}
