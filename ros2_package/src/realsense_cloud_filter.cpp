#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

class CloudFilterNode : public rclcpp::Node
{
public:
  CloudFilterNode()
  : Node("realsense_cloud_filter")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/camera/camera/depth/color/points");

    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/r10/cloud_filtered");

    x_min_ = declare_parameter<double>("x_min", -0.45);
    x_max_ = declare_parameter<double>("x_max",  0.45);
    y_min_ = declare_parameter<double>("y_min", -0.35);
    y_max_ = declare_parameter<double>("y_max",  0.35);
    z_min_ = declare_parameter<double>("z_min",  0.15);
    z_max_ = declare_parameter<double>("z_max",  1.50);

    log_every_n_ = declare_parameter<int>("log_every_n", 30);

    if (!(x_min_ < x_max_) || !(y_min_ < y_max_) || !(z_min_ < z_max_)) {
      throw std::runtime_error(
        "Bornes CropBox invalides: min doit etre strictement inferieur a max");
    }

    rclcpp::QoS input_qos(rclcpp::KeepLast(2));
    input_qos.reliable();
    input_qos.durability_volatile();

    rclcpp::QoS output_qos(rclcpp::KeepLast(2));
    output_qos.reliable();
    output_qos.durability_volatile();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      input_qos,
      std::bind(&CloudFilterNode::callback, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_,
      output_qos);

    RCLCPP_INFO(
      get_logger(),
      "R10 Cloud CropBox actif\n"
      "input  : %s\n"
      "output : %s\n"
      "X      : [%.3f, %.3f] m\n"
      "Y      : [%.3f, %.3f] m\n"
      "Z      : [%.3f, %.3f] m\n"
      "repere : conserve le frame_id du PointCloud2 d'entree",
      input_topic_.c_str(),
      output_topic_.c_str(),
      x_min_, x_max_,
      y_min_, y_max_,
      z_min_, z_max_);
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    auto cloud_in = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cloud_out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::fromROSMsg(*msg, *cloud_in);

    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(cloud_in);
    crop.setMin(Eigen::Vector4f(
      static_cast<float>(x_min_),
      static_cast<float>(y_min_),
      static_cast<float>(z_min_),
      1.0f));
    crop.setMax(Eigen::Vector4f(
      static_cast<float>(x_max_),
      static_cast<float>(y_max_),
      static_cast<float>(z_max_),
      1.0f));
    crop.filter(*cloud_out);

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
        "cloud #%lu | frame=%s | entree=%zu | crop=%zu | conserves=%.2f %% | "
        "X=[%.2f,%.2f] Y=[%.2f,%.2f] Z=[%.2f,%.2f] m",
        static_cast<unsigned long>(frame_count_),
        msg->header.frame_id.c_str(),
        n_in,
        n_out,
        kept_percent,
        x_min_, x_max_,
        y_min_, y_max_,
        z_min_, z_max_);
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  int log_every_n_;
  std::uint64_t frame_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CloudFilterNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("realsense_cloud_filter"),
      "Erreur fatale: %s",
      e.what());
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
