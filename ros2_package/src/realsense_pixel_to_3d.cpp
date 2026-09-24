#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class PixelTo3DNode : public rclcpp::Node
{
public:
  PixelTo3DNode()
  : Node("realsense_pixel_to_3d")
  {
    depth_topic_ = declare_parameter<std::string>(
      "depth_topic",
      "/camera/camera/aligned_depth_to_color/image_raw");

    camera_info_topic_ = declare_parameter<std::string>(
      "camera_info_topic",
      "/camera/camera/color/camera_info");

    u_ = declare_parameter<int>("u", 652);
    v_ = declare_parameter<int>("v", 376);
    roi_radius_ = declare_parameter<int>("roi_radius", 2);
    depth_scale_ = declare_parameter<double>("depth_scale", 0.001);
    log_every_n_ = declare_parameter<int>("log_every_n", 30);

    rclcpp::QoS sensor_qos(rclcpp::KeepLast(5));
    sensor_qos.best_effort();
    sensor_qos.durability_volatile();

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      rclcpp::QoS(1).reliable().durability_volatile(),
      std::bind(&PixelTo3DNode::camera_info_callback, this, std::placeholders::_1));

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_,
      sensor_qos,
      std::bind(&PixelTo3DNode::depth_callback, this, std::placeholders::_1));

    point_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      "/r10/target_point_camera",
      10);

    RCLCPP_INFO(
      get_logger(),
      "R10 Pixel->3D actif\n"
      "depth topic      : %s\n"
      "camera_info      : %s\n"
      "pixel cible      : (%d, %d)\n"
      "ROI              : %dx%d\n"
      "depth scale      : %.9f m/unite\n"
      "sortie Point     : /r10/target_point_camera",
      depth_topic_.c_str(),
      camera_info_topic_.c_str(),
      u_, v_,
      2 * roi_radius_ + 1,
      2 * roi_radius_ + 1,
      depth_scale_);
  }

private:
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    camera_frame_ = msg->header.frame_id;
    camera_width_ = static_cast<int>(msg->width);
    camera_height_ = static_cast<int>(msg->height);

    if (!have_intrinsics_) {
      RCLCPP_INFO(
        get_logger(),
        "Intrinseques recus: fx=%.6f fy=%.6f cx=%.6f cy=%.6f | %dx%d | frame=%s",
        fx_, fy_, cx_, cy_,
        camera_width_, camera_height_,
        camera_frame_.c_str());
    }

    have_intrinsics_ = true;
  }

  std::uint16_t read_pixel(
    const sensor_msgs::msg::Image & msg,
    int x,
    int y) const
  {
    const std::size_t offset =
      static_cast<std::size_t>(y) * msg.step +
      static_cast<std::size_t>(x) * 2;

    const auto b0 = static_cast<std::uint16_t>(msg.data[offset]);
    const auto b1 = static_cast<std::uint16_t>(msg.data[offset + 1]);

    if (msg.is_bigendian) {
      return static_cast<std::uint16_t>((b0 << 8) | b1);
    }

    return static_cast<std::uint16_t>(b0 | (b1 << 8));
  }

  static std::uint16_t median_raw(std::vector<std::uint16_t> values)
  {
    const std::size_t n = values.size();
    const std::size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const std::uint16_t upper = values[mid];

    if (n % 2 == 1) {
      return upper;
    }

    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    const std::uint16_t lower = values[mid - 1];

    return static_cast<std::uint16_t>(
      (static_cast<std::uint32_t>(lower) +
       static_cast<std::uint32_t>(upper)) / 2U);
  }

  void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (!have_intrinsics_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "En attente de camera_info...");
      return;
    }

    if (msg->encoding != "16UC1") {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Encoding depth inattendu: %s",
        msg->encoding.c_str());
      return;
    }

    const int width = static_cast<int>(msg->width);
    const int height = static_cast<int>(msg->height);

    if (u_ < 0 || u_ >= width || v_ < 0 || v_ >= height) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Pixel cible hors image: (%d,%d), image=%dx%d",
        u_, v_, width, height);
      return;
    }

    const int x0 = std::max(0, u_ - roi_radius_);
    const int x1 = std::min(width - 1, u_ + roi_radius_);
    const int y0 = std::max(0, v_ - roi_radius_);
    const int y1 = std::min(height - 1, v_ + roi_radius_);

    std::vector<std::uint16_t> valid_depths;
    valid_depths.reserve(
      static_cast<std::size_t>((x1 - x0 + 1) * (y1 - y0 + 1)));

    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const std::uint16_t raw = read_pixel(*msg, x, y);
        if (raw != 0) {
          valid_depths.push_back(raw);
        }
      }
    }

    if (valid_depths.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Aucune profondeur valide autour de (%d,%d)",
        u_, v_);
      return;
    }

    const std::uint16_t raw_median = median_raw(valid_depths);
    const double z = static_cast<double>(raw_median) * depth_scale_;

    const double x = (static_cast<double>(u_) - cx_) * z / fx_;
    const double y = (static_cast<double>(v_) - cy_) * z / fy_;

    geometry_msgs::msg::PointStamped out;
    out.header = msg->header;
    if (!camera_frame_.empty()) {
      out.header.frame_id = camera_frame_;
    }
    out.point.x = x;
    out.point.y = y;
    out.point.z = z;

    point_pub_->publish(out);

    ++frame_count_;
    if (log_every_n_ > 0 && frame_count_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
      RCLCPP_INFO(
        get_logger(),
        "pixel=(%d,%d) | ROI valides=%zu | depth median=%u -> Z=%.4f m | "
        "Point camera: X=%+.4f Y=%+.4f Z=%.4f m | frame=%s",
        u_, v_,
        valid_depths.size(),
        raw_median,
        z,
        x, y, z,
        out.header.frame_id.c_str());
    }
  }

  std::string depth_topic_;
  std::string camera_info_topic_;

  int u_;
  int v_;
  int roi_radius_;
  int log_every_n_;
  double depth_scale_;

  bool have_intrinsics_{false};
  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};
  int camera_width_{0};
  int camera_height_{0};
  std::string camera_frame_;

  std::uint64_t frame_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr point_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PixelTo3DNode>());
  rclcpp::shutdown();
  return 0;
}
