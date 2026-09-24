#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class RunningStats
{
public:
  void add(double x)
  {
    ++n_;
    const double delta = x - mean_;
    mean_ += delta / static_cast<double>(n_);
    const double delta2 = x - mean_;
    m2_ += delta * delta2;
  }

  std::uint64_t count() const { return n_; }
  double mean() const { return mean_; }

  double stddev() const
  {
    return n_ > 1 ? std::sqrt(m2_ / static_cast<double>(n_ - 1)) : 0.0;
  }

private:
  std::uint64_t n_{0};
  double mean_{0.0};
  double m2_{0.0};
};

class DepthQualityNode : public rclcpp::Node
{
public:
  DepthQualityNode()
  : Node("realsense_depth_quality")
  {
    topic_ = declare_parameter<std::string>(
      "depth_topic",
      "/camera/camera/depth/image_rect_raw");

    depth_scale_ = declare_parameter<double>("depth_scale", 0.001);
    roi_width_ = declare_parameter<int>("roi_width", 80);
    roi_height_ = declare_parameter<int>("roi_height", 80);
    target_frames_ = declare_parameter<int>("frames", 300);
    label_ = declare_parameter<std::string>("label", "unnamed_test");

    rclcpp::QoS qos(rclcpp::KeepLast(5));
    qos.best_effort();
    qos.durability_volatile();

    histogram_.fill(0);

    sub_ = create_subscription<sensor_msgs::msg::Image>(
      topic_,
      qos,
      std::bind(&DepthQualityNode::callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "R10 Depth Quality actif\n"
      "label       : %s\n"
      "topic       : %s\n"
      "depth scale : %.9f m/unite\n"
      "ROI         : %dx%d\n"
      "frames      : %d",
      label_.c_str(),
      topic_.c_str(),
      depth_scale_,
      roi_width_,
      roi_height_,
      target_frames_);
  }

private:
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

  static double median(std::vector<double> values)
  {
    if (values.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const std::size_t n = values.size();
    const std::size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double upper = values[mid];

    if (n % 2 == 1) {
      return upper;
    }

    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return 0.5 * (values[mid - 1] + upper);
  }

  std::uint16_t percentile_raw(double fraction) const
  {
    if (valid_pixels_ == 0) {
      return 0;
    }

    const std::uint64_t target = static_cast<std::uint64_t>(
      std::ceil(fraction * static_cast<double>(valid_pixels_)));

    std::uint64_t cumulative = 0;
    for (std::size_t i = 1; i < histogram_.size(); ++i) {
      cumulative += histogram_[i];
      if (cumulative >= target) {
        return static_cast<std::uint16_t>(i);
      }
    }
    return 65535;
  }

  void callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (finished_) {
      return;
    }

    if (msg->encoding != "16UC1") {
      RCLCPP_ERROR(get_logger(), "Encoding inattendu : %s", msg->encoding.c_str());
      finished_ = true;
      rclcpp::shutdown();
      return;
    }

    if (roi_width_ <= 0 || roi_height_ <= 0 ||
        roi_width_ > static_cast<int>(msg->width) ||
        roi_height_ > static_cast<int>(msg->height))
    {
      RCLCPP_ERROR(
        get_logger(),
        "ROI invalide pour image %ux%u",
        msg->width,
        msg->height);
      finished_ = true;
      rclcpp::shutdown();
      return;
    }

    const int x0 = (static_cast<int>(msg->width) - roi_width_) / 2;
    const int y0 = (static_cast<int>(msg->height) - roi_height_) / 2;

    std::vector<double> frame_values;
    frame_values.reserve(static_cast<std::size_t>(roi_width_ * roi_height_));

    RunningStats spatial_stats;

    for (int y = y0; y < y0 + roi_height_; ++y) {
      for (int x = x0; x < x0 + roi_width_; ++x) {
        ++total_pixels_;

        const std::uint16_t raw = read_pixel(*msg, x, y);

        if (raw == 0) {
          ++invalid_pixels_;
          continue;
        }

        ++valid_pixels_;
        ++histogram_[raw];

        const double z_m = static_cast<double>(raw) * depth_scale_;
        global_stats_.add(z_m);
        spatial_stats.add(z_m);
        frame_values.push_back(z_m);
      }
    }

    if (!frame_values.empty()) {
      const double frame_median = median(frame_values);
      frame_median_stats_.add(frame_median);

      if (spatial_stats.count() > 1) {
        spatial_sigma_stats_.add(spatial_stats.stddev());
      }
    }

    const int cx = static_cast<int>(msg->width) / 2;
    const int cy = static_cast<int>(msg->height) / 2;

    const std::uint16_t center_raw = read_pixel(*msg, cx, cy);
    if (center_raw != 0) {
      center_stats_.add(static_cast<double>(center_raw) * depth_scale_);
    }

    ++frames_received_;

    if (frames_received_ >= static_cast<std::uint64_t>(target_frames_)) {
      report();
      finished_ = true;
      rclcpp::shutdown();
    }
  }

  void report()
  {
    const double invalid_percent =
      total_pixels_ > 0
        ? 100.0 * static_cast<double>(invalid_pixels_) /
            static_cast<double>(total_pixels_)
        : 0.0;

    const auto p05_raw = percentile_raw(0.05);
    const auto p50_raw = percentile_raw(0.50);
    const auto p95_raw = percentile_raw(0.95);

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "========== R10 DEPTH QUALITY ==========\n"
      "label                     : %s\n"
      "frames                    : %lu\n"
      "ROI                       : %dx%d\n"
      "depth scale               : %.9f m/unite\n"
      "\n"
      "--- Validite ---\n"
      "pixels total              : %lu\n"
      "pixels valides            : %lu\n"
      "pixels invalides          : %lu\n"
      "pixels invalides          : %.4f %%\n"
      "\n"
      "--- Distance ROI ---\n"
      "moyenne globale           : %.3f mm\n"
      "P05                       : %.3f mm\n"
      "mediane P50               : %.3f mm\n"
      "P95                       : %.3f mm\n"
      "sigma globale             : %.3f mm\n"
      "\n"
      "--- Bruit spatial ---\n"
      "sigma spatial moyen/frame : %.3f mm\n"
      "\n"
      "--- Stabilite temporelle ---\n"
      "moyenne mediane/frame     : %.3f mm\n"
      "sigma mediane/frame       : %.3f mm\n"
      "\n"
      "--- Pixel central ---\n"
      "frames centrales valides  : %lu\n"
      "distance moyenne          : %.3f mm\n"
      "sigma temporel            : %.3f mm\n"
      "========================================",
      label_.c_str(),
      static_cast<unsigned long>(frames_received_),
      roi_width_,
      roi_height_,
      depth_scale_,
      static_cast<unsigned long>(total_pixels_),
      static_cast<unsigned long>(valid_pixels_),
      static_cast<unsigned long>(invalid_pixels_),
      invalid_percent,
      global_stats_.mean() * 1000.0,
      static_cast<double>(p05_raw) * depth_scale_ * 1000.0,
      static_cast<double>(p50_raw) * depth_scale_ * 1000.0,
      static_cast<double>(p95_raw) * depth_scale_ * 1000.0,
      global_stats_.stddev() * 1000.0,
      spatial_sigma_stats_.mean() * 1000.0,
      frame_median_stats_.mean() * 1000.0,
      frame_median_stats_.stddev() * 1000.0,
      static_cast<unsigned long>(center_stats_.count()),
      center_stats_.mean() * 1000.0,
      center_stats_.stddev() * 1000.0);
  }

  std::string topic_;
  std::string label_;
  double depth_scale_;
  int roi_width_;
  int roi_height_;
  int target_frames_;

  std::uint64_t frames_received_{0};
  std::uint64_t total_pixels_{0};
  std::uint64_t valid_pixels_{0};
  std::uint64_t invalid_pixels_{0};

  bool finished_{false};

  RunningStats global_stats_;
  RunningStats frame_median_stats_;
  RunningStats spatial_sigma_stats_;
  RunningStats center_stats_;

  std::array<std::uint64_t, 65536> histogram_{};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthQualityNode>());

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
