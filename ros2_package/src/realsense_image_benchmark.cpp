#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <string>

class StreamStats
{
public:
  explicit StreamStats(std::string name, double expected_fps)
  : name_(std::move(name)), expected_fps_(expected_fps)
  {}

  void add(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::chrono::steady_clock::time_point & recv_now)
  {
    const double stamp =
      static_cast<double>(msg->header.stamp.sec) +
      static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    if (count_ == 0) {
      first_recv_ = recv_now;
      last_recv_ = recv_now;
      first_stamp_ = stamp;
      last_stamp_ = stamp;

      min_stamp_dt_ = 1e9;
      max_stamp_dt_ = 0.0;
    } else {
      const double stamp_dt = stamp - last_stamp_;

      if (stamp_dt > 0.0) {
        sum_stamp_dt_ += stamp_dt;
        sum_stamp_dt_sq_ += stamp_dt * stamp_dt;

        min_stamp_dt_ = std::min(min_stamp_dt_, stamp_dt);
        max_stamp_dt_ = std::max(max_stamp_dt_, stamp_dt);

        const double nominal_dt = 1.0 / expected_fps_;

        if (stamp_dt > 1.5 * nominal_dt) {
          const long periods =
            std::max<long>(1, std::lround(stamp_dt / nominal_dt));

          if (periods > 1) {
            estimated_missed_frames_ +=
              static_cast<std::uint64_t>(periods - 1);
          }
        }
      }

      last_recv_ = recv_now;
      last_stamp_ = stamp;
    }

    total_bytes_ += msg->data.size();
    ++count_;
  }

  void report(rclcpp::Logger logger)
  {
    if (count_ < 2) {
      RCLCPP_INFO(
        logger,
        "%s : pas encore assez de frames",
        name_.c_str());
      reset();
      return;
    }

    const double recv_duration =
      std::chrono::duration<double>(last_recv_ - first_recv_).count();

    const double stamp_duration =
      last_stamp_ - first_stamp_;

    const double recv_fps =
      recv_duration > 0.0 ?
      static_cast<double>(count_ - 1) / recv_duration :
      0.0;

    const double stamp_fps =
      stamp_duration > 0.0 ?
      static_cast<double>(count_ - 1) / stamp_duration :
      0.0;

    const std::uint64_t intervals = count_ - 1;

    const double mean_dt =
      intervals > 0 ?
      sum_stamp_dt_ / static_cast<double>(intervals) :
      0.0;

    double variance = 0.0;

    if (intervals > 0) {
      variance =
        (sum_stamp_dt_sq_ / static_cast<double>(intervals)) -
        (mean_dt * mean_dt);

      variance = std::max(0.0, variance);
    }

    const double jitter_ms = std::sqrt(variance) * 1000.0;

    const double mb_received =
      static_cast<double>(total_bytes_) / 1e6;

    const double mb_per_s =
      recv_duration > 0.0 ?
      mb_received / recv_duration :
      0.0;

    RCLCPP_INFO(
      logger,
      "\n"
      "===== %s =====\n"
      "frames recues           : %lu\n"
      "FPS reception           : %.3f\n"
      "FPS timestamps          : %.3f\n"
      "intervalle moyen        : %.3f ms\n"
      "jitter timestamps       : %.3f ms\n"
      "intervalle min          : %.3f ms\n"
      "intervalle max          : %.3f ms\n"
      "frames manquees estimees: %lu\n"
      "debit recu              : %.2f MB/s",
      name_.c_str(),
      static_cast<unsigned long>(count_),
      recv_fps,
      stamp_fps,
      mean_dt * 1000.0,
      jitter_ms,
      min_stamp_dt_ * 1000.0,
      max_stamp_dt_ * 1000.0,
      static_cast<unsigned long>(estimated_missed_frames_),
      mb_per_s);

    reset();
  }

private:
  void reset()
  {
    count_ = 0;
    total_bytes_ = 0;
    estimated_missed_frames_ = 0;

    first_stamp_ = 0.0;
    last_stamp_ = 0.0;

    sum_stamp_dt_ = 0.0;
    sum_stamp_dt_sq_ = 0.0;

    min_stamp_dt_ = 1e9;
    max_stamp_dt_ = 0.0;
  }

  std::string name_;
  double expected_fps_;

  std::uint64_t count_{0};
  std::uint64_t total_bytes_{0};
  std::uint64_t estimated_missed_frames_{0};

  double first_stamp_{0.0};
  double last_stamp_{0.0};

  double sum_stamp_dt_{0.0};
  double sum_stamp_dt_sq_{0.0};

  double min_stamp_dt_{1e9};
  double max_stamp_dt_{0.0};

  std::chrono::steady_clock::time_point first_recv_;
  std::chrono::steady_clock::time_point last_recv_;
};


class RealSenseImageBenchmark : public rclcpp::Node
{
public:
  RealSenseImageBenchmark()
  : Node("realsense_image_benchmark"),
    color_stats_("COLOR", 30.0),
    depth_stats_("DEPTH", 60.0)
  {
    const std::string color_topic =
      declare_parameter<std::string>(
      "color_topic",
      "/camera/camera/color/image_raw");

    const std::string depth_topic =
      declare_parameter<std::string>(
      "depth_topic",
      "/camera/camera/depth/image_rect_raw");

    const double report_period =
      declare_parameter<double>("report_period", 5.0);

    rclcpp::QoS sensor_qos(rclcpp::KeepLast(5));
    sensor_qos.best_effort();
    sensor_qos.durability_volatile();

    color_sub_ =
      create_subscription<sensor_msgs::msg::Image>(
      color_topic,
      sensor_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg)
      {
        color_stats_.add(
          msg,
          std::chrono::steady_clock::now());
      });

    depth_sub_ =
      create_subscription<sensor_msgs::msg::Image>(
      depth_topic,
      sensor_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg)
      {
        depth_stats_.add(
          msg,
          std::chrono::steady_clock::now());
      });

    timer_ =
      create_wall_timer(
      std::chrono::duration<double>(report_period),
      [this]()
      {
        RCLCPP_INFO(
          get_logger(),
          "\n========== R10 CAMERA BENCHMARK ==========");

        color_stats_.report(get_logger());
        depth_stats_.report(get_logger());
      });

    RCLCPP_INFO(
      get_logger(),
      "Benchmark actif.\n"
      "COLOR: %s\n"
      "DEPTH: %s\n"
      "QoS subscriber: BEST_EFFORT / VOLATILE / KEEP_LAST(5)",
      color_topic.c_str(),
      depth_topic.c_str());
  }

private:
  StreamStats color_stats_;
  StreamStats depth_stats_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<RealSenseImageBenchmark>());

  rclcpp::shutdown();

  return 0;
}
