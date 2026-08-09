// Purpose: ROS2 large Image普通publish与Loaned Message A/B，并测callback age。
// 注意：can_loan_messages()取决于RMW/DDS和消息类型，失败时安全回退普通消息。
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

using namespace std::chrono_literals;

class PublisherNode : public rclcpp::Node {
 public:
  PublisherNode() : Node("loaned_image_publisher") {
    width_ = declare_parameter<int>("width", 1920);
    height_ = declare_parameter<int>("height", 1080);
    fps_ = declare_parameter<double>("fps", 30.0);
    use_loan_ = declare_parameter<bool>("use_loan", true);
    auto qos = rclcpp::SensorDataQoS().keep_last(2);
    pub_ = create_publisher<sensor_msgs::msg::Image>("/profiling/loaned_image", qos);
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / fps_), [this] { publish(); });
    RCLCPP_INFO(get_logger(), "can_loan_messages=%s", pub_->can_loan_messages() ? "true" : "false");
  }
 private:
  void fill(sensor_msgs::msg::Image& msg) {
    msg.header.stamp = now();
    msg.header.frame_id = std::to_string(sequence_++);
    msg.width = width_; msg.height = height_; msg.encoding = "mono8"; msg.step = width_;
    msg.data.resize(static_cast<std::size_t>(width_) * height_);
    std::fill(msg.data.begin(), msg.data.end(), static_cast<uint8_t>(sequence_));
  }
  void publish() {
    if (use_loan_ && pub_->can_loan_messages()) {
      auto loaned = pub_->borrow_loaned_message();
      fill(loaned.get());
      pub_->publish(std::move(loaned));
    } else {
      sensor_msgs::msg::Image msg; fill(msg); pub_->publish(std::move(msg));
    }
  }
  int width_, height_; double fps_; bool use_loan_; uint64_t sequence_{0};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

class SubscriberNode : public rclcpp::Node {
 public:
  SubscriberNode() : Node("loaned_image_subscriber") {
    sub_ = create_subscription<sensor_msgs::msg::Image>("/profiling/loaned_image",
      rclcpp::SensorDataQoS().keep_last(2),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        const auto sent = rclcpp::Time(msg->header.stamp);
        const double age_ms = (now() - sent).seconds() * 1000.0;
        ++received_; checksum_ += msg->data.empty() ? 0 : msg->data.front();
        if (received_ % 100 == 0)
          RCLCPP_INFO(get_logger(), "received=%lu age_ms=%.3f bytes=%zu checksum=%lu",
                      received_, age_ms, msg->data.size(), checksum_);
      });
  }
 private:
  uint64_t received_{0}, checksum_{0};
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc,char**argv){rclcpp::init(argc,argv);auto pub=std::make_shared<PublisherNode>();auto sub=std::make_shared<SubscriberNode>();rclcpp::executors::MultiThreadedExecutor exec;exec.add_node(pub);exec.add_node(sub);exec.spin();rclcpp::shutdown();}
