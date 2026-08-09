# ROS2 Loaned Image A/B

```bash
mkdir -p /tmp/ros2-loan-ws/src
cp -r /home/ghr/code/cuda_pytorch/Performance_Tuning/profiling/src/integrations/ros2_loaned_image /tmp/ros2-loan-ws/src/
cd /tmp/ros2-loan-ws && colcon build --symlink-install
source install/setup.bash
ros2 run ros2_loaned_image loaned_image_nodes --ros-args -p use_loan:=false
ros2 run ros2_loaned_image loaned_image_nodes --ros-args -p use_loan:=true
```

Fast DDS Shared Memory对照：

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/ghr/code/cuda_pytorch/Performance_Tuning/profiling/src/configs/fastdds_shm_profile.xml
ros2 run ros2_loaned_image loaned_image_nodes --ros-args -p use_loan:=true
```

以运行时输出的`can_loan_messages`为准。分别采集CPU、topic bw、callback age和ros2 trace；不要仅因使用SHM配置就声称完全Zero-Copy。
