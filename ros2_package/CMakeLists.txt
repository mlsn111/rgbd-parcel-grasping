cmake_minimum_required(VERSION 3.8)
project(realsense_benchmark)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation sample_consensus search kdtree)

add_executable(
  realsense_image_benchmark
  src/realsense_image_benchmark.cpp
)
ament_target_dependencies(
  realsense_image_benchmark
  rclcpp
  sensor_msgs
)

add_executable(
  realsense_depth_quality
  src/realsense_depth_quality.cpp
)
ament_target_dependencies(
  realsense_depth_quality
  rclcpp
  sensor_msgs
)

add_executable(
  realsense_pixel_to_3d
  src/realsense_pixel_to_3d.cpp
)
ament_target_dependencies(
  realsense_pixel_to_3d
  rclcpp
  sensor_msgs
  geometry_msgs
)

add_executable(
  realsense_cloud_filter
  src/realsense_cloud_filter.cpp
)
ament_target_dependencies(
  realsense_cloud_filter
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_include_directories(
  realsense_cloud_filter PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_cloud_filter ${PCL_LIBRARIES}
)

add_executable(
  realsense_voxel_filter
  src/realsense_voxel_filter.cpp
)
ament_target_dependencies(
  realsense_voxel_filter
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_include_directories(
  realsense_voxel_filter PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_voxel_filter ${PCL_LIBRARIES}
)

add_executable(
  realsense_plane_ransac
  src/realsense_plane_ransac.cpp
)
ament_target_dependencies(
  realsense_plane_ransac
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_include_directories(
  realsense_plane_ransac PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_plane_ransac ${PCL_LIBRARIES}
)

add_executable(
  realsense_euclidean_cluster
  src/realsense_euclidean_cluster.cpp
)
ament_target_dependencies(
  realsense_euclidean_cluster
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
)
target_include_directories(
  realsense_euclidean_cluster PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_euclidean_cluster ${PCL_LIBRARIES}
)


add_executable(
  realsense_multiplane_ransac
  src/realsense_multiplane_ransac.cpp
)
ament_target_dependencies(
  realsense_multiplane_ransac
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
)
target_include_directories(
  realsense_multiplane_ransac PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_multiplane_ransac ${PCL_LIBRARIES}
)


add_executable(
  realsense_support_plane_selector
  src/realsense_support_plane_selector.cpp
)
ament_target_dependencies(
  realsense_support_plane_selector
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
)
target_include_directories(
  realsense_support_plane_selector PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_support_plane_selector ${PCL_LIBRARIES}
)


add_executable(
  realsense_temporal_obb_tracker
  src/realsense_temporal_obb_tracker.cpp
)
ament_target_dependencies(
  realsense_temporal_obb_tracker
  rclcpp
  geometry_msgs
  visualization_msgs
)
target_include_directories(
  realsense_temporal_obb_tracker PRIVATE ${PCL_INCLUDE_DIRS}
)


add_executable(
  realsense_grasp_candidate_generator
  src/realsense_grasp_candidate_generator.cpp
)
ament_target_dependencies(
  realsense_grasp_candidate_generator
  rclcpp
  geometry_msgs
  visualization_msgs
)
target_include_directories(
  realsense_grasp_candidate_generator PRIVATE ${PCL_INCLUDE_DIRS}
)


add_executable(
  realsense_grasp_candidate_filter
  src/realsense_grasp_candidate_filter.cpp
)
ament_target_dependencies(
  realsense_grasp_candidate_filter
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
)
target_include_directories(
  realsense_grasp_candidate_filter PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_grasp_candidate_filter ${PCL_LIBRARIES}
)


add_executable(
  realsense_antipodal_grasp_validator
  src/realsense_antipodal_grasp_validator.cpp
)
ament_target_dependencies(
  realsense_antipodal_grasp_validator
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
)
target_include_directories(
  realsense_antipodal_grasp_validator PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
  realsense_antipodal_grasp_validator ${PCL_LIBRARIES}
)


add_executable(
  realsense_grasp_refinement_selector
  src/realsense_grasp_refinement_selector.cpp
)
ament_target_dependencies(
  realsense_grasp_refinement_selector
  rclcpp
  geometry_msgs
  visualization_msgs
)
target_include_directories(
  realsense_grasp_refinement_selector PRIVATE ${PCL_INCLUDE_DIRS}
)

install(
  TARGETS
    realsense_image_benchmark
    realsense_depth_quality
    realsense_pixel_to_3d
    realsense_cloud_filter
    realsense_voxel_filter
    realsense_plane_ransac
    realsense_euclidean_cluster
    realsense_multiplane_ransac
    realsense_support_plane_selector
    realsense_temporal_obb_tracker
    realsense_grasp_candidate_generator
    realsense_grasp_candidate_filter
    realsense_antipodal_grasp_validator
    realsense_grasp_refinement_selector
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
