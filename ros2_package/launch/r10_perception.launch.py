from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_camera = LaunchConfiguration('use_camera')
    use_clustering = LaunchConfiguration('use_clustering')
    use_rviz = LaunchConfiguration('use_rviz')
    use_multiplane_debug = LaunchConfiguration('use_multiplane_debug')
    support_plane_id = LaunchConfiguration('support_plane_id')
    use_support_selector = LaunchConfiguration('use_support_selector')
    cluster_input = LaunchConfiguration('cluster_input')
    use_tracking = LaunchConfiguration('use_tracking')
    use_grasp_candidates = LaunchConfiguration('use_grasp_candidates')
    use_grasp_filter = LaunchConfiguration('use_grasp_filter')
    use_antipodal_filter = LaunchConfiguration('use_antipodal_filter')
    use_grasp_refinement = LaunchConfiguration('use_grasp_refinement')

    x_min = LaunchConfiguration('x_min')
    x_max = LaunchConfiguration('x_max')
    y_min = LaunchConfiguration('y_min')
    y_max = LaunchConfiguration('y_max')
    z_min = LaunchConfiguration('z_min')
    z_max = LaunchConfiguration('z_max')

    voxel_leaf = LaunchConfiguration('voxel_leaf')

    ransac_distance = LaunchConfiguration('ransac_distance')
    ransac_iterations = LaunchConfiguration('ransac_iterations')
    ransac_min_inliers = LaunchConfiguration('ransac_min_inliers')
    ransac_every_n = LaunchConfiguration('ransac_every_n')
    ransac_axis_tolerance_deg = LaunchConfiguration('ransac_axis_tolerance_deg')

    cluster_tolerance = LaunchConfiguration('cluster_tolerance')
    cluster_min_size = LaunchConfiguration('cluster_min_size')
    cluster_max_size = LaunchConfiguration('cluster_max_size')

    return LaunchDescription([
        # Child processes inherit CycloneDDS, which is the validated RMW for this R10 pipeline.
        SetEnvironmentVariable(
            name='RMW_IMPLEMENTATION',
            value='rmw_cyclonedds_cpp'
        ),

        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_clustering', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('use_support_selector', default_value='true'),
        DeclareLaunchArgument('support_plane_id', default_value='-1'),
        DeclareLaunchArgument('cluster_input', default_value='/r10/cloud_without_support'),
        DeclareLaunchArgument('use_tracking', default_value='true'),
        DeclareLaunchArgument('use_grasp_candidates', default_value='true'),
        DeclareLaunchArgument('use_grasp_filter', default_value='true'),
        DeclareLaunchArgument('use_antipodal_filter', default_value='true'),
        DeclareLaunchArgument('use_grasp_refinement', default_value='true'),
        DeclareLaunchArgument('use_multiplane_debug', default_value='true'),

        DeclareLaunchArgument('x_min', default_value='-0.45'),
        DeclareLaunchArgument('x_max', default_value='0.45'),
        DeclareLaunchArgument('y_min', default_value='-0.35'),
        DeclareLaunchArgument('y_max', default_value='0.35'),
        DeclareLaunchArgument('z_min', default_value='0.15'),
        DeclareLaunchArgument('z_max', default_value='1.50'),

        DeclareLaunchArgument('voxel_leaf', default_value='0.005'),

        DeclareLaunchArgument('ransac_distance', default_value='0.010'),
        DeclareLaunchArgument('ransac_iterations', default_value='800'),
        DeclareLaunchArgument('ransac_min_inliers', default_value='800'),
        DeclareLaunchArgument('ransac_every_n', default_value='3'),
        DeclareLaunchArgument('ransac_axis_tolerance_deg', default_value='12.0'),

        DeclareLaunchArgument('cluster_tolerance', default_value='0.020'),
        DeclareLaunchArgument('cluster_min_size', default_value='100'),
        DeclareLaunchArgument('cluster_max_size', default_value='30000'),

        LogInfo(msg=[
            '\n=== R10 PERCEPTION PIPELINE ===\n',
            'RealSense -> CropBox -> VoxelGrid -> RANSAC -> Euclidean clustering\n',
            'RMW: rmw_cyclonedds_cpp\n',
            'Robot motion: NONE\n',
            '================================\n'
        ]),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            condition=IfCondition(use_camera),
            parameters=[{
                'enable_color': True,
                'enable_depth': True,
                'rgb_camera.color_profile': '1280x720x30',
                'depth_module.depth_profile': '848x480x30',
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'align_depth.enable': True,
                'pointcloud.enable': True,
                'color_qos': 'SENSOR_DATA',
                'depth_qos': 'SENSOR_DATA',
                'depth_module.emitter_enabled': 1,
                'depth_module.laser_power': 150.0,
                'diagnostics_period': 1.0,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_cloud_filter',
            name='realsense_cloud_filter',
            output='screen',
            parameters=[{
                'input_topic': '/camera/camera/depth/color/points',
                'output_topic': '/r10/cloud_filtered',
                'x_min': ParameterValue(x_min, value_type=float),
                'x_max': ParameterValue(x_max, value_type=float),
                'y_min': ParameterValue(y_min, value_type=float),
                'y_max': ParameterValue(y_max, value_type=float),
                'z_min': ParameterValue(z_min, value_type=float),
                'z_max': ParameterValue(z_max, value_type=float),
                'log_every_n': 30,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_voxel_filter',
            name='realsense_voxel_filter',
            output='screen',
            parameters=[{
                'input_topic': '/r10/cloud_filtered',
                'output_topic': '/r10/cloud_voxel',
                'leaf_size': ParameterValue(voxel_leaf, value_type=float),
                'log_every_n': 30,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_plane_ransac',
            name='realsense_plane_ransac',
            output='screen',
            parameters=[{
                'input_topic': '/r10/cloud_voxel',
                'plane_topic': '/r10/cloud_plane',
                'objects_topic': '/r10/cloud_without_plane',
                'distance_threshold': ParameterValue(
                    ransac_distance, value_type=float),
                'max_iterations': ParameterValue(
                    ransac_iterations, value_type=int),
                'min_inliers': ParameterValue(
                    ransac_min_inliers, value_type=int),
                'process_every_n': ParameterValue(
                    ransac_every_n, value_type=int),
                'use_axis_constraint': True,
                'axis_x': 0.0,
                'axis_y': 0.0,
                'axis_z': 1.0,
                'axis_tolerance_deg': ParameterValue(
                    ransac_axis_tolerance_deg, value_type=float),
                'log_every_n': 10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_euclidean_cluster',
            name='realsense_euclidean_cluster',
            output='screen',
            condition=IfCondition(use_clustering),
            parameters=[{
                'input_topic': cluster_input,
                'largest_topic': '/r10/cloud_largest_cluster',
                'centroid_topic': '/r10/largest_cluster_centroid',
                'cluster_tolerance': ParameterValue(
                    cluster_tolerance, value_type=float),
                'min_cluster_size': ParameterValue(
                    cluster_min_size, value_type=int),
                'max_cluster_size': ParameterValue(
                    cluster_max_size, value_type=int),
                'process_every_n': 1,
                'log_every_n': 10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_multiplane_ransac',
            name='realsense_multiplane_ransac',
            output='screen',
            condition=IfCondition(use_multiplane_debug),
            parameters=[{
                'input_topic': '/r10/cloud_voxel',
                'colored_topic': '/r10/cloud_planes_colored',
                'remainder_topic': '/r10/cloud_nonplanar',
                'markers_topic': '/r10/plane_markers',
                'max_planes': 5,
                'distance_threshold': 0.010,
                'max_iterations': 400,
                'min_plane_size': 300,
                'process_every_n': 10,
                'log_every_n': 1,
                'normal_arrow_length': 0.10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_support_plane_selector',
            name='realsense_support_plane_selector',
            output='screen',
            condition=IfCondition(use_support_selector),
            parameters=[{
                'input_topic': '/r10/cloud_voxel',
                'support_topic': '/r10/cloud_support_plane',
                'without_support_topic': '/r10/cloud_without_support',
                'markers_topic': '/r10/support_plane_marker',
                'max_planes': 5,
                'calibration_plane_id': ParameterValue(
                    support_plane_id, value_type=int),
                'distance_threshold': 0.010,
                'max_iterations': 400,
                'min_plane_size': 300,
                'process_every_n': 10,
                'max_normal_angle_deg': 15.0,
                'max_plane_offset_m': 0.050,
                'log_every_n': 1,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_temporal_obb_tracker',
            name='realsense_temporal_obb_tracker',
            output='screen',
            condition=IfCondition(use_tracking),
            parameters=[{
                'input_topic': '/r10/cluster_geometry_markers',
                'markers_topic': '/r10/tracked_object_markers',
                'poses_topic': '/r10/tracked_obb_poses',
                'max_position_distance': 0.15,
                'max_relative_dimension_error': 0.65,
                'position_alpha': 0.35,
                'dimension_alpha': 0.25,
                'orientation_alpha': 0.25,
                'max_missed_frames': 12,
                'log_every_n': 10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_grasp_candidate_generator',
            name='realsense_grasp_candidate_generator',
            output='screen',
            condition=IfCondition(use_grasp_candidates),
            parameters=[{
                'input_topic': '/r10/tracked_object_markers',
                'marker_topic': '/r10/grasp_candidate_markers',
                'grasp_pose_topic': '/r10/grasp_candidate_poses',
                'pregrasp_pose_topic': '/r10/pregrasp_candidate_poses',
                'pregrasp_distance': 0.10,
                'finger_clearance': 0.005,
                'contact_marker_radius': 0.008,
                'max_objects': 10,
                'log_every_n': 10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_grasp_candidate_filter',
            name='realsense_grasp_candidate_filter',
            output='screen',
            condition=IfCondition(use_grasp_filter),
            parameters=[{
                'objects_topic': '/r10/tracked_object_markers',
                'support_topic': '/r10/cloud_support_plane',
                'marker_topic': '/r10/grasp_filtered_markers',
                'grasp_pose_topic': '/r10/grasp_filtered_poses',
                'pregrasp_pose_topic': '/r10/pregrasp_filtered_poses',
                'max_gripper_opening_m': 0.080,
                'finger_clearance_m': 0.005,
                'pregrasp_distance_m': 0.10,
                'min_table_clearance_m': 0.015,
                'top_k_per_object': 3,
                'log_every_n': 10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_antipodal_grasp_validator',
            name='realsense_antipodal_grasp_validator',
            output='screen',
            condition=IfCondition(use_antipodal_filter),
            parameters=[{
                'cloud_topic': '/r10/cloud_without_support',
                'candidates_topic': '/r10/grasp_filtered_markers',
                'markers_topic': '/r10/grasp_antipodal_markers',
                'accepted_pose_topic': '/r10/grasp_antipodal_poses',
                'accepted_pregrasp_topic': '/r10/pregrasp_antipodal_poses',
                'max_contact_distance_m': 0.020,
                'normal_radius_m': 0.025,
                'min_neighbors': 8,
                'max_normal_alignment_deg': 30.0,
                'max_normal_opposition_deg': 30.0,
                'max_curvature': 0.12,
                'log_every_n': 10,
            }]
        ),

        Node(
            package='realsense_benchmark',
            executable='realsense_grasp_refinement_selector',
            name='realsense_grasp_refinement_selector',
            output='screen',
            condition=IfCondition(use_grasp_refinement),
            parameters=[{
                'input_topic': '/r10/grasp_antipodal_markers',
                'marker_topic': '/r10/grasp_selected_markers',
                'grasp_pose_topic': '/r10/grasp_selected_poses',
                'pregrasp_pose_topic': '/r10/pregrasp_selected_poses',
                'max_gripper_opening_m': 0.080,
                'finger_clearance_m': 0.005,
                'pregrasp_distance_m': 0.10,
                'min_antipodal_score': 0.55,
                'log_every_n': 10,
            }]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_r10',
            output='screen',
            condition=IfCondition(use_rviz),
        ),
    ])
