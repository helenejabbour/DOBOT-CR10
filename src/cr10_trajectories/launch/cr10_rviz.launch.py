from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cr10_trajectories',
            executable='trajectory_node',
            name='trajectory_node',
            output='screen',
        ),
        Node(
            package='cr10_trajectories',
            executable='gui_node',
            name='gui_node',
            output='screen',
        ),
    ])