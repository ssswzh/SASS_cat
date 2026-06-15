from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # sensors
        Node(
            package="sensors",
            executable="camera",
            name="camera",
            output="screen"
        ),

        Node(
            package="sensors",
            executable="microphone",
            name="microphone",
            output="screen"
        ),

        # Cat processing nodes
        Node(
            package="cat",
            executable="vision",
            name="vision",
            output="screen"
        ),

        Node(
            package="cat",
            executable="speech",
            name="speech",
            output="screen"
        ),

        # Actuator nodes
        Node(
            package="actuators",
            executable="head",
            name="head",
            output="screen"
        ),

        Node(
            package="actuators",
            executable="speaker",
            name="speaker",
            output="screen"
        ),
    ])
