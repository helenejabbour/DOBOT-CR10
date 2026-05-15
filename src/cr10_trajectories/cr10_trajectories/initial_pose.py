#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time


class ZeroJointStatePublisher(Node):
    def __init__(self):
        super().__init__('zero_joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

        joint_state = JointState()
        joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        for _ in range(5):
            joint_state.header.stamp = self.get_clock().now().to_msg()
            self.get_logger().info('Initializing joints to zero pose.')
            self.publisher_.publish(joint_state)
            time.sleep(0.5)

        self.get_logger().info('Initial pose set. Shutting down.')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    ZeroJointStatePublisher()


if __name__ == '__main__':
    main()
