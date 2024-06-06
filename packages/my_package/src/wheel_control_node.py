#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge
import numpy as np

# throttle and direction for each wheel
THROTTLE_LEFT = 0.5        # 50% throttle
DIRECTION_LEFT = 1         # forward
THROTTLE_RIGHT = 0.75       # 30% throttle
DIRECTION_RIGHT = 1       # backward

CONST = 0.3
GAIN = 0.4
LEFT = 0.2
RIGHT = 0.2

class WheelControlNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._bridge = CvBridge()
        # form the message
        self._vel_left = THROTTLE_LEFT * DIRECTION_LEFT
        self._vel_right = THROTTLE_RIGHT * DIRECTION_RIGHT
        # construct publisher
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self.sub = rospy.Subscriber("eyes", CompressedImage, self.callback)


    def callback(self, msg):
        concatenated_images = self._bridge.compressed_imgmsg_to_cv2(msg)
        height, width, _ = concatenated_images.shape
        left_image = concatenated_images[:, :width//2]
        right_image = concatenated_images[:, width//2:]

        # Convert images to grayscale (if they are not already)
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Count the non-zero elements in each half
        max_count = height * width
        left_non_zero_count = cv2.countNonZero(left_gray) / max_count
        right_non_zero_count = cv2.countNonZero(right_gray) / max_count

        left_motor = CONST + GAIN * left_non_zero_count * LEFT
        right_motor = CONST + GAIN * right_non_zero_count * RIGHT

        if(left_non_zero_count > right_non_zero_count):
            right_motor -= 0.2
            right_motor = 0.25 if right_motor < 0.25 else right_motor

            left_motor += 0.25
            left_motor = 0.35 if left_motor > 0.35 else left_motor

            message = WheelsCmdStamped(vel_left=left_motor, vel_right=right_motor)
            self._publisher.publish(message)

        elif(left_non_zero_count < right_non_zero_count):
            left_motor -= 0.2
            left_motor = 0.25 if left_motor < 0.25 else left_motor

            right_motor += 0.25
            right_motor = 0.35 if right_motor > 0.35 else right_motor


            message = WheelsCmdStamped(vel_left=left_motor, vel_right=right_motor)
            self._publisher.publish(message)   
        
        else:
            message = WheelsCmdStamped(vel_left=left_motor, vel_right=right_motor)
            self._publisher.publish(message) 

        self._vel_left = left_motor
        self._vel_right = right_motor

        rospy.loginfo(f"I heard {left_motor}, {right_motor}")


    def run(self):
        # publish 10 messages every second (10 Hz)
        rate = rospy.Rate(0.1)
        message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
        while not rospy.is_shutdown():
            self._publisher.publish(message)
            rate.sleep()

    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

if __name__ == '__main__':

    # create the node
    node = WheelControlNode(node_name='wheel_control_node')
    # run node
    # node.run()
    # keep the process from terminating
    rospy.spin()
