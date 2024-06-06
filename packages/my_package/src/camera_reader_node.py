#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge
import numpy as np
from typing import Tuple



class CameraReaderNode(DTROS):

    def detect_lane_markings(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Apply bilateral filter to the image
        image = cv2.bilateralFilter(image, 12, 125, 155)
        
        # Split the image into left and right halves
        height, width, _ = image.shape
        left_half = image[:, :width//2]
        right_half = image[:, width//2:]     

        gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)  

        # Define color range for yellow
        yellow_lower_color = np.array([9, 100, 0])
        yellow_upper_color = np.array([80, 255, 255])
        # Define color range for white
        # white_lower_color = np.array([0, 0, 150])9 148 155 83 355 355
        # # white_upper_color = np.array([255, 60, 255])
        # white_lower_color = np.array([170, 80, 87]) #0 173 0
        # white_upper_color = np.array([270, 99, 185]) #179 255 255 hsl
        # white_lower_color = np.array([0, 173, 0])
        # white_upper_color = np.array([180, 255, 255])
        white_lower_color = np.array([0, 177, 0])
        white_upper_color = np.array([255, 255, 120])

        # Convert left half to HSV and right half to HLS
        hsv_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2HSV)
        # hsl_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2HLS)
        # l_channel = hsl_right[:,:,1]
        hls_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2HLS)
        # Split HLS image into its channels
        # h_channel, l_channel, s_channel = cv2.split(hls_right)


        # Create masks for yellow and white colors
        mask_yellow_left = cv2.inRange(hsv_left, yellow_lower_color, yellow_upper_color)
        mask_white_right = cv2.inRange(hls_right, white_lower_color, white_upper_color)
        # mask = cv2.inRange(l_channel, 180, 255)

        # Create result images
        result_left = cv2.bitwise_and(left_half, left_half, mask=mask_yellow_left)
        # result_right = cv2.bitwise_and(right_half, right_half, mask=mask_white_right)
        # result_right = cv2.bitwise_and(right_half, right_half, mask=mask_white_right)
        result_right = cv2.threshold(gray, 62, 255, cv2.THRESH_BINARY)


        result_left[:height//3] = 0
        # result_right[:height//3] = 0

        return result_left, result_right
    

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        self._window2 = "second-camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        # construct publisher
        self.pub = rospy.Publisher("eyes", CompressedImage, queue_size=10)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        left, right = self.detect_lane_markings(image)
        concatenated_img = np.hstack((left, right))
        concatenated_img_msg = self._bridge.cv2_to_compressed_imgmsg(concatenated_img)
        # publish the image
        self.pub.publish(concatenated_img_msg)
        # display frame
        cv2.imshow(self._window, concatenated_img)
        cv2.imshow(self._window2, image)
        cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
