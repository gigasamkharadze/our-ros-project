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
        """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
        """
        h, w, _ = image.shape
    
        consta = 100
        white_lower_color = np.array([0,0,255-consta])
        white_upper_color = np.array([255,consta,255])
        yellow_lower_color = np.array([10,50,80])
        yellow_upper_color = np.array([30,255,255])
    
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
        gaussian_blur = cv2.GaussianBlur(img_gray_scale, (0,0), np.pi)
        sobel_x_crd = cv2.Sobel(gaussian_blur, cv2.CV_64F,1,0)
        sobel_y_crd = cv2.Sobel(gaussian_blur, cv2.CV_64F,0,1)
        G = np.sqrt(sobel_x_crd*sobel_x_crd + sobel_y_crd*sobel_y_crd)
    
        threshold_w = 27
        threshold_y = 47
    
        mask_bool_white = (G > threshold_w)
        mask_bool_yellow = (G > threshold_y)
    
        real_mask_white = cv2.inRange(img_hsv, white_lower_color, white_upper_color)
        real_mask_yellow = cv2.inRange(img_hsv, yellow_lower_color, yellow_upper_color)
    
        left_mask = np.ones(sobel_x_crd.shape)
        left_mask[:,int(np.floor(w/2)):w+1] = 0
        right_mask = np.ones(sobel_x_crd.shape)
        right_mask[:,0:int(np.floor(w/2))] = 0
        left_mask[:int(sobel_x_crd.shape[0]/2)] = 0
        right_mask[:int(sobel_x_crd.shape[0]/2)] = 0
        sobel_x_pos_mask = (sobel_x_crd > 0)
        sobel_x_neg_mask = (sobel_x_crd < 0)
        sobel_y_pos_mask = (sobel_y_crd > 0)
        sobel_y_neg_mask = (sobel_y_crd < 0)
    
        left_edge_mask = left_mask * mask_bool_yellow * sobel_x_neg_mask * sobel_y_neg_mask * real_mask_yellow
        right_edge_mask = right_mask * mask_bool_white * sobel_x_pos_mask * sobel_y_neg_mask * real_mask_white
    
        return  left_edge_mask + right_edge_mask

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
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        # construct publisher
        self.pub = rospy.Publisher("eyes", CompressedImage, queue_size=10)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        image = self.detect_lane_markings(image)
        # publish the image
        self.pub.publish(msg)
        # display frame
        cv2.imshow(self._window, image)
        cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
