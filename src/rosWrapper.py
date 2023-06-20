#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from interface import HSMInterface
from stereo_msgs.msg import DisparityImage
import message_filters


class HSMWrapper():
    def __init__(self,leftTopic,rightTopic,modelPath,max_disparity,level,clean,testres=1) -> None:
        self.leftTopic = leftTopic
        self.rightTopic = rightTopic
        self.max_disparity = max_disparity
        self.hsm = HSMInterface(max_disparity,level,clean,testres)
        self.hsm.load_model(modelPath)
        self.bridge = CvBridge()
        self.leftImage = None
        self.rightImage = None
        self.disparityPub = rospy.Publisher("/zed/manual/disparity",DisparityImage,queue_size=1)
    
    def getCameraParameters(self)->None:
        #Hardcoded for now
        self.camerasParameters = {}
        self.camerasParameters["focalLength"] = 359.0479229
        # self.camerasParameters["focalLength"] = 697.0
        self.camerasParameters["baseline"] = 0.120161
        self.camerasParameters["doffs"] = 9

    def callBack(self,leftImage,rightImage):
        self.leftImage = leftImage
        self.rightImage = rightImage

    def run(self):
        if self.leftImage is None or self.rightImage is None:
            return
        leftImage = self.bridge.imgmsg_to_cv2(self.leftImage,desired_encoding='passthrough')
        rightImage = self.bridge.imgmsg_to_cv2(self.rightImage,desired_encoding='passthrough')
        disparity,entropy = self.hsm.test(leftImage,rightImage)
        
        img = disparity.copy()
        disparityImage = DisparityImage()
        disparityImage.header = self.leftImage.header
        disparityImage.image = self.bridge.cv2_to_imgmsg(img, encoding="32FC1")
        disparityImage.f = self.camerasParameters["focalLength"]
        disparityImage.T = self.camerasParameters["baseline"]
        disparityImage.min_disparity = 0
        disparityImage.max_disparity = self.max_disparity
        disparityImage.delta_d = 1.0/self.max_disparity
        disparityImage.valid_window.x_offset = 0
        disparityImage.valid_window.y_offset = 0
        disparityImage.valid_window.width = self.leftImage.width
        disparityImage.valid_window.height = self.leftImage.height
        disparityImage.image.header.frame_id = "camera_link"
        self.disparityPub.publish(disparityImage)


    def start(self):
        rospy.init_node("hsm_wrapper")
        self.getCameraParameters()
        leftSub = message_filters.Subscriber(self.leftTopic,Image)
        rightSub = message_filters.Subscriber(self.rightTopic,Image)
        ts = message_filters.ApproximateTimeSynchronizer([leftSub,rightSub],queue_size=1,slop=0.1)
        ts.registerCallback(self.callBack)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.run()
            rate.sleep()
            