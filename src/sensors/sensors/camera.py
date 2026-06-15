"""captures video frames from a usb camera and publishes them."""

import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

OUTPUT_RATE = 60  # Hz
NODE_NAME = "camera"
TOPIC_OUT_IMAGE = "/video_data"

class CameraNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        
        self.bridge = CvBridge()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("could not open usb camera!")
            
        self.publisher_ = self.create_publisher(Image, TOPIC_OUT_IMAGE, 10)
        
        self.timer = self.create_timer(1.0 / OUTPUT_RATE, self.timer_callback)
        self.get_logger().info("camera node started, sending data.")

        time.sleep(2.0)
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("failed to capture frame from camera")

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            return
            
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        
        self.publisher_.publish(msg)

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
