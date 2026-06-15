"""
Processes video frames using yolo for object tracking and publishes the results for the brain
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

from ultralytics import YOLO
from cat_msgs.msg import VisionObject, VisionObjectArray

OUTPUT_RATE = 60
NODE_NAME = "vision"
TOPIC_IN_IMAGE = "/video_data"
TOPIC_OUT_VISION = "/vision_data"

class VisionNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        
        self.declare_parameter("window", False)
        self.window = self.get_parameter("window").value
        
        self.bridge = CvBridge()

        model_path = os.path.expanduser("~/yolo11m.pt")
        self.model = YOLO(model_path)
        self.get_logger().info("model loaded successfully.")
        self.latest_frame = None

        self.target_classes = {
            "person": VisionObject.PERSON,
            "bottle": VisionObject.BOTTLE,
            "mouse":  VisionObject.MOUSE,
            "bowl":   VisionObject.BOWL,
            "banana": VisionObject.BANANA
        }

        self.track_history = {}
        self.track_start_times = {}

        if self.window:
            cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
            cv2.startWindowThread()

        self.subscriber_ = self.create_subscription(Image, TOPIC_IN_IMAGE, self.img_callback, 1)
        self.publisher_ = self.create_publisher(VisionObjectArray, TOPIC_OUT_VISION, 10)
        self.timer = self.create_timer(1.0 / OUTPUT_RATE, self.timer_callback)

    def img_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"failed to convert image: {e}")

    def timer_callback(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame
        
        img_h, img_w = frame.shape[:2]
        img_center_x = img_w / 2.0
        img_center_y = img_h / 2.0

        current_time_ns = self.get_clock().now().nanoseconds
        
        results = self.model.track(source=frame, persist=True, device="cuda", conf=0.25, verbose=False)

        out_msg = VisionObjectArray()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = "camera_frame"

        current_frame_history = {} # per object information
        render_overlays = []

        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            
            active_combos = set()

            for box, track_id, class_id in zip(boxes, track_ids, classes):
                class_name = self.model.names[class_id]

                if class_name in self.target_classes:
                    tracking_key = (track_id, class_name)
                    active_combos.add(tracking_key)

                    obj_msg = VisionObject()
                    obj_msg.identifier = self.target_classes[class_name]
                    obj_msg.confidence = float(box.conf[0].item())
                    # relaxed confidences for some types
                    if obj_msg.identifier == VisionObject.PERSON and obj_msg.confidence < 0.4:
                        continue
                    if obj_msg.identifier == VisionObject.BOTTLE and obj_msg.confidence < 0.25:
                        continue

                    obj_msg.track_id = track_id
                    
                    xyxy = box.xyxy[0].tolist()
                    obj_msg.x_min = float(xyxy[0])
                    obj_msg.y_min = float(xyxy[1])
                    obj_msg.x_max = float(xyxy[2])
                    obj_msg.y_max = float(xyxy[3])
                    
                    cx = (obj_msg.x_min + obj_msg.x_max) / 2.0
                    cy = (obj_msg.y_min + obj_msg.y_max) / 2.0

                    obj_msg.error_x = float((cx - img_center_x) / img_center_x)
                    obj_msg.error_y = float((cy - img_center_y) / img_center_y)

                    if tracking_key not in self.track_start_times:
                        self.track_start_times[tracking_key] = current_time_ns
                    
                    duration_s = (current_time_ns - self.track_start_times[tracking_key]) / 1e9
                    obj_msg.duration = float(duration_s)

                    if track_id in self.track_history:
                        prev_cx, prev_cy, prev_time_ns = self.track_history[track_id]
                        dt_s = (current_time_ns - prev_time_ns) / 1e9
                        
                        if dt_s > 0:
                            obj_msg.v_x = (cx - prev_cx) / dt_s
                            obj_msg.v_y = (cy - prev_cy) / dt_s
                        else:
                            obj_msg.v_x = 0.0
                            obj_msg.v_y = 0.0
                    else:
                        obj_msg.v_x = 0.0
                        obj_msg.v_y = 0.0

                    current_frame_history[track_id] = (cx, cy, current_time_ns)

                    if obj_msg.identifier == VisionObject.BOTTLE or duration_s > 0.3:
                        out_msg.objects.append(obj_msg)
                    
                    if self.window:
                        render_overlays.append((int(obj_msg.x_min), int(obj_msg.y_min) - 8, class_name, track_id, duration_s))

            # clean up start times
            self.track_start_times = {k: v for k, v in self.track_start_times.items() if k in active_combos}
        else:
            self.track_start_times = {}

        self.track_history = current_frame_history
        self.publisher_.publish(out_msg)

        if self.window:
            annotated_frame = results[0].plot(labels=False, conf=False)
            for x, y, c_name, t_id, dur in render_overlays:
                text = f"{c_name} #{t_id} ({dur:.2f}s)"
                cv2.putText(annotated_frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
            cv2.imshow("YOLO", annotated_frame)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.window:
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
