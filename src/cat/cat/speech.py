"""
listens for audio data, performs speech recognition using vosk, and publishes commands for the brain
"""

import os
import json
import numpy as np
import rclpy
from rclpy.node import Node
from cat_msgs.msg import Speech
from std_msgs.msg import Int16MultiArray, Header
from vosk import Model, KaldiRecognizer

NODE_NAME = "speech"
TOPIC_IN_AUDIO = "/audio_data"
TOPIC_OUT_HEADER = "/speech_data"
home_dir = os.path.expanduser("~")
MODEL_PATH = os.path.join(
    home_dir, 
    "ros2_ws/src/cat/models/vosk-model-small-en-us-0.15"
)

class SpeechNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.get_logger().info("initializing speech node:")
        
        try:
            if not os.path.exists(MODEL_PATH):
                self.get_logger().error(f"model path not found: {MODEL_PATH}")
                return

            self.model = Model(MODEL_PATH)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.get_logger().info("vosk model loaded!")
        except Exception as e:
            self.get_logger().error(f"failed to load vosk: {e}")
            return

        self.publisher = self.create_publisher(Speech, TOPIC_OUT_HEADER, 10)
        self.subscriber = self.create_subscription(
            Int16MultiArray, 
            TOPIC_IN_AUDIO, 
            self.audio_callback, 
            10
        )

    def audio_callback(self, msg):
        if self.recognizer is None:
            return

        audio_16k = np.array(msg.data, dtype=np.int16)

        if self.recognizer.AcceptWaveform(audio_16k.tobytes()):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").lower()
            if text:
                self.get_logger().info(f"detected: {text}")
                self.process_text(text)

    def process_text(self, text):
        if "follow" in text or "come" in text:
            self.publish_command(Speech.FOLLOW)
        elif "go away" in text in text:
            self.publish_command(Speech.GO_AWAY)
        elif "good" in text:
            self.publish_command(Speech.PRAISE)
        elif "play" in text:
            self.publish_command(Speech.PLAY)
        elif "stop" in text:
            self.publish_command(Speech.STOP)
        elif "rest" in text or "sleep" in text:
            self.publish_command(Speech.REST)
        elif "wake up" in text:
            self.publish_command(Speech.WAKE_UP)
        elif "dance" in text:
            self.publish_command(Speech.DANCE)

    def publish_command(self, command_value):
        msg = Speech()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "cat"
        
        msg.speech = command_value 
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpeechNode()
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
