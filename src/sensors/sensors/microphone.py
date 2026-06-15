"""Captures audio from the microphone and publishes it."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
import pyaudio
import numpy as np

OUTPUT_RATE = 50 
NODE_NAME = "microphone"
TOPIC_OUT_AUDIO = "/audio_data"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

CHUNK = int(RATE / OUTPUT_RATE)

class MicrophoneNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        
        self.publisher_ = self.create_publisher(Int16MultiArray, TOPIC_OUT_AUDIO, 100)

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        self.timer = self.create_timer(1.0 / OUTPUT_RATE, self.timer_callback)
        self.get_logger().info(f"microphone node started streaming at {RATE} Hz...")

    def timer_callback(self):
        try:
            raw_data = self.stream.read(CHUNK, exception_on_overflow=False)

            msg = Int16MultiArray()
            msg.data = np.frombuffer(raw_data, dtype=np.int16).tolist()
            self.publisher_.publish(msg)

        except Exception as e:
            self.get_logger().error(f"error reading microphone data: {e}")

    def cleanup(self):
        self.timer.cancel()
        if self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = MicrophoneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        if rclpy.ok():
            node.cleanup()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
