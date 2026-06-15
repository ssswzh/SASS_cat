"""listens for meow messages and plays corresponding cat sounds via an external audio player."""

import os
import glob
import random
import subprocess
import rclpy
from rclpy.node import Node
from cat_msgs.msg import Meow

NODE_NAME = "speaker"
TOPIC_IN_MEOW = "/brain_meow"
SOUNDS_DIR = os.path.expanduser("~/ros2_ws/sounds/")

class SpeakerNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.active_process = None
        
        self.sound_library = {
            Meow.MEOW: [],
            Meow.PURR: [],
            Meow.HISS: [],
            Meow.ANGRY: [],
            Meow.CURIOUS: [],
            Meow.HAPPY: []
        }
        
        self.preload_sounds()
        self.subscriber_ = self.create_subscription(Meow, TOPIC_IN_MEOW, self.meow_callback, 10)

    def preload_sounds(self):
        """
        we preload by loading multiple sounds for each category
        """

        patterns = {
            Meow.MEOW: "meow_*.mp3",
            Meow.PURR: "purr_*.mp3",
            Meow.HISS: "hiss_*.mp3",
            Meow.ANGRY: "angry_*.mp3",
            Meow.CURIOUS: "curious_*.mp3",
            Meow.HAPPY: "happy_*.mp3",
        }
        
        for sound_id, pattern in patterns.items():
            search_path = os.path.join(SOUNDS_DIR, pattern)
            self.sound_library[sound_id] = glob.glob(search_path)
            
            self.get_logger().debug(f"loaded {len(self.sound_library[sound_id])} files for sound_id {sound_id}")

    def play_sound(self, sound_id):
        sound_files = self.sound_library.get(sound_id, [])
        if not sound_files:
            self.get_logger().warn(f"no audio files found for sound_id: {sound_id}")
            return

        if self.active_process and self.active_process.poll() is None:
            self.active_process.terminate()
            self.active_process.wait()

        chosen_file = random.choice(sound_files)
        self.get_logger().info(f"playing audio: {os.path.basename(chosen_file)}")

        try:
            # yes, we use mpv.
            self.active_process = subprocess.Popen(
                ["mpv", "--no-video", chosen_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            self.get_logger().error("could not play sound!!!")

    def meow_callback(self, msg):
        self.play_sound(msg.sound_id)

    def destroy_node(self):
        if self.active_process and self.active_process.poll() is None:
            self.active_process.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeakerNode()
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
