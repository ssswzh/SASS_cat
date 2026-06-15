"""controls the head and ear servos via hardware pwm based on incoming ros 2 head messages."""

import os
import time
import rclpy
from rclpy.node import Node
from cat_msgs.msg import Head

import Jetson.GPIO as GPIO

NODE_NAME = "head"
TOPIC_IN_HEAD = "/head_data"

PIN_HEAD = 15
PIN_EAR_L = 32
PIN_EAR_R = 33

# calibrated values - head
SERVO_HEAD_LEFT = 8.3
SERVO_HEAD_MIDDLE = 7.5
SERVO_HEAD_RIGHT = 6.5

# calibrated values - ears
SERVO_EAR_LEFT_LOW = 6.5
SERVO_EAR_LEFT_MIDDLE = 7.45
SERVO_EAR_LEFT_HIGH = 8.4

SERVO_EAR_RIGHT_LOW = 8.4
SERVO_EAR_RIGHT_MIDDLE = 7.2
SERVO_EAR_RIGHT_HIGH = 6.0


class HeadControllerNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        
        GPIO.setmode(GPIO.BOARD)
        
        # track execution intervals
        self.last_command_time = 0.0
        self.cooldown_duration = 2.0
        
        # head servo
        GPIO.setup(PIN_HEAD, GPIO.OUT)
        self.pwm_head = GPIO.PWM(PIN_HEAD, 50)
        self.pwm_head.start(SERVO_HEAD_MIDDLE)

        # left ear servo
        GPIO.setup(PIN_EAR_L, GPIO.OUT)
        self.pwm_ear_l = GPIO.PWM(PIN_EAR_L, 50)
        self.pwm_ear_l.start(SERVO_EAR_LEFT_HIGH)

        # right ear servo
        GPIO.setup(PIN_EAR_R, GPIO.OUT)
        self.pwm_ear_r = GPIO.PWM(PIN_EAR_R, 50)
        self.pwm_ear_r.start(SERVO_EAR_RIGHT_HIGH)
        
        self.get_logger().info("initializing default servo positions...")
        self.set_servos(SERVO_HEAD_MIDDLE, SERVO_EAR_LEFT_HIGH, SERVO_EAR_RIGHT_HIGH)

        self.subscriber_ = self.create_subscription(Head, TOPIC_IN_HEAD, self.head_callback, 1)
        self.get_logger().info("head controller initialized.")

    def clamp_value(self, value, bound_1, bound_2):
        min_val = min(bound_1, bound_2)
        max_val = max(bound_1, bound_2)
        return max(min_val, min(value, max_val))

    def set_servos(self, head_val, ear_l_val, ear_r_val, wait_time=0.33, release=True):
        # enforce safety boundaries
        head_val = self.clamp_value(head_val, SERVO_HEAD_RIGHT, SERVO_HEAD_LEFT)       
        ear_l_val = self.clamp_value(ear_l_val, SERVO_EAR_LEFT_LOW, SERVO_EAR_LEFT_HIGH) 
        ear_r_val = self.clamp_value(ear_r_val, SERVO_EAR_RIGHT_HIGH, SERVO_EAR_RIGHT_LOW) 

        self.pwm_head.ChangeDutyCycle(head_val)
        self.pwm_ear_l.ChangeDutyCycle(ear_l_val)
        self.pwm_ear_r.ChangeDutyCycle(ear_r_val)

        time.sleep(wait_time)

        if release:
            self.pwm_head.ChangeDutyCycle(0)
            self.pwm_ear_l.ChangeDutyCycle(0)
            self.pwm_ear_r.ChangeDutyCycle(0)

    def do_twitch(self):
        self.get_logger().info("executing twitch sequence.")
        
        self.set_servos(SERVO_HEAD_MIDDLE, SERVO_EAR_LEFT_HIGH, SERVO_EAR_RIGHT_HIGH, wait_time=0.4, release=False)
        self.set_servos(SERVO_HEAD_MIDDLE, SERVO_EAR_LEFT_LOW, SERVO_EAR_RIGHT_LOW, wait_time=0.4, release=False)
        self.set_servos(SERVO_HEAD_MIDDLE, SERVO_EAR_LEFT_HIGH, SERVO_EAR_RIGHT_HIGH, wait_time=1.0, release=True)

    def move_head(self, position, ear_pos):
        self.get_logger().info(f"moving hardware -> head: {position}, ears: {ear_pos}")

        if position == Head.HEAD_LEFT:
            head_val = SERVO_HEAD_LEFT
        elif position == Head.HEAD_RIGHT:
            head_val = SERVO_HEAD_RIGHT
        else:
            head_val = SERVO_HEAD_MIDDLE

        if ear_pos == Head.EAR_HIGH:
            ear_l_val = SERVO_EAR_LEFT_HIGH
            ear_r_val = SERVO_EAR_RIGHT_HIGH
        elif ear_pos == Head.EAR_MIDDLE:
            ear_l_val = SERVO_EAR_LEFT_MIDDLE
            ear_r_val = SERVO_EAR_RIGHT_MIDDLE
        else:
            ear_l_val = SERVO_EAR_LEFT_LOW
            ear_r_val = SERVO_EAR_RIGHT_LOW

        self.set_servos(head_val, ear_l_val, ear_r_val, release=True)

    def head_callback(self, msg):
        now = time.time()
        
        if msg.twitch:
            self.do_twitch()
            self.last_command_time = time.time()
            # twitch is a special case
            return

        if now - self.last_command_time < self.cooldown_duration:
            self.get_logger().debug("head command dropped.")
            return

        self.last_command_time = now
        self.move_head(msg.position, msg.ear_pos)

    def destroy_node(self):
        # clean up hardware states
        self.pwm_head.stop()
        self.pwm_ear_l.stop()
        self.pwm_ear_r.stop()
        GPIO.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HeadControllerNode()
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

