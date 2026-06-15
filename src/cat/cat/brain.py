import sys
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

from dataclasses import dataclass

from cat_msgs.msg import Meow, Speech, Head, VisionObject, VisionObjectArray

from irobot_create_msgs.msg import KidnapStatus
from irobot_create_msgs.action import Dock, Undock, NavigateToPosition

import random
from enum import Enum

FREQUENCY_RATE = 30
PERIOD = 1.0 / FREQUENCY_RATE
BEHAVIOR_RATE = 2
NODE_NAME = "brain"

# topics
TOPIC_IN_VISION = "/vision_data"
TOPIC_IN_SPEECH = "/speech_data"
TOPIC_IN_ODOMETRY = "/odom"
TOPIC_IN_TF = "/tf"
TOPIC_IN_KIDNAP = "/kidnap_status"

TOPIC_OUT_MEOW = "/brain_meow"
TOPIC_OUT_HEAD = "/head_data"
TOPIC_OUT_CMD_VEL = "/cmd_vel"

ACTION_DOCK = "/dock"
ACTION_UNDOCK = "/undock"

IDLE_SLEEP_TIME = 20.0

"""
state bound variables structs
"""
@dataclass
class TiredData:
    is_navigating: bool = False
    is_docking: bool = False
    has_twitched_ears: bool = False

@dataclass
class SleepData:
    is_undocking: bool = False

@dataclass
class AttendData:
    target_identifier: int = VisionObject.PERSON
    has_target: bool = False
    attend_start_time: float = 0.0
    is_detected: bool = False
    has_meowed: bool = False

@dataclass
class HappyData:
    is_docking: bool = False
    has_meowed: bool = False
    has_twitched_ears: bool = False

@dataclass
class AngryData:
    is_docking: bool = False
    motion_started: bool = False
    motion_done: bool = False
    linear_x: float = 0.0
    angular_z: float = 0.0
    motion_duration_sec: float = 0.0
    has_meowed: bool = False

@dataclass
class CuriousData:
    frames_lost: int = 0
    track_id: int = 0
    last_error_x: float = 0.0
    last_linear_x: float = 0.0
    last_angular_z: float = 0.0
    has_meowed: bool = False
    has_twitched_ears: bool = False

@dataclass
class FollowData:
    frames_lost: int = 0 # because this node is faster than vision.py :^)
    track_id: int = 0
    last_error_x: float = 0.0
    last_linear_x: float = 0.0
    last_angular_z: float = 0.0
    has_meowed: bool = False

class DanceData:
    has_meowed: bool = False

# state machine enum
class CatState(str, Enum):
    TIRED = "TIRED"
    SLEEP = "SLEEP"
    IDLE = "IDLE"
    ATTEND = "ATTEND"
    FOLLOW = "FOLLOW"
    HAPPY = "HAPPY"
    ANGRY = "ANGRY"
    CURIOUS = "CURIOUS"
    DANCE = "DANCE"


class StateMachine:
    PRIORITY = {
        CatState.SLEEP: 0,
        CatState.TIRED: 0,
        CatState.IDLE: 1,
        CatState.ATTEND: 2,
        CatState.FOLLOW: 3,
        CatState.CURIOUS: 4,
        CatState.HAPPY: 5,
        CatState.ANGRY: 6,
        CatState.DANCE: 7,
    }

    DEFAULT_STOWED = {
        CatState.TIRED: TiredData(),
        CatState.SLEEP: SleepData(),
        CatState.IDLE: None,
        CatState.ATTEND: AttendData(),
        CatState.HAPPY: HappyData(),
        CatState.ANGRY: AngryData(),
        CatState.CURIOUS: CuriousData(),
        CatState.FOLLOW: FollowData(),
        CatState.DANCE: DanceData(),
    }

    def __init__(self, now: Time):
        self._set_state(CatState.SLEEP, now)

    def get_state_duration(self, now: Time):
        return now - self.time

    def _set_state(self, state: CatState, time: Time, stowed=None):
        self.state = state
        self.time = time
        if stowed is not None:
            self.stowed = stowed
        else:
            dataclass_template = self.DEFAULT_STOWED.get(state)
            self.stowed = dataclass_template.__class__() if dataclass_template else None
            
    def get_state(self) -> CatState:
        return self.state

    def get_things(self, now: Time) -> tuple[CatState, Time, dict]:
        return (self.state, self.get_state_duration(now), self.stowed)


class BrainNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        
        self.state_machine = StateMachine(self.get_clock().now())

        self.state_handlers = {
            CatState.TIRED: self.execute_tired_state,
            CatState.SLEEP: self.execute_sleep_state,
            CatState.IDLE: self.execute_idle_state,
            CatState.HAPPY: self.execute_happy_state,
            CatState.ANGRY: self.execute_angry_state,
            CatState.CURIOUS: self.execute_curious_state,
            CatState.ATTEND: self.execute_attend_state,
            CatState.FOLLOW: self.execute_follow_state,
            CatState.DANCE: self.execute_dance_state,
        }

        self.timer = self.create_timer(1.0/FREQUENCY_RATE, self.timer_callback)

        # inputs
        self.sub_vision = self.create_subscription(VisionObjectArray, TOPIC_IN_VISION, self.vision_cb, 10)
        self.sub_speech = self.create_subscription(Speech, TOPIC_IN_SPEECH, self.speech_cb, 10)

        self.sub_odometry = self.create_subscription(Odometry, TOPIC_IN_ODOMETRY, self.odom_cb, qos_profile_sensor_data)
        self.sub_kidnap = self.create_subscription(KidnapStatus, TOPIC_IN_KIDNAP, self.kidnap_cb, qos_profile_sensor_data)

        # outputs
        self.publish_meow = self.create_publisher(Meow, TOPIC_OUT_MEOW, 10)
        self.publish_head = self.create_publisher(Head, TOPIC_OUT_HEAD, 10)
        self.publish_cmd_vel = self.create_publisher(Twist, TOPIC_OUT_CMD_VEL, 10)

        # actions
        self.dock_client = ActionClient(self, Dock, "/dock")
        self.undock_client = ActionClient(self, Undock, "/undock")
        self.nav_client = ActionClient(self, NavigateToPosition, "/navigate_to_position")

        self.dock_goal_handle = None
        self.undock_goal_handle = None
        self.nav_goal_handle = None

        self.has_undocked = False
        self.has_docked = False
        self.is_kidnapped = False

        self.reset_stored_messages()

    """
    time callback
    """
    def timer_callback(self):
        now = self.get_clock().now()
        (state, state_duration, stowed) = self.state_machine.get_things(now)

        handler = self.state_handlers.get(state)
        if handler:
            handler(state_duration, stowed)
        else:
            self.get_logger().error(f"unhandled state encountered: {state}")

        self.reset_stored_messages()

    """
    state functions
    """
    def execute_tired_state(self, time_duration, stowed):
        for msg in self.stored_speech:
            if msg.speech == Speech.WAKE_UP:
                self.exit_tired_state(CatState.IDLE)
                return
            elif msg.speech == Speech.FOLLOW:
                self.exit_tired_state(CatState.FOLLOW)
                return
            elif msg.speech == Speech.GO_AWAY:
                self.exit_tired_state(CatState.ANGRY)
                return
            elif msg.speech == Speech.PRAISE:
                self.exit_tired_state(CatState.HAPPY)
                return
            elif msg.speech == Speech.PLAY:
                self.exit_tired_state(CatState.CURIOUS)
                return
            elif msg.speech == Speech.DANCE:
                self.exit_tired_state(CatState.DANCE)
                return
                
        # phase 1: navigate to (0, 0)
        if not stowed.is_navigating and not stowed.is_docking:
            self.stop_motion(repeat=5)
            
            goal_msg = NavigateToPosition.Goal()
            goal_msg.achieve_goal_heading = False
            goal_msg.max_translation_speed = 0.3
            
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = 0.0
            pose.pose.position.y = 0.0
            goal_msg.goal_pose = pose
            
            def feedback_cb(feedback_msg):
                feedback = feedback_msg.feedback
                # only check distance if we are actively driving (state 2) or close to the end
                if feedback.remaining_travel_distance <= 1.5:
                    if not stowed.is_docking:
                        self.get_logger().info("docking now.")
                        if self.nav_goal_handle is not None:
                            self.nav_goal_handle.cancel_goal_async()
                        self._start_docking(stowed)

            f = self.nav_client.send_goal_async(goal_msg, feedback_callback=feedback_cb)
            
            def goal_cb(future):
                goal_handle = future.result()
                if not goal_handle.accepted:
                    stowed.is_navigating = False
                    return
                    
                self.nav_goal_handle = goal_handle
                if self.state_machine.get_state() != CatState.TIRED:
                    goal_handle.cancel_goal_async()
                    self.nav_goal_handle = None
                    return

                res_f = goal_handle.get_result_async()
                
                def result_cb(res_future):
                    if self.state_machine.get_state() != CatState.TIRED:
                        return
                    
                    # if nav completes normally (or we were already at 0,0), trigger dock
                    if not stowed.is_docking:
                        self.get_logger().info("navigation finished. starting dock.")
                        self._start_docking(stowed)
                        
                    self.nav_goal_handle = None

                res_f.add_done_callback(result_cb)
                
            f.add_done_callback(goal_cb)
            stowed.is_navigating = True
            
        # transition out once fully docked
        elif self.has_docked:
            self.set_state(CatState.SLEEP)          
            self.has_docked = False

    def _start_docking(self, stowed):
        if stowed.is_docking:
            return
            
        stowed.is_docking = True
        stowed.is_navigating = False
        self.twitch_ears_once(stowed) # twitch ears because why not
        
        goal_msg = Dock.Goal()
        f = self.dock_client.send_goal_async(goal_msg)
        
        def goal_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                stowed.is_docking = False
                return
            
            self.dock_goal_handle = goal_handle
            if self.state_machine.get_state() != CatState.TIRED:
                goal_handle.cancel_goal_async()
                self.dock_goal_handle = None
                return

            res_f = goal_handle.get_result_async()
            
            def result_cb(res_future):
                if self.state_machine.get_state() != CatState.TIRED:
                    return

                if res_future.result().status == 4: # 4 is succeeded
                    self.has_docked = True
                else:
                    stowed.is_docking = False
                self.dock_goal_handle = None
                    
            res_f.add_done_callback(result_cb)
            
        f.add_done_callback(goal_cb)

    def execute_sleep_state(self, time_duration, stowed):
        if not stowed.is_undocking:
            for msg in self.stored_speech:
                if msg.speech == Speech.WAKE_UP:
                    goal_msg = Undock.Goal()
                    f = self.undock_client.send_goal_async(goal_msg)
                    
                    def goal_cb(future):
                        goal_handle = future.result()
                        if goal_handle.accepted:
                            self.undock_goal_handle = goal_handle
                            res_f = goal_handle.get_result_async()
                            
                            def result_cb(res_future):
                                if res_future.result().status == 4:
                                    self.has_undocked = True
                                    self.undock_goal_handle = None
                                else:
                                    stowed.is_undocking = False
                                    
                            res_f.add_done_callback(result_cb)
                        else:
                            stowed.is_undocking = False
                    f.add_done_callback(goal_cb)
                    stowed.is_undocking = True
                    break
                    
        elif self.has_undocked:
            self.set_state(CatState.IDLE)          
            self.has_undocked = False

    def execute_idle_state(self, time_duration, stowed):
        tired_bool = False
        for msg in self.stored_speech:
            if msg.speech == Speech.REST:
                self.set_state(CatState.TIRED)
                break
            elif msg.speech == Speech.FOLLOW:
                self.set_state(CatState.FOLLOW)
                return
            elif msg.speech == Speech.GO_AWAY:
                self.set_state(CatState.ANGRY)
                return
            elif msg.speech == Speech.PRAISE:
                self.set_state(CatState.HAPPY)
                return
            elif msg.speech == Speech.PLAY:
                self.set_state(CatState.CURIOUS)
                return 
            elif msg.speech == Speech.DANCE:
                self.set_state(CatState.DANCE)
                return
            
        if time_duration > Duration(seconds=IDLE_SLEEP_TIME) or tired_bool:
            self.set_state(CatState.TIRED)
            return 

    def execute_happy_state(self, time_duration, stowed):
        if not stowed.has_meowed:
            msg = Meow()
            msg.sound_id = Meow.HAPPY
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publish_meow.publish(msg)
            self.get_logger().info("happy meow!")
            stowed.has_meowed = True
        
        if not stowed.has_twitched_ears:
            self.move_head(random.choice([Head.HEAD_LEFT, Head.HEAD_RIGHT]), Head.EAR_MIDDLE)
            self.twitch_ears_once(stowed)
        
        if time_duration > Duration(seconds=3):
            self.get_logger().info("happy moment over, returning to idle.")
            self.set_state(CatState.IDLE)

    def execute_angry_state(self, time_duration, stowed):
        if not stowed.has_meowed:
            msg = Meow()
            msg.sound_id = Meow.ANGRY
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publish_meow.publish(msg)
            self.get_logger().info("angry meow!")
            msg = Meow()
            msg.sound_id = Meow.HISS
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publish_meow.publish(msg)
            self.get_logger().info("angry hiss!")
            stowed.has_meowed = True

        self.move_head(Head.HEAD_MIDDLE, Head.EAR_HIGH)
        if not stowed.motion_started:
            angry_actions = [
                (0.18, 1.4, 0.8),    # forward-left
                (0.18, -1.4, 0.8),   # forward-right
                (-0.18, 1.4, 1.1),   # backward-left
                (-0.18, -1.4, 1.1),  # backward-right
            ]

            stowed.linear_x, stowed.angular_z, stowed.motion_duration_sec = random.choice(angry_actions)
            stowed.motion_started = True

        if time_duration.nanoseconds / 1e9 <= stowed.motion_duration_sec:
            cmd = Twist()
            cmd.linear.x = stowed.linear_x
            cmd.angular.z = stowed.angular_z
            self.publish_cmd_vel.publish(cmd)
            return

        if not stowed.motion_done:
            self.publish_cmd_vel.publish(Twist())
            stowed.motion_done = True

        if time_duration > Duration(seconds=3):
            self.get_logger().info("anger cooling down, returning to idle.")
            self.set_state(CatState.IDLE)

    def execute_curious_state(self, time_duration, stowed):
        """
        curious means: look for a bottle and follow only that bottle.
        this state does not follow people or other objects.
        """
        elapsed_sec = time_duration.nanoseconds / 1e9

        if not stowed.has_twitched_ears:
            self.move_head(random.choice([Head.HEAD_LEFT, Head.HEAD_RIGHT]), Head.EAR_MIDDLE)
            self.twitch_ears_once(stowed)

        if not stowed.has_meowed:
            msg = Meow()
            msg.sound_id = Meow.CURIOUS
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publish_meow.publish(msg)
            self.get_logger().info("curious meow! searching for bottle.")
            stowed.has_meowed = True

        # 1. voice commands take absolute priority
        for msg in self.stored_speech:
            if msg.speech == Speech.STOP:
                self.stop_motion(repeat=3)
                stowed.track_id = 0
                stowed.frames_lost = 0
                stowed.last_error_x = 0.0
                stowed.last_linear_x = 0.0
                stowed.last_angular_z = 0.0
                self.set_state(CatState.IDLE)
                return
            elif msg.speech == Speech.REST:
                self.set_state(CatState.TIRED)
                return
            elif msg.speech == Speech.GO_AWAY:
                self.set_state(CatState.ANGRY)
                return
            elif msg.speech == Speech.PRAISE:
                self.set_state(CatState.HAPPY)
                return
            elif msg.speech == Speech.DANCE:
                self.set_state(CatState.DANCE)
                return
            
        cmd = Twist()
        target_obj = None
        best_bottle = None

        for vision_array in self.stored_vision:
            for obj in vision_array.objects:
                if obj.identifier != VisionObject.BOTTLE:
                    continue
                if best_bottle is None or obj.confidence > best_bottle.confidence:
                    best_bottle = obj

        # 2. acquire a target if we don't have one active
        if stowed.track_id == 0 and best_bottle is not None:
            stowed.track_id = best_bottle.track_id
            stowed.frames_lost = 0
            self.get_logger().info(f"locked onto bottle track id: {stowed.track_id}")

        # target not found, rotate
        if stowed.track_id == 0:
            cmd.angular.z = 0.25
            cmd.linear.x = 0.0
            self.publish_cmd_vel.publish(cmd)
            return
            
        # 3. locate active target in current frame
        if stowed.track_id != 0:
            for vision_array in self.stored_vision:
                for obj in vision_array.objects:
                    if obj.track_id == stowed.track_id:
                        target_obj = obj
                        break
                if target_obj:
                    break
        # detect bottle in current image if the bottle is lost
        if target_obj is None and best_bottle is not None:
            target_obj = best_bottle
            stowed.track_id = best_bottle.track_id
            stowed.frames_lost = 0
            self.get_logger().info(f"relocked onto bottle track id: {stowed.track_id}")

        if target_obj:
            stowed.frames_lost = 0 
            stowed.last_error_x = float(target_obj.error_x)
            
            error_sign = -stowed.last_error_x
            
            if abs(stowed.last_error_x) > 0.12:
                kp_angular = 0.6
                max_angular = 2.65
                cmd.angular.z = error_sign * kp_angular
                cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))
            else:
                cmd.angular.z = 0.0
                
            if abs(stowed.last_error_x) < 0.15:
                cmd.linear.x = 1.00
            elif abs(stowed.last_error_x) < 0.45:
                cmd.linear.x = 0.50
            else:
                cmd.linear.x = 0.0 

            stowed.last_linear_x = cmd.linear.x
            stowed.last_angular_z = cmd.angular.z
            
            self.get_logger().info(f"tracking | error_x: {stowed.last_error_x:.2f} | v_x: {cmd.linear.x:.2f} | w_z: {cmd.angular.z:.3f}")
            self.publish_cmd_vel.publish(cmd)
            
        elif stowed.track_id != 0:
            if not self.stored_vision:
                cmd.linear.x = stowed.last_linear_x
                cmd.angular.z = stowed.last_angular_z
                self.publish_cmd_vel.publish(cmd)
                return

            stowed.frames_lost += 1
            
            if stowed.frames_lost <= 15:
                spin_direction = -0.30 if stowed.last_error_x > 0 else 0.30
                cmd.angular.z = spin_direction
                cmd.linear.x = 0.0 
                
                self.get_logger().info(f"lost target! searching frame {stowed.frames_lost}/15", throttle_duration_sec=0.2)
                self.publish_cmd_vel.publish(cmd)
            else:
                self.get_logger().info("target completely lost. resetting lock to find next person.")
                self.publish_cmd_vel.publish(Twist()) 
                stowed.track_id = 0
                stowed.frames_lost = 0
                stowed.last_error_x = 0.0
                stowed.last_linear_x = 0.0
                stowed.last_angular_z = 0.0
                self.set_state(CatState.IDLE)
        
    def execute_follow_state(self, time_duration, stowed):
        if not stowed.has_meowed:
            msg = Meow()
            msg.sound_id = Meow.MEOW
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publish_meow.publish(msg)
            self.get_logger().info("follow meow!")
            stowed.has_meowed = True

        self.move_head(Head.HEAD_MIDDLE, Head.EAR_HIGH)
        # 1. voice commands take absolute priority
        for msg in self.stored_speech:
            if msg.speech == Speech.STOP:
                self.stop_motion(repeat=3)
                stowed.track_id = 0
                stowed.frames_lost = 0
                stowed.last_error_x = 0.0
                stowed.last_linear_x = 0.0
                stowed.last_angular_z = 0.0
                self.set_state(CatState.IDLE)
                return
            elif msg.speech == Speech.REST:
                self.set_state(CatState.TIRED)
                return
            elif msg.speech == Speech.GO_AWAY:
                self.set_state(CatState.ANGRY)
                return
            elif msg.speech == Speech.PRAISE:
                self.set_state(CatState.HAPPY)
                return
            elif msg.speech == Speech.DANCE:
                self.set_state(CatState.DANCE)
                return
            
        cmd = Twist()
        target_obj = None

        # 2. acquire a target if we don't have one active
        if stowed.track_id == 0:
            longest_duration = -1.0
            best_track_id = 0
            
            for vision_array in self.stored_vision:
                for obj in vision_array.objects:
                    if obj.identifier == VisionObject.PERSON:
                        if obj.duration > longest_duration:
                            longest_duration = obj.duration
                            best_track_id = obj.track_id
            
            if best_track_id != 0:
                stowed.track_id = best_track_id
                stowed.frames_lost = 0
                self.get_logger().info(f"locked onto track id: {stowed.track_id}")

        # 3. locate active target in current frame
        if stowed.track_id != 0:
            for vision_array in self.stored_vision:
                for obj in vision_array.objects:
                    if obj.track_id == stowed.track_id:
                        target_obj = obj
                        break
                if target_obj:
                    break

        # 4. control loop mechanics
        if target_obj:
            stowed.frames_lost = 0 
            stowed.last_error_x = float(target_obj.error_x)
            
            error_sign = -stowed.last_error_x
            
            if abs(stowed.last_error_x) > 0.12:
                kp_angular = 0.6
                max_angular = 0.35
                cmd.angular.z = error_sign * kp_angular
                cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))
            else:
                cmd.angular.z = 0.0
                
            if abs(stowed.last_error_x) < 0.15:
                cmd.linear.x = 0.18
            elif abs(stowed.last_error_x) < 0.45:
                cmd.linear.x = 0.10
            else:
                cmd.linear.x = 0.0 

            stowed.last_linear_x = cmd.linear.x
            stowed.last_angular_z = cmd.angular.z
            
            self.get_logger().info(f"tracking | error_x: {stowed.last_error_x:.2f} | v_x: {cmd.linear.x:.2f} | w_z: {cmd.angular.z:.3f}")
            self.publish_cmd_vel.publish(cmd)
            
        elif stowed.track_id != 0:
            if not self.stored_vision:
                cmd.linear.x = stowed.last_linear_x
                cmd.angular.z = stowed.last_angular_z
                self.publish_cmd_vel.publish(cmd)
                return

            stowed.frames_lost += 1
            
            if stowed.frames_lost <= 15:
                spin_direction = -0.30 if stowed.last_error_x > 0 else 0.30
                cmd.angular.z = spin_direction
                cmd.linear.x = 0.0 
                
                self.get_logger().info(f"lost target! searching frame {stowed.frames_lost}/15", throttle_duration_sec=0.2)
                self.publish_cmd_vel.publish(cmd)
            else:
                self.get_logger().info("target completely lost. resetting lock to find next person.")
                self.publish_cmd_vel.publish(Twist()) 
                stowed.track_id = 0
                stowed.frames_lost = 0
                stowed.last_error_x = 0.0
                stowed.last_linear_x = 0.0
                stowed.last_angular_z = 0.0
                self.set_state(CatState.IDLE)

    def execute_dance_state(self, time_duration, stowed):
        for msg in self.stored_speech:
            if msg.speech == Speech.FOLLOW:
                self.publish_cmd_vel.publish(Twist())
                self.set_state(CatState.FOLLOW)
                return
            elif msg.speech == Speech.REST:
                self.publish_cmd_vel.publish(Twist())
                self.set_state(CatState.TIRED)
                return
            elif msg.speech == Speech.STOP:
                self.publish_cmd_vel.publish(Twist())
                self.set_state(CatState.IDLE)
                return
            elif msg.speech == Speech.GO_AWAY:
                self.set_state(CatState.ANGRY)
                return
            elif msg.speech == Speech.PRAISE:
                self.set_state(CatState.HAPPY)
                return

        if not stowed.has_meowed:
            msg = Meow()
            msg.sound_id = Meow.HAPPY
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publish_meow.publish(msg)
            self.get_logger().info("happy meow! let's dance!")
            stowed.has_meowed = True

        # sequence definition: (linear_x, angular_z, duration_sec, head_pos)
        dance_sequence = [
            # 1. turn left, center, right, (and back to center)
            (0.0, 0.8, 0.6, Head.HEAD_LEFT),
            (0.0, -0.8, 0.6, Head.HEAD_MIDDLE),
            (0.0, -0.8, 0.6, Head.HEAD_RIGHT),
            (0.0, 0.8, 0.6, Head.HEAD_MIDDLE),

            # 2. turn left, center, right, (and back to center)
            (0.0, 0.8, 0.6, Head.HEAD_LEFT),
            (0.0, -0.8, 0.6, Head.HEAD_MIDDLE),
            (0.0, -0.8, 0.6, Head.HEAD_RIGHT),
            (0.0, 0.8, 0.6, Head.HEAD_MIDDLE),

            # 3. move forward, move backward
            (0.2, 0.0, 1.0, Head.HEAD_MIDDLE),
            (-0.2, 0.0, 1.0, Head.HEAD_MIDDLE),

            # 4. final turn left, center, right (and back to center)
            (0.0, 0.8, 0.6, Head.HEAD_LEFT),
            (0.0, -0.8, 0.6, Head.HEAD_MIDDLE),
            (0.0, -0.8, 0.6, Head.HEAD_RIGHT),
            (0.0, 0.8, 0.6, Head.HEAD_MIDDLE),
        ]

        elapsed_sec = time_duration.nanoseconds / 1e9
        cumulative_time = 0.0
        current_move = None

        for move in dance_sequence:
            lin_x, ang_z, dur, head_pos = move
            cumulative_time += dur
            if elapsed_sec < cumulative_time:
                current_move = move
                break

        if current_move is None:
            # sequence complete, switch to happy state
            self.publish_cmd_vel.publish(Twist())
            self.set_state(CatState.HAPPY)
            return

        linear_x, angular_z, _, head_pos = current_move

        self.move_head(head_pos, Head.EAR_HIGH)

        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.publish_cmd_vel.publish(cmd)
    
    def execute_attend_state(self, time_duration, stowed):
        for msg in self.stored_speech:
            if msg.speech == Speech.FOLLOW:
                self.publish_cmd_vel.publish(Twist())
                self.set_state(CatState.FOLLOW)
                return
            elif msg.speech == Speech.REST:
                self.publish_cmd_vel.publish(Twist())
                self.set_state(CatState.TIRED)
                return
            elif msg.speech == Speech.STOP:
                self.publish_cmd_vel.publish(Twist())
                stowed.has_target = False
                stowed.is_detected = False
                self.set_state(CatState.IDLE)
                return
            elif msg.speech == Speech.GO_AWAY:
                self.set_state(CatState.ANGRY)
                return
            elif msg.speech == Speech.PRAISE:
                self.set_state(CatState.HAPPY)
                return
            elif msg.speech == Speech.DANCE:
                self.set_state(CatState.DANCE)
                return

        target_in_view = False
        for vision_array in self.stored_vision:
            for obj in vision_array.objects:
                if obj.identifier == stowed.target_identifier:
                    target_in_view = True
                    break
            if target_in_view:
                break

        if target_in_view:
            stowed.has_target = True
            stowed.is_detected = True
            self.publish_cmd_vel.publish(Twist())
            self.set_state(CatState.FOLLOW)
            return

        stowed.has_target = False
        elapsed_sec = time_duration.nanoseconds / 1e9
        search_angular_z = 0.4
        full_rotation_sec = 6.283185307179586 / abs(search_angular_z)

        if elapsed_sec <= full_rotation_sec:
            cmd = Twist()
            cmd.angular.z = search_angular_z
            self.publish_cmd_vel.publish(cmd)
            return

        self.publish_cmd_vel.publish(Twist())

        if not stowed.has_meowed:
            meow_msg = Meow()
            meow_msg.header.stamp = self.get_clock().now().to_msg()
            meow_msg.sound_id = Meow.MEOW
            self.publish_meow.publish(meow_msg)
            stowed.has_meowed = True

        if elapsed_sec > full_rotation_sec + 10.0:
            self.get_logger().info("no person found after search, returning to idle.")
            self.publish_cmd_vel.publish(Twist())
            self.set_state(CatState.IDLE)
    
    """
    topic callbacks
    """
    def vision_cb(self, msg: VisionObjectArray):
        self.stored_vision.append(msg)

    def speech_cb(self, msg: Speech):
        self.stored_speech.append(msg)
        
    def odom_cb(self, msg: Odometry):
        self.stored_odom.append(msg)

    def kidnap_cb(self, msg: KidnapStatus):
        self.is_kidnapped = msg.is_kidnapped

    """
    helper functions
    """
    def stop_motion(self, repeat=1):
        for _ in range(repeat):
            self.publish_cmd_vel.publish(Twist())
            time.sleep(0.02)

    def cancel_tired_actions(self):
        if self.dock_goal_handle is not None:
            self.dock_goal_handle.cancel_goal_async()
            self.dock_goal_handle = None
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None

    def exit_tired_state(self, next_state: CatState):
        self.cancel_tired_actions()
        self.has_docked = False
        self.stop_motion(repeat=5)
        self.set_state(next_state)

    def cancel_active_actions(self):
        for goal_handle in (self.dock_goal_handle, self.undock_goal_handle, self.nav_goal_handle):
            if goal_handle is None:
                continue
            future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        self.dock_goal_handle = None
        self.undock_goal_handle = None
        self.nav_goal_handle = None

    def set_state(self, state: CatState, stowed=None):
        self.get_logger().info(f"new state: {state} from {self.state_machine.get_state()}")
        if state in (CatState.TIRED, CatState.IDLE, CatState.SLEEP):
            self.stop_motion()
        self.state_machine._set_state(state, self.get_clock().now(), stowed)

    def reset_stored_messages(self):
        self.stored_vision = []
        self.stored_speech = []
        self.stored_odom = []
        self.stored_kidnaps = []

    def move_head(self, head_position: int, ear_position: int):
        msg = Head()
        msg.position = head_position
        msg.ear_pos = ear_position
        msg.twitch = False

        self.publish_head.publish(msg)

    def twitch_ears(self):
        msg = Head()
        msg.position = Head.HEAD_MIDDLE 
        msg.ear_pos = Head.EAR_HIGH
        msg.twitch = True
        self.publish_head.publish(msg)

    def twitch_ears_once(self, stowed):
        if getattr(stowed, "has_twitched_ears", False):
            return
        self.twitch_ears()
        setattr(stowed, "has_twitched_ears", True)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("shutting down...")
    finally:
        if rclpy.ok():
            node.cancel_active_actions()
            node.stop_motion(repeat=5)
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
