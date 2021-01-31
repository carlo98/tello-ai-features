"""
tellotracker:
Allows manual operation of the drone and demo tracking mode.

Requires mplayer to record/save video.

Controls:
- tab to lift off
- WASD to move the drone
- arrow keys to ascend, descend, or yaw quickly
  (zoomed-in widescreen or high FOV 4:3)
@author Leonie Buckley, Saksham Sinha and Jonathan Byrne
@copyright 2018 see license file for details

   IMPORTANT: Only one feature (1, 2) can be activate at any time.
 - 1 to toggle collision avoidance
 - 2 to toggle tracking
 - 3 to toggle reinforcement learning training for collision avoidance (If activated then also collision avoidance will be ON)
 - 4 to toggle go back to starting point
 - x to end/start episode of RL
 - F to save frame as free (collision avoidance)
 - B to save frame as blocked (collision avoidance) 
"""
import time
import datetime
import os
import copy
from djitellopy.tello import Tello
import numpy as np
import av
import cv2
from pynput import keyboard
from Face_Recognition.face_rec_tracker import Tracker
from Collision_Avoidance.collision_avoidance import Agent
from Collision_Avoidance.RL import RL_Agent
from Camera_Calibration.process_image import FrameProc
from Localization.pose_estimation import Pose
from scipy.interpolate import interp1d
import sys
import traceback
import threading


MAX_SPEED_AUTONOMOUS=30
SPEED_HAND = 60
DISTANCE_FAC_REC = 70
AREA_MIN = 4000
AREA_MAX = 8000

def main():
    """ Create a tello controller and show the video feed."""
    tellotrack = TelloCV()
    
    try:
        frame = tellotrack.drone.get_frame_read().frame
        tellotrack.frameproc = FrameProc(frame.shape[0], frame.shape[1])
        tellotrack.tracker.init_video(frame.shape[0], frame.shape[1])
        while True:
            frame = tellotrack.drone.get_frame_read().frame
            image = tellotrack.process_frame(frame)
            show(image)
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        cv2.destroyAllWindows()


def show(frame):
    """show the frame to cv2 window"""
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        exit()


class TelloCV(object):
    """
    TelloTracker builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self):
        self.prev_flight_data = None
        self.record = False
        self.tracking = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.speed = MAX_SPEED_AUTONOMOUS
        self.speed_hand = SPEED_HAND
        if os.path.isdir('Collision_Avoidance/data'):
            if not os.path.isdir('Collision_Avoidance/data/blocked') or not os.path.isdir('Collision_Avoidance/data/free'):
                print("Either 'blocked' folder or 'free' folder or both don't exist, any attempt to save images for NN training will fail!")
        else:
            print("'data' folder doesn't exists, any attempt to save images for NN training will fail!")
        self.avoidance = False
        self.rl_training = False
        self.reward = 0
        self.episode_cont = 1
        self.current_step = 0
        self.old_state = None
        self.current_state = None
        self.training_thread = None
        self.train_rl_sem = threading.Semaphore(1)
        self.episode_start = True
        self.eps_pose = 10
        self.save_frame = False
        self.blocked_free = 0
        self.go_back = False
        self.start_time_command = 0
        self.distance = DISTANCE_FAC_REC
        self.area_min = AREA_MIN
        self.area_max = AREA_MAX
        self.track_cmd = ""
        self.ca_agent = Agent()
        self.rl_agent = RL_Agent(self.ca_agent.model, self.ca_agent.device)
        self.tracker = Tracker()
        self.pose_estimator = Pose()
        self.drone = Tello()
        self.init_drone()
        self.init_controls()

        # Processing frames
        self.video_initialized = False
        self.frameproc = None

    def init_drone(self):
        """Connect, uneable streaming and subscribe to events"""
        self.drone.connect()
        self.drone.streamon()

    def on_press(self, keyname):
        """handler for keyboard listener"""
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.land()
                exit(0)
            if keyname in self.controls:
                key_handler = self.controls[keyname]
                if keyname in ['1', '2', '3', '4', 'x']:
                        key_handler(self.speed)
                else:
                    key_handler(self.speed_hand)
                    if keyname in ['w', 'a', 's', 'd', 'Key.left', 'Key.right', 'Key.up', 'Key.down']:
                        self.start_time_command = time.time()  # Get start time of new command
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""
        self.keydown = False
        keyname = str(keyname).strip('\'')
        print('-' + keyname)
        if keyname in self.controls and keyname in ['w', 'a', 's', 'd', 'Key.left', 'Key.right', 'Key.up', 'Key.down']:
            key_handler = self.controls[keyname]
            key_handler(0)
            time_laps = (time.time()-self.start_time_command)
            if self.track_cmd == 'forward' or self.track_cmd == 'backward':
                curr_vel = self.drone.get_speed_x()
                curr_acc = self.drone.get_acceleration_x()
                self.pose_estimator.move(dist= curr_vel*time_laps + 0.5*curr_acc*(time_laps**2))
            elif self.track_cmd == 'Key.left' or self.track_cmd == 'Key.right':
                self.pose_estimator.move(yaw=self.drone.get_yaw())
            elif self.track_cmd == 'Key.down' or self.track_cmd == 'Key.up':
                curr_vel = self.drone.get_speed_z()
                curr_acc = self.drone.get_acceleration_z()
                self.pose_estimator.move(dist=-curr_vel*time_laps + 0.5*curr_acc*(time_laps**2), vertical=True)

    def init_controls(self):
        """Define keys and add listener"""
        self.controls = {
            'w': lambda speed: self.drone.send_rc_control(0, speed, 0, 0),
            's': lambda speed: self.drone.send_rc_control(0, -speed, 0, 0),
            'a': lambda speed: self.drone.send_rc_control(-speed, 0, 0, 0),
            'd': lambda speed: self.drone.send_rc_control(speed, 0, 0, 0),
            'i': lambda speed: self.drone.flip_forward(),
            'k': lambda speed: self.drone.flip_back(),
            'j': lambda speed: self.drone.flip_left(),
            'l': lambda speed: self.drone.flip_right(),
            # arrow keys for fast turns and altitude adjustments
            'Key.left': lambda speed: self.drone.send_rc_control(0, 0, 0, -speed),
            'Key.right': lambda speed: self.drone.send_rc_control(0, 0, 0, speed),
            'Key.up': lambda speed: self.drone.send_rc_control(0, 0, speed, 0),
            'Key.down': lambda speed: self.drone.send_rc_control(0, 0, -speed, 0),
            'Key.tab': lambda speed: self.drone.takeoff(),
            'Key.backspace': lambda speed: self.drone.land(),
            'b': lambda speed: self.toggle_blocked_free(0),
            'f': lambda speed: self.toggle_blocked_free(1),
            '1': lambda speed: self.toggle_collisionAvoidance(speed),
            '2': lambda speed: self.toggle_tracking(speed),
            '3': lambda speed: self.toggle_rl_training(speed),
            '4': lambda speed: self.toggle_go_back(speed),
            # Reinforcement learning commands
            'x': lambda speed: self.toggle_episode_done(True),
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()
        
    def interpolate_readings(self, raw_readings):
        """
        Predicts next position of target
        """
        readings = []
        readings_index = []
        flag = True # Set to false if last reading has no face
        for i, reading in enumerate(raw_readings):
            if reading[2] != 0:
                readings.append(reading)
                readings_index.append(i)
            elif i == len(raw_readings)-1:
                flag = False

        if len(readings) >= 2:
            readings = np.array(readings)
            fx = interp1d(readings_index, readings[:, 0], fill_value="extrapolate")
            fy = interp1d(readings_index, readings[:, 1], fill_value="extrapolate")
            farea = interp1d(readings_index, readings[:, 2], fill_value="extrapolate")
            return fx(len(raw_readings)), fy(len(raw_readings)), farea(len(raw_readings))
            
        # If only one reading available using it only if it is the most recent one
        if len(readings) == 1 and flag:
            return readings[0][0], readings[0][1], readings[0][2]

        return -1, -1, -1

    def process_frame(self, frame):
        """converts frame to cv2 image and show"""
        
        x = np.array(frame)
        # Get undistorted frame
        x = self.frameproc.undistort_frame(x)
         
        image = cv2.cvtColor(copy.deepcopy(x), cv2.COLOR_RGB2BGR)
        image = self.write_hud(image)
        if self.record:
            self.record_vid(frame)

        cmd = ""
        if self.save_frame:
            if self.blocked_free == 0:
                cv2.imwrite("Collision_Avoidance/data/blocked/"+datetime.datetime.now().strftime(self.date_fmt)+".png", x)
            elif self.blocked_free == 1:
                cv2.imwrite("Collision_Avoidance/data/free/"+datetime.datetime.now().strftime(self.date_fmt)+".png", x)
            self.save_frame = False
            
        ## Start Collision Avoidance code
        if self.avoidance:
            cmd_ca_agent, display_frame = self.ca_agent.track(x)
            if cmd_ca_agent == 1:
                if not self.rl_training or self.rl_training and self.episode_start:
                    self.drone.send_rc_control(0, 0, 0, self.speed)
                    self.track_cmd = "clockwise"
            else:
                if not self.rl_training or self.rl_training and self.episode_start:
                    self.drone.send_rc_control(0, self.speed, 0, 0)
                    self.track_cmd = "forward"
                
            ## Start Reinforcement Learning code
            if self.rl_training and self.episode_start:
                self.current_state = display_frame.get()
                if self.current_state is not None and self.old_state is not None:
                    if self.track_cmd == "forward":  # Reward each forward movement
                        new_reward = 1 / self.rl_agent.max_steps
                        self.reward += new_reward
                    else:
                        new_reward = 0
                    self.train_rl_sem.acquire()
                    self.rl_agent.appendMemory(self.old_state, (lambda action: 0 if self.track_cmd == 'clockwise' else 1)(self.track_cmd), new_reward, self.current_state, 0)
                    self.train_rl_sem.release()
                    if self.current_step >= self.rl_agent.max_steps:
                        self.toggle_episode_done(False)
                    self.current_step += 1
                self.old_state = copy.deepcopy(self.current_state)
            ## End Reinforcement Learning code
                
            image = display_frame
        ## End Collision Avoidance code
        
        ## Start Tracking code
        elif self.tracking:
            readings, display_frame = self.tracker.track(image)
            xoff, yoff, distance_measure = self.interpolate_readings(copy.deepcopy(readings))
            if xoff == -1:
                if self.track_cmd is not "":
                    self.drone.send_rc_control(0, 0, 0, 0)
                    self.track_cmd = ""
            elif xoff < -self.distance:
                cmd = "counter_clockwise"
                self.drone.send_rc_control(0, 0, 0, -self.speed)
            elif xoff > self.distance:
                cmd = "clockwise"
                self.drone.send_rc_control(0, 0, 0, self.speed)
            elif yoff < -self.distance:
                cmd = "down"
                self.drone.send_rc_control(0, 0, -self.speed, 0)
            elif yoff > self.distance:
                cmd = "up"
                self.drone.send_rc_control(0, 0, self.speed, 0)
            elif distance_measure <= self.area_min:
                print("Forward ", distance_measure)
                cmd = "forward"
                self.drone.send_rc_control(0, self.speed, 0, 0)
            elif distance_measure >= self.area_max:
                print("backward ", distance_measure)
                cmd = "backward"
                self.drone.send_rc_control(0, -self.speed, 0, 0)
            else:
                if self.track_cmd is not "":
                    self.drone.send_rc_control(0, 0, 0, 0)
                    self.track_cmd = ""
            
            image = display_frame
        ## End Tracking code
        
        if cmd is not self.track_cmd:
            if cmd is not "":
                print("track command:", cmd)
                self.track_cmd = cmd

        return image

    def write_hud(self, frame):
        """Draw drone info, tracking and record on frame"""
        #stats = self.prev_flight_data.split('|')
        stats = []
        #stats.append("Battery: " + str(self.drone.get_battery()))
        #stats.append("Wifi SNR: " + str(self.drone.query_wifi_signal_noise_ratio()))
        stats.append("Tracking:" + str(self.tracking))
        stats.append("Collision Avoidance NN:" + str(self.avoidance))
        stats.append("RL Training:" + str(self.rl_training))
        stats.append("Go Back:" + str(self.go_back))

        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=30)
        return frame
        
    def toggle_blocked_free(self, block_free):
        self.save_frame = True
        self.blocked_free = block_free

    def toggle_tracking(self, speed):
        """ Handle tracking keypress"""
        if speed == 0:  # handle key up event
            return
        self.tracking = not self.tracking
        self.avoidance = False
        self.rl_training = False
        print("tracking:", self.tracking)
        
    def toggle_collisionAvoidance(self, speed):
        """ Handle avoidance keypress"""
        if speed == 0:  # handle key up event
            return
        self.avoidance = not self.avoidance
        self.tracking = False
        print("avoidance:", self.avoidance)
        
    def toggle_go_back(self, speed):
        """ Handle go back to origin keypress """
        self.go_back = True
        return_path = self.pose_estimator.go_back_same_path()
        print(self.pose_estimator.get_current_pose())
        print(return_path)
        for movement in return_path:
            self.pose_estimator.move(dist=movement[0], yaw=movement[1], vertical = True if movement[2] != 0 else False)
            if movement[1] > self.eps_pose:
                cmd = "clockwise"
                self.drone.rotate_clockwise(int(movement[1]))
            elif movement[1] < -self.eps_pose:
                cmd = "counter_clockwise"
                self.drone.rotate_counter_clockwise(int(movement[1]))
            elif movement[0] < -self.eps_pose and not movement[2]:
                cmd = "backward"
                self.drone.move_back(int(movement[0]))
            elif movement[0] > self.eps_pose and not movement[2]:
                cmd = "forward"
                self.drone.move_forward(int(movement[0]))
            elif movement[0] > self.eps_pose and movement[2]:
                cmd = "up"
                self.drone.move_up(int(movement[0]))
            elif movement[0] < -self.eps_pose and movement[2]:
                cmd = "down"
                self.drone.move_down(int(movement[0]))
            self.track_cmd = cmd

    def toggle_rl_training(self, speed):
        """ Handle reinforcement learning training keypress """
        self.rl_training = not self.rl_training
        self.avoidance = self.rl_training
        self.tracking = False
        print("RL training:", self.rl_training)
        print("avoidance:", self.avoidance)
        
    def toggle_episode_done(self, collision):
        """
        RL episode finished, either max number of steps or collision detected.
        """
        if self.episode_start:
            if self.track_cmd is not "":
                self.drone.send_rc_control(0, 0, 0, 0)
                self.drone.send_rc_control(0, -self.speed, 0, 0)  # Avoid crash
                self.track_cmd = "backward"
                time.sleep(0.5)
                self.drone.send_rc_control(0, 0, 0, 0)
                self.track_cmd = ""
            self.speed = 0
            self.train_rl_sem.acquire()
            if collision:
                print("Collision detected by you, great work!")
                self.reward -= 1
                self.rl_agent.appendMemory(self.old_state, (lambda action: 0 if self.track_cmd == 'clockwise' else 1)(self.track_cmd), -1, self.current_state, 1)
            else:
                self.rl_agent.appendMemory(self.old_state, (lambda action: 0 if self.track_cmd == 'clockwise' else 1)(self.track_cmd), 0, self.current_state, 1)
                print("Episode completed, good Tommy!")
            print("Episode ", self.episode_cont, " reward: ", self.reward)

            self.training_thread = threading.Thread(target=self.rl_agent.update_model, args=(self.ca_agent.model, self.episode_cont))
            self.training_thread.start()
            self.rl_agent.save_model(self.ca_agent.model, self.episode_cont)
            self.train_rl_sem.release()
            self.episode_start = False
        else:
            if self.training_thread is not None:
                self.training_thread.join()
            print("Episode Start")
            self.episode_start = True
            self.speed = MAX_SPEED_AUTONOMOUS
            self.episode_cont += 1
            self.reward = 0
            self.current_step = 0


if __name__ == '__main__':
    main()
