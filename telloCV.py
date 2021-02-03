"""
   IMPORTANT: Only one feature (1, 2) can be activate at any time.
 - 1 to toggle collision avoidance
 - 2 to toggle tracking
 - 3 to toggle reinforcement learning training for collision avoidance (If activated then also collision avoidance will be ON)
 - 4 to toggle go back to starting point
 - 5 to toggle imitation learning
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


MAX_SPEED_AUTONOMOUS=20
SPEED_HAND = 40
DISTANCE_FAC_REC = 70
AREA_MIN = 4000
AREA_MAX = 8000

def main():
    """ Create a tello controller and show the video feed."""
    tellotrack = TelloCV()
    
    try:
        frame = tellotrack.drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tellotrack.frameproc = FrameProc(frame.shape[1], frame.shape[0])
        tellotrack.tracker.init_video(frame.shape[1], frame.shape[0])
        while True:
            frame = tellotrack.drone.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = tellotrack.process_frame(frame)
            show(image)
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        tellotrack.stop_threads = True
        tellotrack.thread_position.join()
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
        self.imit_learning = False
        self.eps_pose = 20
        self.save_frame = False
        self.blocked_free = 0
        self.go_back = False
        self.stop_threads = False
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.curr_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.new_position = False
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
        self.thread_position = threading.Thread(target=self.compute_position)
        self.thread_position.start()

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
                    if keyname in ['w', 'a', 's', 'd', 'Key.left', 'Key.right', 'Key.up', 'Key.down']:
                        self.new_position = True
                    key_handler(self.speed_hand)
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
            self.new_position = False
            tmp = list(self.position)
            tmp.append(self.drone.get_yaw())
            self.pose_estimator.move(np.array(tmp, dtype=np.float32))
            self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.curr_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.track_cmd = ""

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
            '5': lambda speed: self.toggle_imit_learning(speed),
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
        x = cv2.resize(x, (500, 500))
        image = cv2.cvtColor(copy.deepcopy(x), cv2.COLOR_RGB2BGR)
        image = self.write_hud(image)

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
            readings, display_frame = self.tracker.track(x)
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
            
            image = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        ## End Tracking code
        
        ## Start imitation learning
        if self.imit_learning:
            _, display_frame = self.ca_agent.preprocess(x)
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
        ## End imitation learning
        
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
        
    def compute_position(self):
        while not self.stop_threads:
            if not self.new_position:
                continue
            start_time = time.time()
            roll = self.drone.get_roll()*np.pi/180
            pitch = self.drone.get_pitch()*np.pi/180
            yaw = self.drone.get_yaw()*np.pi/180
            Rr = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]], dtype=np.float32)
            Rp = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
            Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
            R = np.dot(np.dot(Rr, Rp), Ry)
            R = R/np.cbrt(np.linalg.det(R))
            curr_acc = np.array([self.drone.get_acceleration_y(), self.drone.get_acceleration_x(), self.drone.get_acceleration_z()], dtype=np.float32)*980.7
            curr_acc[2] -= 980.7  # cm/s^2
            global_acc = np.dot(R, curr_acc)
            time_laps = time.time()-start_time
            self.position += self.curr_vel * time_laps + 0.5 * global_acc * (time_laps ** 2)
            self.curr_vel += global_acc * time_laps
            #print(self.position, self.curr_vel, global_acc)
            
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
            self.pose_estimator.move(movement)
            if movement[1] > self.eps_pose:
                cmd = "clockwise"
                self.drone.rotate_clockwise(int(movement[1]))
            elif movement[1] < -self.eps_pose:
                cmd = "counter_clockwise"
                self.drone.rotate_counter_clockwise(int(movement[1]))
            elif movement[0] < -self.eps_pose and not movement[2]:
                cmd = "backward"
                self.drone.move_back(int(np.abs(movement[0])))
            elif movement[0] > self.eps_pose and not movement[2]:
                cmd = "forward"
                self.drone.move_forward(int((np.abs(movement[0]))))
            elif movement[0] > self.eps_pose and movement[2]:
                cmd = "up"
                self.drone.move_up(int(np.abs(movement[0])))
            elif movement[0] < -self.eps_pose and movement[2]:
                cmd = "down"
                self.drone.move_down(int(np.abs(movement[0])))
            self.track_cmd = cmd

    def toggle_rl_training(self, speed):
        """ Handle reinforcement learning training keypress """
        self.rl_training = not self.rl_training
        self.avoidance = self.rl_training
        self.tracking = False
        print("RL training:", self.rl_training)
        print("avoidance:", self.avoidance)
        
    def toggle_imit_learning(self, speed):
        """ Handle imitation learning keypress """
        self.imit_learning = not self.imit_learning
        self.tracking = False
        print("Imitation learning:", self.imit_learning)
        
    def toggle_episode_done(self, collision):
        """
        RL or Imitation Learning episode finished, either max number of steps or collision detected.
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

            if self.rl_training:  # Not training if imitation learning
                self.training_thread = threading.Thread(target=self.rl_agent.update_model, args=(self.ca_agent.model, self.episode_cont))
                self.training_thread.start()
                self.rl_agent.save_model(self.ca_agent.model, self.episode_cont)
            else:
                self.rl_agent.save_memory()
            self.train_rl_sem.release()
            self.episode_start = False
        else:
            if self.training_thread is not None and self.rl_training:
                self.training_thread.join()
            print("Episode Start")
            self.episode_start = True
            self.speed = MAX_SPEED_AUTONOMOUS
            self.episode_cont += 1
            self.reward = 0
            self.current_step = 0


if __name__ == '__main__':
    main()
