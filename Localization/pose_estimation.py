"""
Keeps track of drone position with respect to the starting point.
"""
import numpy as np


class Pose:

    def __init__(self):
        self.accumulate_yaw = self.yaw = 0.0
        self.poses = [[0.0, 0.0, 0.0, 0.0]]
        
    def move(self, speed=None, yaw=0, vertical=False):
        """
        speed: [x-axis, y-axis, z-axis]
        yaw: 
        Appends new position to poses, as [x, y, z]
        """
        if yaw != 0:
            self.accumulate_yaw += yaw
            self.yaw += yaw * np.pi / 180
        else:
            speed /= 100
            last_pose = self.poses[-1]
            if vertical:
                self.poses.append(np.array([last_pose[0], last_pose[1], last_pose[2] + speed, self.accumulate_yaw * np.pi / 180]))
            else:
                self.accumulate_yaw *= np.pi / 180
                self.poses.append(np.array([last_pose[0] + speed*np.cos(self.yaw), last_pose[1] + speed*np.sin(self.yaw), last_pose[2], self.accumulate_yaw]))
                self.accumulate_yaw = 0.0
                
    def go_back_same_path(self):
        """
        Returns order of commands to execute to go back to origin y the same route.
        """
        speeds = [[0.0, 100, False], [0.0, 80, False]]
        len_poses = len(self.poses)
        for i in range(len_poses):
            if i == len_poses-1:
                break
            vector_speed = np.array(self.poses[len_poses-1-(i+1)]) - np.array(self.poses[len_poses-1-i])
            if vector_speed[2] != 0:
                speeds.append([vector_speed[2]*100, 0.0, True])
            else:
                #tetha = np.arctan(vector_speed[1]/vector_speed[0])
                tetha = vector_speed[3]
                speed = vector_speed[0] / np.cos(tetha)
                speeds.append([-speed*100, 0.0, False])
                if tetha != 0:
                    speeds.append([0.0, tetha*180/np.pi, False])
        self.poses = [self.poses[-1]]
        return np.array(speeds)
        
    def get_current_pose(self):
        """
        Returns current position with respect to origin
        """
        return [self.poses[-1][0]*100, self.poses[-1][1]*100, self.poses[-1][2]*100, self.yaw*180/np.pi]
        
