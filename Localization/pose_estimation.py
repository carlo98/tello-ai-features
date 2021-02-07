"""
Keeps track of drone position with respect to the starting point.
"""
import numpy as np


class Pose:

    def __init__(self):
        self.poses = [np.array([0.0, 0.0, 0.0, 0.0])]
        self.pose = np.array([0.0, 0.0, 0.0, 0.0])

    def move(self, position):
        """
        Changes current position with respect to origin.
        """
        self.pose = np.array(position)
        self.angle_0_360()

    def angle_0_360(self):
        """
        Keeps the yaw in range 0-360 degrees.
        """
        div = int(self.pose[3] / 360)
        self.pose[3] -= div*360
        if self.pose[3] < 0:
            self.pose[3] = 360 + self.pose[3]

    def get_current_pose(self):
        """
        Returns current position with respect to origin.
        """
        return self.pose

    def move_2(self, position):
        """
        dist: [x-axis, y-axis, z-axis, yaw]
        Appends new position to poses, as [x, y, z, yaw]
        """
        if np.abs(position[3]) > 360:
            times = int(position[3]/360)
            position[3] -= times*360 
        max_direction = np.argmax(np.abs(position))
        max_value = position[max_direction]
        position = np.zeros(position.shape)
        position[max_direction] = max_value
        self.poses.append(position)
                
    def go_back_same_path(self):
        """
        Returns order of commands to execute to go back to origin y the same route.
        """
        print(self.poses)
        dists = [[0.0, 180, False]]
        len_poses = len(self.poses)
        for i in range(1, len_poses):
            print(len_poses, len_poses-(i+1), len_poses-i)
            vector_dist = np.array(self.poses[len_poses-(i+1)]) - np.array(self.poses[len_poses-i])
            if vector_dist[2] != 0:
                dists.append([vector_dist[2], 0.0, True])
            else:
                tetha = vector_dist[3]*np.pi/180
                dist = vector_dist[0] / np.cos(tetha)
                dists.append([np.abs(dist), 0.0, False])
                if tetha != 0:
                    dists.append([0.0, vector_dist[3], False])
        # self.poses = [self.poses[-1]]
        return np.array(dists)
        
    def get_current_pose_2(self):
        """
        Returns current position with respect to origin, in metres
        """
        return self.poses[-1]

