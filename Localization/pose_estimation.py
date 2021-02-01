"""
Keeps track of drone position with respect to the starting point.
"""
import numpy as np


class Pose:

    def __init__(self):
        self.poses = [np.array([0.0, 0.0, 0.0, 0.0])]
        
    def move(self, position):
        """
        dist: [x-axis, y-axis, z-axis, yaw]
        Appends new position to poses, as [x, y, z, yaw]
        """
        if np.abs(position[3]) > 360:
            times = int(position[3]/360)
            position[3] -= times*360 
        self.poses.append(position)
                
    def go_back_same_path(self):
        """
        Returns order of commands to execute to go back to origin y the same route.
        """
        print(self.poses)
        dists = [[0.0, 180, False]]
        len_poses = len(self.poses)
        for i in range(len_poses):
            vector_dist = np.array(self.poses[len_poses-(i+1)]) - np.array(self.poses[len_poses-i])
            if vector_dist[2] != 0:
                dists.append([vector_dist[2], 0.0, True])
            else:
                tetha = vector_dist[3]*np.pi/180
                dist = vector_dist[0] / np.cos(tetha)
                dists.append([np.abs(dist), 0.0, False])
                if tetha != 0:
                    dists.append([0.0, vector_dist[3], False])
        self.poses = [self.poses[-1]]
        return np.array(dists)
        
    def get_current_pose(self):
        """
        Returns current position with respect to origin, in metres
        """
        return self.poses[-1]
        
