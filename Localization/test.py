from pose_estimation import Pose
import numpy as np

p = Pose()
p.move(np.array([0.0, 0.0, 0.0, 0.0]))
p.move(np.array([0.0, 0.0, 1.0, 0.0]))
p.move(np.array([0.0, 0.0, 0.0, 30.0]))
speed=np.array([10.0, 0.0, 0.0, 0.0])
path = p.go_back_same_path()
print(path)
print(p.get_current_pose())

for movement in path:
    p.move(dist=movement[0], yaw=movement[1], vertical = True if movement[2] != 0 else False)
    print(movement)
    print(p.get_current_pose())

print(p.get_current_pose())
path = p.go_back_same_path()
print(path)
