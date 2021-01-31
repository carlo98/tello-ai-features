from pose_estimation import Pose

p = Pose()
p.move(speed=2)
p.move(speed=1, vertical=True)
p.move(yaw=30)
p.move(speed=10)
path = p.go_back_same_path()
print(path)
print(p.get_current_pose())

for movement in path:
    p.move(speed=movement[0], yaw=movement[1], vertical = True if movement[2] != 0 else False)
    print(movement)
    print(p.get_current_pose())

print(p.get_current_pose())
path = p.go_back_same_path()
print(path)
