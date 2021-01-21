import time
import cv2
import numpy as np


class SimulatorEnv:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width = 500
        self.height = 500
        self.state = []
        self.maxspeed = 30
        print('initialized')

    def move(self, action):
        (lr, fb, ud) = action
        prev_x, prev_y, prev_w = self.state[-1]
        prev_x = prev_x * self.width
        prev_y = prev_y * self.width
        prev_w = prev_w * self.width
        print('prev', (prev_x, prev_y, prev_w))

        curr_x = prev_x - 0.5*self.maxspeed * lr
        curr_y = prev_y + 0.5*self.maxspeed * ud
        curr_w = prev_w + 0.5*self.maxspeed * fb

        difference = abs(prev_w-curr_w)
        if prev_w > curr_w:
            curr_x = curr_x + 0.5 * difference
            curr_y = curr_y + 0.5 * difference
        elif prev_w < curr_w:
            curr_x = curr_x - 0.5 * difference
            curr_y = curr_y - 0.5 * difference
        print('next', (curr_x, curr_y, curr_w))
        
    def get_action(self, index):
        if index == 0:
            return 0.5, 0, 0
        if index == 1:
            return -0.5, 0, 0
        if index == 2:
            return 0, 0.5, 0
        if index == 3:
            return 0, -0.5, 0
        if index == 4:
            return 0, 0, 0.5
        if index == 5:
            return 0, 0, -0.5
        if index == 6:
            return 1, 0, 0
        if index == 7:
            return -1, 0, 0
        if index == 8:
            return 0, 1, 0
        if index == 9:
            return 0, -1, 0
        if index == 10:
            return 0, 0, 1
        if index == 11:
            return 0, 0, -1

    def step(self, action_index):
        action = self.get_action(action_index)
        
        reward = 0
        prev_x, prev_y, prev_w = self.state[-1]
        prev_x = prev_x * self.width
        prev_y = prev_y * self.width
        prev_w = prev_w * self.width
        cv2.waitKey(1)
        #prev_rem_x = self.width - (prev_x + prev_w)
        prev_rem_x = int(self.width / 2)
        prev_diff_x = abs(prev_x - prev_rem_x)

        #prev_rem_y = self.width - (prev_y + prev_w)
        prev_rem_y = int(self.height / 2)
        prev_diff_y = abs(prev_y-prev_rem_y)

        done = False

        (lr, fb, ud) = action
        curr_x = prev_x - self.maxspeed * lr
        curr_y = prev_y + self.maxspeed * ud
        curr_w = prev_w + 1.5 * self.maxspeed * fb

        difference = abs(prev_w - curr_w)
        if prev_w > curr_w:
            curr_x = curr_x + 0.5 * difference
            curr_y = curr_y + 0.5 * difference
        else:
            curr_x = curr_x - 0.5 * difference
            curr_y = curr_y - 0.5 * difference

        cv2.waitKey(1)

        if curr_x < 0 or curr_x + curr_w > self.width or curr_y < 0 or curr_y+curr_w > self.width or curr_w < 50:
            done = True
            reward = reward - 10.0
        else:
            new_state = (curr_x / self.width, curr_y / self.width, curr_w / self.width)

        #cv2.imshow('Frame', frame)

        if not done:
            #rem = self.width - (curr_x + curr_w)
            rem = int(self.width / 2)
            diff_x = abs(curr_x - rem)

            remy = self.width - (curr_y + curr_w)
            remy = int(self.height/2)
            diff_y = abs(curr_y - remy)

            if (abs(curr_w - int(self.width / 5)) <= 10) and (diff_y < 20) and (diff_x < 20):
                #print('perfect')
                reward = reward + 10.0
            else:
                #print('diff', diff)
                if diff_x > 20 and diff_x < prev_diff_x:
                    reward = reward + (0.003 * (self.width-diff_x))
                elif diff_y > 20 and diff_y < prev_diff_y:
                    reward = reward + (0.001 * (self.width - diff_y))
                elif curr_w - int(self.width / 5) > 10:
                    if curr_w < prev_w:
                        reward = reward + 0.2
                    else:
                        reward = reward - 0.2
        if len(self.state) == 0:
            for i in range(4):
                self.state.append(np.array(new_state))
        elif not done:
            self.state.pop(0)
            self.state.append(np.array(new_state))
        return np.reshape(np.array(self.state), (12)), reward, done

    def render(self, mode='human'):
        cv2.waitKey(1)
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.width, self.height))
        cv2.rectangle(frame, (int(self.state[-1][0]*self.width), int(self.state[-1][1]*self.width)), (int(self.state[-1][0]*self.width) + int(self.state[-1][2]*self.width), int(self.state[-1][1]*self.width) + int(self.state[-1][2]*self.width)),
                      (0, 255, 0), 3)
        cv2.imshow('Frame', frame)

    def reset(self):
        data = np.random.randint(1, self.width-1, 3)
        while data[0] + data[2] > self.width-1 or data[1] + data[2] > self.width-1:
            data = np.random.randint(1, self.width-1, 3)
        self.state = []
        new_state = (data[0] / self.width, data[1] / self.width, data[2] / self.width)
        for i in range(4):
                self.state.append(np.array(new_state))
        
        return np.reshape(np.array(self.state), (12))

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
