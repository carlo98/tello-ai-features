import numpy as np
import random
import time
import torch
import copy
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import deque
from Collision_Avoidance.model import tommy_net


class RL_Agent:

    def __init__(self, device):
        self.discount_factor = 0.8
        self.max_steps = 100
        self.save_path = "Collision_Avoidance/rl_saved_models/best_model.pth"
        self.save_freq = 5  # In episodes
        self.update_target_freq = 4
        self.batch_size = 4
        self.action_size = 2
        self.train_freq = 1
        self.memory = deque(maxlen=1000000)
        self.device = device
        self.state_size = None
        self.model = tommy_net()
        self.model.load_state_dict(torch.load('Collision_Avoidance/saved_models/best_model.pth', map_location=torch.device(device)))
        self.model = self.model.to(device)
        
        self.target_model = tommy_net()
        self.target_model.load_state_dict(torch.load('Collision_Avoidance/saved_models/best_model.pth', map_location=torch.device(device)))
        self.target_model = self.target_model.to(device)
        self.target_model.eval()
        
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-6)
        
    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * torch.argmax(next_target)
            
    def appendMemory(self, state, action, reward, next_state, done):
        if self.state_size is None:
            tmp = state[None, None, ...]
            self.state_size = tmp.shape
        self.memory.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))

    def update_model(self, episode_num):
    
        mini_batch = random.sample(self.memory, self.batch_size)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            states = torch.from_numpy(states).float()
            states = states.to(self.device)
            states = states[None, None, ...]
            next_states = mini_batch[i][3]
            next_states = torch.from_numpy(next_states).float()
            next_states = next_states.to(self.device)
            next_states = next_states[None, None, ...]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            dones = mini_batch[i][4]

            q_value = self.model(states)

            if episode_num % self.update_target_freq == 0:
                next_target = self.target_model(next_states)

            else:
                next_target = self.model(next_states)

            next_q_value = self.getQvalue(rewards, next_target, dones)
                
            loss = F.smooth_l1_loss(torch.tensor(next_q_value, dtype=torch.float32).requires_grad_(), torch.max(q_value))
            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()
            
        if episode_num % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, num_episodes):
        if num_episodes % self.save_freq == 0:
            torch.save(self.model.state_dict(), self.save_path)
