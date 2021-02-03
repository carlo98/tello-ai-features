import random
import torch
import copy
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from Collision_Avoidance.model import tommy_net
import pickle
import sys
import os


class RL_Agent:

    def __init__(self, model, device):
        self.discount_factor = 0.8
        self.max_steps = 500
        if not os.path.isdir("Collision_Avoidance/rl_saved_models/"):
            print("Please first create the folder 'Collision_Avoidance/rl_saved_models/' with 'mkdir Collision_Avoidance/rl_saved_models/'.")
            sys.exit()
        self.save_dir = "Collision_Avoidance/rl_saved_models/"
        self.save_freq = 5  # In episodes
        self.update_target_freq = 4  # In episodes, should be a multiple of train_freq
        self.batch_size = 16
        self.num_epochs = 2
        self.action_size = 2
        self.train_freq = 2  # In episodes
        self.memory = deque(maxlen=1000000)
        if os.path.isfile(self.save_dir + "memory.bin"):
            with open(self.save_dir + "memory.bin", "rb") as f:
                self.memory = pickle.load(f)
            print("Found pre-existing memory, loaded")
        self.device = device
        self.state_size = None
        
        self.target_model = tommy_net()
        self.target_model.load_state_dict(model.state_dict())
        self.target_model = self.target_model.to(device)
        self.target_model.eval()
        
        self.optimizer = optim.RMSprop(model.parameters(), lr=1e-6)
        
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

    def update_model(self, model, episode_num):
    
        if episode_num % self.train_freq != 0 or len(self.memory) < self.batch_size:
            return
            
        print("Training")
    
        for j in range(self.num_epochs):
            mini_batch = random.sample(self.memory, self.batch_size)
            
            mean_loss = 0.0

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

                q_value = model(states)

                if episode_num % self.update_target_freq == 0:
                    next_target = self.target_model(next_states)

                else:
                    next_target = model(next_states)

                next_q_value = self.getQvalue(rewards, next_target, dones)
                
                loss = F.smooth_l1_loss(torch.tensor(next_q_value, dtype=torch.float32).requires_grad_(), torch.max(q_value))
                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                self.optimizer.step()
                
                mean_loss += loss.item()
                
            print("Epoch ", j, " mean loss: ", mean_loss / self.batch_size)
        
        print("Training Ended")
            
        if episode_num % self.update_target_freq == 0:
            self.target_model.load_state_dict(model.state_dict())

    def save_model(self, model, num_episodes):
        if num_episodes % self.save_freq == 0:
            torch.save(model.state_dict(), self.save_dir + "best_model_" + str(num_episodes) + ".pth")  # Save model state dict
            self.save_memory()
                
    def save_memory(self):
        with open(self.save_dir + "memory.bin", "wb") as f:
            pickle.dump(self.memory, f)  # Save memory
