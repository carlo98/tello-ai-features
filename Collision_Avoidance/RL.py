import numpy as np
import random
import time
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import tdqm
from collections import deque
#from Collision_Avoidance.model import tommy_net
from model import tommy_net


class RL_Agent:

    def __init__(self, parameters):
        self.optimizer = optim.RMSprop(parameters, lr=1e-6)
        self.discount_factor = 0.8
        self.max_steps = 100
        self.device = torch.device('cpu')
        self.save_path = "rl_saved_models/best_model.pth"
        self.save_freq = 5  # In episodes

    def update_model(self, model, state, action, reward, next_state):

        q_preds = model(state)

        q_vals = q_preds[:, action]

        with torch.no_grad():
            q_next_preds = model(next_state)
            q_next_vals = q_next_preds.max(1).values
            targets = reward + q_next_vals * self.discount_factor

        loss = F.smooth_l1_loss(targets.detach(), q_vals)

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        self.optimizer.step()

        return loss.item()

    def save_model(self, model, num_episodes):
        if num_episodes % self.save_freq == 0:
            torch.save(model.state_dict(), self.save_path)
