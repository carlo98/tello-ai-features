import numpy as np
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import tdqm
from collections import deque
#from Collision_Avoidance.model import tommy_net
from model import tommy_net

class RL_Agent:
    def train(env, policy, optimizer, discount_factor, epsilon, device):
    
        policy.train()
    
        states = []
    actions = []
    rewards = []
    next_states = []
    done = False
    episode_reward = 0

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    while not done:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()
        
        next_state, reward, done, _ = env.step(action)

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        state = next_state

        episode_reward += reward

        loss = update_policy(policy, state, action, reward, next_state, discount_factor, optimizer)

    return loss, episode_reward, epsilon

def update_policy(policy, state, action, reward, next_state, discount_factor, optimizer):
    
    q_preds = policy(state)

    q_vals = q_preds[:, action]

    with torch.no_grad():
        q_next_preds = policy(next_state)
        q_next_vals = q_next_preds.max(1).values
        targets = reward + q_next_vals * discount_factor

    loss = F.smooth_l1_loss(targets.detach(), q_vals)
    
    optimizer.zero_grad()
    
    loss.backward()

    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

    optimizer.step()
    
    return loss.item()
    
def evaluate(env, policy, device):
    
    policy.eval()
    
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
        
            q_pred = policy(state)
            
            action = torch.argmax(q_pred).item()

        state, reward, done, _ = env.step(action)

        episode_reward += reward

    return episode_reward
    
n_runs = 10
n_episodes = 500
discount_factor = 0.8
start_epsilon = 1.0
end_epsilon = 0.01
epsilon_decay = 0.995

train_rewards = torch.zeros(n_runs, n_episodes)
test_rewards = torch.zeros(n_runs, n_episodes)
device = torch.device('cpu')

for run in range(n_runs):
    
    model = tommy_net()
    self.model.load_state_dict(torch.load('Collision_Avoidance/saved_models/best_model.pth', map_location=torch.device('cpu')))
    policy = model.to(device)
    epsilon = start_epsilon

    optimizer = optim.RMSprop(policy.parameters(), lr=1e-6)

    for episode in tqdm.tqdm(range(n_episodes), desc=f'Run: {run}'):

        loss, train_reward, epsilon = train(train_env, model, optimizer, discount_factor, epsilon, device)

        epsilon *= epsilon_decay
        epsilon = min(epsilon, end_epsilon)

        test_reward = evaluate(test_env, policy, device)
        
        train_rewards[run][episode] = train_reward
        test_rewards[run][episode] = test_reward
