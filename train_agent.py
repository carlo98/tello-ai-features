import os
import json
import numpy as np
import random
import time
from collections import deque
import sys
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
from Simulator import SimulatorEnv

EPISODES = 30000

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.dirPath = "saved_models/"

        self.load_model = False
        self.load_episode = 3473
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0020
        self.epsilon = 0.85
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 16
        self.train_start = 1500
        self.training_frequency = 150
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = 0.0 #param.get('epsilon')
                self.load_episode = 0

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(32, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(32, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == '__main__':

    state_size = 12
    action_size = 12

    env = SimulatorEnv()

    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        trained_ep = False
        for t in range(agent.episode_step):
            action = agent.getAction(state)
            
            next_state, reward, done = env.step(action)
            #env.render()            

            agent.appendMemory(state, action, reward, next_state, done)
            
            trained_step = False
            if len(agent.memory) >= agent.train_start and (len(agent.memory) % agent.training_frequency) == 0:
                print("Training!")
                trained_step = True
                trained_ep = True
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state

            if trained_step:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= 500:
                print("Done!")
                done = True

            if done:
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                print('Ep: ', e, ' score: ',score,' memory: ',len(agent.memory),' epsilon: ',agent.epsilon,' time: ',h,':',m,':',s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                print("UPDATE TARGET NETWORK")

        if trained_ep and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
