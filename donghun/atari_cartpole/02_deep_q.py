import sys
import gym
import pylab
import random
import numpy as np
import collections
import keras

EPISODES = 300
TRAIN_MODE = False

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        self.memory = collections.deque(maxlen = 2000)

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()

        if not TRAIN_MODE:
            self.epsilon = 0
            self.render = True
            self.model.load_weights("./save_model/cartpole_dqn.h5")

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(24, activation='relu',kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [],[],[]

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else :
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1,state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state,[1, state_size])

            reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, reward, next_state, done)

            if TRAIN_MODE and len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                if TRAIN_MODE :
                    agent.update_target_model()

                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)

                print("episode:", e, " score:", score, " memory_length:", len(agent.memory), " epsilon:", agent.epsilon)

                mean_score = np.mean(scores[-min(10, len(scores)) : ])
                print('mean score:', mean_score)

                if TRAIN_MODE and  mean_score > 490:
                    agent.model.save_weights("./save_model/cartpole_dqn.h5")
                    sys.exit()







