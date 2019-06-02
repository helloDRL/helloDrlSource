import gym
import numpy as np
import random
import pylab
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import time

env = gym.make('Enduro-ram-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#model
model = Sequential()
model.add(Dense(52, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(52, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))

epsilon = 0
discount_factor = 0.99

def get_action(state):
    q_values = model.predict(state)
    action = np.argmax(q_values[0])
    return action

episodes = []
scores = []

model.load_weights('./save_model/enduro_dqn500.h5' )

#episode
for i in range(3):

    state = env.reset()
    state = np.reshape(state,[1,state_size])
    action = 0
    step = 0

    total_reward = 0

    while True:
        time.sleep(0.01)
        env.render()
        step = step + 1
        action = get_action(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print('action=',action, ',reward=', reward, ',total_reward=',total_reward)

        state = np.reshape(next_state,[1,state_size])

        if done:
            episodes.append(i)
            scores.append(step)
            break
    print('episode=',i, ',step=', step, 'total_reward', total_reward)







