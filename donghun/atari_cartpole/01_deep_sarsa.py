import gym
import numpy as np
import random
import pylab
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#model
model = Sequential()
model.add(Dense(30, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))

epsilon = 0.01
discount_factor = 0.99

def get_action(state):
    if np.random.rand() <= epsilon:
        action = random.randrange(action_size)
    else:
        q_values = model.predict(state)
        action = np.argmax(q_values[0])

    return action

episodes = []
scores = []

#episode
for i in range(10000000):

    state = env.reset()
    state = np.reshape(state,[1,state_size])
    action = 0
    step = 0
    #epsilon = epsilon * 0.999

    while True:
        #env.render()
        step = step + 1
        action = get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state,[1,state_size])
        next_action = get_action(next_state)

        target = model.predict(state)[0]

        if done:
            target[action] = reward
        else:
            target[action] = (reward + discount_factor*model.predict(next_state)[0][next_action] )

        target = np.reshape(target,[1,action_size])
        model.fit(state, target, epochs=1, verbose=0)

        state = next_state

        if done:
            episodes.append(i)
            scores.append(step)
            break
    print('episode=',i, ',step=', step)

    if i % 1000 == 0 :
        pylab.plot(episodes, scores, 'b')
        pylab.savefig('./deep_sarsa.png')
        model.save_weights('./save_model/cartpole_sarsa.h5')




