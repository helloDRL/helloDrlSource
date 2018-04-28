import numpy as np
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import random as pr

register(
    id = 'FrozenLake-non-slippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4',  #4x4, 8x8
           'is_slippery' : False}
)

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


env = gym.make('FrozenLake-non-slippery-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episode = 1000
rList = []
cList = []

for i in range(num_episode):
    state = env.reset()
    rAll = 0
    done = False
    count = 0

    while not done:
        action = rargmax(Q[state,:])
        new_state, reward, done, info = env.step(action)
        count += 1

        Q[state,action] = reward + np.max(Q[new_state,:])
        rAll += reward
        state = new_state

    print("episode=",i,", reward=",reward)
    rList.append(rAll)
    #성공일 때의 step count를 기록
    if reward > 0:
        cList.append(count)
        print("stepCount", count)


print(Q)

print("Success rate:", str(sum(rList)/len(rList)))
print("step count:", str(sum(cList)/len(cList)))

plt.bar(range(len(rList)), rList)
plt.show()
plt.bar(range(len(cList)), cList)
plt.show()





