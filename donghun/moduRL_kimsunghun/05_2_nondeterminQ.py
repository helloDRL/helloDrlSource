import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id = 'FrozenLake-slippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', #4x4, 8x8
           'is_slippery' : True}
)

#env = gym.make('FrozenLake--v0')   #map size변경 필요시 위에 register 사용

env = gym.make('FrozenLake-slippery-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

discount = .99
num_episode = 2000
learning_rate = .85

rList = []
cList = []

for i in range(num_episode):
    e =  1. / ((i//100)+1)

    state = env.reset()
    rAll = 0
    done = False
    count = 0   #step count

    while not done:
        #add noise
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)/(i+1) )

        #e-greedy
        #if np.random.rand(1) < e :
        #    action = env.action_space.sample()
        #else:
        #    action = np.argmax(Q[state,:])


        new_state, reward, done, info = env.step(action)

        #Q[state, action] = reward + discount * np.max(Q[new_state,:])
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * ( reward + discount * np.max(Q[new_state,:]))

        rAll += reward
        count += 1
        state = new_state


    #print("episode=",i,", reward=",reward)
    rList.append(rAll)
    #성공일 때의 step count를 기록
    if reward > 0:
        cList.append(count)
        #print("stepCount", count)


print(Q)

print("Success rate:", str(sum(rList)/len(rList)))
print("step count:", str(sum(cList)/len(cList)))

plt.bar(range(len(rList)), rList)
plt.show()
plt.bar(range(len(cList)), cList)
plt.show()



