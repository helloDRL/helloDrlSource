import keyin
import gym
from gym.envs.registration import register

register(
    id = 'FrozenLake-non-slippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'8x8',
           'is_slippery' : False}
)

env = gym.make('FrozenLake-non-slippery-v0')

env.reset()
env.render()

while True:
    key = keyin.inkey()
    if key not in keyin.arrow_keys.keys():
        print('Game aborted~~')
        break

    action = keyin.arrow_keys[key]

    state, reward, done, info = env.step(action)

    env.render()

    print('State:', state, ",Action:", action, ",Reward:", reward, ",Info:", info)

    if done:
        print("Finished with reward", reward)
        break


