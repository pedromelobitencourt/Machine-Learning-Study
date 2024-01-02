import gym
import random
import numpy as np

env = gym.make("Taxi-v3")
env.reset()
env.render()

print('Action space: ', env.action_space) # north (1), east (2), south (0), west (3), pickup (4), dropoff (5)
print('States', env.observation_space)
print(len(env.P))
print('Possibilities', env.P[484]) # 0 (south): Probability of executing this step, next stop, reward, done

q_table = np.zeros([env.observation_space.n, env.action_space.n]) # 500 states and 6 actions each

## Training
from IPython.display import clear_output

alpha = 0.1 # learning rate
gamma = 0.6 # discount factor
epsilon = 0.1 # probability to explore other paths (exploration)

for i in range(100000):
    state = env.reset() # initial state
    penalty, reward = 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon: # exploration
            action = env.action_space.sample() # random action
        else: # exploitation
            action = np.argmax(q_table[state])
            next_state, reward, done, info = env.step(action)