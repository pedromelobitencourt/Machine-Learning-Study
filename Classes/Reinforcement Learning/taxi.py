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
        q_old = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_new = ((1 - alpha) * q_old) + (alpha * (reward + (gamma * next_max)))
        q_table[state, action] = q_new

        if reward == 10:
            penalty += 1
        
        state = next_state

    if i % 100 == 0:
        clear_output(wait=True)
        print('Episode: ', i)

print('Done')

print(q_table)

env.reset()
env.render()

# env.encode(row, column, passenger, destination): to view the state
# q_table[state]: to view what should be done (north, south...) (greatest value)

env.encode(4, 1, 1, 0) # each state is this
q_table[424] # the best path choise has the greatest value


total_penalty = 0
episodes = 50
frames = []

for _ in range(episodes):
  state = env.reset()
  penalty, reward = 0, 0
  done = False

  while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == 10:
      penalty += 1

    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    })

    total_penalty += penalty

print('Total penalty:', total_penalty)


from time import sleep

# animation

for frame in frames:
  clear_output(wait=True)
  print(frame['frame'])
  print('state', frame['state'])
  print('action', frame['action'])
  print('reward', frame['reward'])
  sleep(.5)