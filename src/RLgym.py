import gym
import numpy as np

# create the FrozenLake-v0 environment
env = gym.make('FrozenLake-v1')

# define the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# set the learning parameters
num_episodes = 5000
discount_factor = 0.99
learning_rate = 0.8
epsilon = 0.1

# train the Q-table
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # choose an action using the ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state[0], :])
        next_state, reward, done, _ = env.step(action)
        # update the Q-table
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state

# evaluate the performance of the Q-table
total_rewards = 0
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        total_rewards += reward
print("Average rewards over 100 episodes:", total_rewards / 100)
