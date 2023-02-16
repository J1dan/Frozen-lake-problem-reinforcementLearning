import numpy as np
import random
import gym

# Define environment and hyperparameters
env = gym.make('FrozenLake-v1')
num_episodes = 1000
discount_factor = 0.99
epsilon = 0.1
alpha = 0.5

# Initialize Q-value function for Q-learning
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Initialize dictionary to store returns and Q-values for first-visit Monte Carlo
returns = {}
Q_mc = np.zeros((env.observation_space.n, env.action_space.n))
N = np.zeros((env.observation_space.n, env.action_space.n))

# Run Q-learning and record cumulative reward
Q_rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    cum_reward = 0
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward
        Q[state, action] += alpha * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state
    
    Q_rewards.append(cum_reward)

# Run first-visit Monte Carlo and record cumulative reward
mc_rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    cum_reward = 0
    episode_list = []
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_mc[state, :])
        
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward
        episode_list.append((state, action, reward))
        state = next_state
    
    G = 0
    visited_states = set()
    for t in range(len(episode_list) - 1, -1, -1):
        state, action, reward = episode_list[t]
        G = discount_factor * G + reward
        if state not in visited_states:
            visited_states.add(state)
            returns[(state, action)] = returns.get((state, action), []) + [G]
            Q_mc[state, action] = np.mean(returns[(state, action)])
    
    mc_rewards.append(cum_reward)
