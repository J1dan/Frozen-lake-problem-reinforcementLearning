import numpy as np




# Define the transition function
def transition(state, action):
    i, j = state
    if action == 'up' or action == 0:
        i = max(i-1, 0)
    elif action == 'down' or action == 1:
        i = min(i+1, 3)
    elif action == 'left' or action == 2:
        j = max(j-1, 0)
    elif action == 'right' or action == 3:
        j = min(j+1, 3)
    return (i, j)

# Define the first-visit Monte Carlo control without exploring starts
def first_visit_monte_carlo_control(reward_map, actions, num_episodes, discount_factor):
    Q = np.zeros((4, 4, 4))
    returns = np.zeros((4, 4, 4))
    count = np.zeros((4, 4, 4))
    for episode in range(num_episodes):
        state = (0, 0)
        episode_states = []
        episode_actions = []
        episode_rewards = []
        # while state != (3, 3):
        while state != (3, 3) and state !=(1, 1) and state !=(1, 3) and state !=(3, 1) and state !=(2, 3):
            action = np.random.choice(actions)
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
        G = 0
        for i in range(len(episode_states) - 1, -1, -1):
            state = episode_states[i]
            action = episode_actions[i]
            reward = episode_rewards[i]
            G = discount_factor * G + reward
            count[state[0], state[1], actions.index(action)] += 1
            Q[state[0], state[1], actions.index(action)] = Q[state[0], state[1], actions.index(action)] + (G - Q[state[0], state[1], actions.index(action)]) / count[state[0], state[1], actions.index(action)]
    return Q

# Define the SARSA with an ε-greedy behavior policy
def sarsa(reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon):
    Q = np.zeros((4, 4, 4))
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    for episode in range(num_episodes):
        state = (0, 0)
        while state != (3, 3) and state !=(1, 1) and state !=(1, 3) and state !=(3, 1) and state !=(2, 3):
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(Q[state[0], state[1]])]
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            accumulated_reward += reward
            accumulated_reward_50 += reward
            accumulated_reward_list.append(accumulated_reward)
            if np.random.rand() < epsilon:
                next_action = np.random.choice(actions)
            else:
                next_action = actions[np.argmax(Q[next_state[0], next_state[1]])]
            Q[state[0], state[1], actions.index(action)] += learning_rate * \
                (reward + discount_factor * Q[next_state[0], next_state[1], actions.index(next_action)] \
                - Q[state[0], state[1], actions.index(action)])
            state = next_state
        if (episode+1)%50 == 0:
            reward_per_episode.append(accumulated_reward_50/50)
            accumulated_reward_50 = 0
    # print(reward_per_episode)
    return Q, reward_per_episode, accumulated_reward_list

# Define the Q-learning with an ε-greedy behavior policy
def q_learning(reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon):
    Q = np.zeros((4, 4, 4))
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    for episode in range(num_episodes):
        state = (0, 0)
        while state != (3, 3) and state !=(1, 1) and state !=(1, 3) and state !=(3, 1) and state !=(2, 3):
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                if np.allclose(Q[state[0], state[1]],Q[state[0],state[1],0]):
                    if np.random.rand() < 0.5:
                        action = 'right'
                    else:
                        action = 'down'
                    # action = np.random.choice(actions)
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            accumulated_reward += reward
            accumulated_reward_50 += reward
            accumulated_reward_list.append(accumulated_reward)
            Q[state[0], state[1], actions.index(action)] += \
                learning_rate * (reward + discount_factor * np.max(Q[next_state[0], next_state[1]]) \
                - Q[state[0], state[1], actions.index(action)])
            state = next_state
        if (episode+1)%50 == 0:
            reward_per_episode.append(accumulated_reward_50/50)
            accumulated_reward_50 = 0
    return Q, reward_per_episode, accumulated_reward_list

# Define the policy extraction function
def get_policy(Q):
    policy = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            policy[i, j] = np.argmax(Q[i, j])
    return policy

def evaluate_performance(reward_map, policy, num_episodes):
    total_rewards = 0
    for episode in range(num_episodes):
        state = (0, 0)
        episode_reward = 0
        while state != (3, 3) and state !=(1, 1) and state !=(1, 3) and state !=(3, 1) and state !=(2, 3):
            action = policy[state[0], state[1]]
            print(f"state = {state}")
            print(f"action = {action}")
            next_state = transition(state, action)
            print(f"next_state = {next_state}")
            reward = reward_map[next_state[0], next_state[1]]
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward
    avg_reward = total_rewards / num_episodes
    return avg_reward