import numpy as np




# Define the transition function
def transition(state, action):
    i, j = state
    if action == 'up':
        i = max(i-1, 0)
    elif action == 'down':
        i = min(i+1, 3)
    elif action == 'left':
        j = max(j-1, 0)
    elif action == 'right':
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
    for episode in range(num_episodes):
        state = (0, 0)
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]
        # while state != (3, 3):
        while state != (3, 3) and state !=(1, 1) and state !=(1, 3) and state !=(3, 1) and state !=(2, 3):
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            if np.random.rand() < epsilon:
                next_action = np.random.choice(actions)
            else:
                next_action = actions[np.argmax(Q[next_state[0], next_state[1]])]
            Q[state[0], state[1], actions.index(action)] += learning_rate * (reward + discount_factor * Q[next_state[0], next_state[1], actions.index(next_action)] - Q[state[0], state[1], actions.index(action)])
            state = next_state
            action = next_action
    return Q

# Define the Q-learning with an ε-greedy behavior policy
def q_learning(reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon):
    Q = np.zeros((4, 4, 4))
    for episode in range(num_episodes):
        state = (0, 0)
        while state != (3, 3) and state !=(1, 1) and state !=(1, 3) and state !=(3, 1) and state !=(2, 3):
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(Q[state[0], state[1]])]
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            Q[state[0], state[1], actions.index(action)] += learning_rate * (reward + discount_factor * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], actions.index(action)])
            state = next_state
    return Q

# Define the policy extraction function
def get_policy(Q):
    policy = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            policy[i, j] = np.argmax(Q[i, j])
    return policy