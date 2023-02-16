import numpy as np
import time



# Define the transition function
def transition(state, action):
    i, j = state
    if action == 'up' or action == 0:
        i = max(i-1, 0)
    elif action == 'down' or action == 1:
        i = min(i+1, 9)
    elif action == 'left' or action == 2:
        j = max(j-1, 0)
    elif action == 'right' or action == 3:
        j = min(j+1, 9)
    return (i, j)

# Define the first-visit Monte Carlo control without exploring starts
def first_visit_monte_carlo_control(holes, reward_map, actions, num_episodes, discount_factor, epsilon):
    Q = 0.1*np.ones((10, 10, 4))
    # Q_ = 0.1*np.ones((10, 10, 4))
    returns = {}
    count = np.zeros((10, 10, 4))
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    danger_list = []
    start_time = time.time()
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(episode)
        state = (0, 0)
        episode_actions = []
        episode_rewards = []
        while state not in holes and state != (9,9):
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)  

            elif np.random.rand() < 0.5:
                for pair in danger_list:
                    if state == pair[0]:
                        actions_without_danger = list(actions)
                        actions_without_danger.remove(pair[1])
                        action = np.random.choice(actions_without_danger) 
            else:
                action = actions[np.argmax(Q[state[0], state[1]])]

            # if np.random.rand() < epsilon:
            #     action = np.random.choice(actions)      
            # else:
            #     # print([np.argmax(Q[state[0], state[1]])])
            #     action = actions[np.argmax(Q[state[0], state[1]])]

            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            accumulated_reward += reward
            accumulated_reward_50 += reward
            accumulated_reward_list.append(accumulated_reward)
            episode_actions.append((state, action))
            episode_rewards.append(reward)
            state = next_state
            if state in holes:
                if episode_actions[-1] not in danger_list:
                    danger_list.append(episode_actions[-1])
            if state == (9,9):
                print("Goal")
        G = 0
        if (episode+1)%50 == 0:
            reward_per_episode.append(accumulated_reward_50/50)
            accumulated_reward_50 = 0
        for i in range(len(episode_actions) - 1, -1, -1):
            state, action = episode_actions[i]
            G = discount_factor * G + episode_rewards[i]
            count[state[0], state[1], actions.index(action)] += 1 
            if (state, action) not in episode_actions[:i]:
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)
                # Q[state[0], state[1], actions.index(action)] = np.mean(returns[(state, action)])
                
                Q[state[0], state[1], actions.index(action)] += (G - Q[state[0], state[1], actions.index(action)]) / count[state[0], state[1], actions.index(action)]
            else:
                count[state[0], state[1], actions.index(action)] -= 1
        end_time = time.time()
        processing_time = end_time - start_time
    return Q, reward_per_episode, accumulated_reward_list, processing_time

# Define the SARSA with an ε-greedy behavior policy
def sarsa(holes, reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon):
    Q = 0.1*np.ones((10, 10, 4))
    # Q = np.zeros((10, 10, 4))
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    start_time = time.time()
    for episode in range(num_episodes):
        state = (0, 0)
        while state not in holes and state != (9,9):
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
    end_time = time.time() 
    processing_time = end_time - start_time

    return Q, reward_per_episode, accumulated_reward_list, processing_time

# Define the Q-learning with an ε-greedy behavior policy
def q_learning(holes, reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon):
    Q = 0.1*np.ones ((10, 10, 4))
    # Q = np.zeros((10, 10, 4))
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    start_time = time.time()
    for episode in range(num_episodes):
        state = (0, 0)
        while state not in holes and state != (9,9):
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
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
    end_time = time.time()
    processing_time = end_time - start_time
    return Q, reward_per_episode, accumulated_reward_list, processing_time

# Define the policy extraction function
def get_policy(holes, Q):
    policy = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            policy[i, j] = np.argmax(Q[i, j])
    for hole in holes:
        policy[hole] = -1
    policy[9,9] = 10
    return policy

def vis_policy(policy):
    vis_policy = np.zeros((10, 10), dtype=str)
    mapping = {0: '↑', 1: '↓', 2: '←', 3: '→', -1: 'O', 10: 'G'}
    for i in range(len(policy)):
        for j in range(len(policy[0])):
            vis_policy[i,j] = mapping[policy[i,j]]
    return vis_policy

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