import numpy as np
import time

actionMap = {'up':0, 'down':1, 'left':2, 'right':3}

# Define the transition function
def transition(state, action):
    ''' Calculate the transition given the state and the action
    
    Parameters
    ----------
    
    state (`tuple`): the coordinate of the robot
    action (`string`): the action of the robot

    Returns
    -------
    state (`tuple`): next state
    '''
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
def first_visit_monte_carlo_control(holes, reward_map, actions, num_episodes, discount_factor, epsilon, method = 'None'):
    ''' Use first-visit Monte Carlo control to solve the RL problem
    
    Parameters
    ----------
    
    holes (`list[tuple]`): list of tuples containing the coordinates of the holes
    reward_map (`ndarray`): the reward map
    actions (`list`): list of strings representing all actions
    num_episodes (`int`): the number of episodes
    discount_factor (`int`): the discount factor
    epsilon (`string`): the hyperparameter epsilon used for epsilong-greedy
    method (`string`): the method being used. Options: 'None', 'randomArgmax', 'q-reset', 'combined'

    Returns
    -------
    Q (`ndarray`): the trained Q-table of the robot
    reward_per_episode (list): list of reward per 50 episodes
    accumulated_reward_list (list): list of accumulated reward
    processing_time (int): the processing time
    convergence_episode (int): the number of episode needed to converge
    convergence_action (int): the number of action needed to converge
    '''
    if method == 'None' or method == 'randomArgmax':
        Q = np.zeros((10, 10, 4))
    elif method == 'q_reset' or method == 'combined' or method == 'dangerAvoidance'or method == 'exploringStart':
        Q = 0.1 * np.ones((10, 10, 4))
    else:
        print("InvalidMethod!")
        return
    # Initialization
    returns = {}
    count = np.zeros((10, 10, 4))
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    danger_list = []
    convergence_episode = 1.5 * num_episodes
    convergence_action = 500000
    convergence_label = 0
    start_time = time.time()
    for episode in range(num_episodes): # Training begins
        # if episode % 100 == 0:
        #     print(episode)
        if method == 'exploringStart':
            state = (np.random.randint(0, 10), np.random.randint(0, 10)) # random initial state
            while state in holes or state == (9, 9): # if initial state is a hole or the frisbee, then re-randomize
                state = (np.random.randint(0, 10), np.random.randint(0, 10))
        else:
            state = (0, 0)
        episode_rewards = []
        episode_actions = []
        while state not in holes and state != (9, 9):
            if method == 'None' or method == 'q_reset' or method == 'exploringStart':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]

            elif method == 'randomArgmax' or method == 'combined':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)
                else:
                    if np.allclose(Q[state[0], state[1]],Q[state[0],state[1],0]):
                        action = np.random.choice(actions)
                    else:
                        action = actions[np.argmax(Q[state[0], state[1]])]
            elif method == 'dangerAvoidance':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)  
                elif np.random.rand() < 0.5:
                    action = actions[np.argmax(Q[state[0], state[1]])]
                    for pair in danger_list:
                        if state == pair[0]: # State in danger_list
                            actions_without_danger = list(actions)
                            actions_without_danger.remove(pair[1])
                            action = np.random.choice(actions_without_danger)
                            break
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]

            else:
                print("InvalidMethod!")
                return
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            # Update
            accumulated_reward += reward
            accumulated_reward_50 += reward
            accumulated_reward_list.append(accumulated_reward)
            episode_actions.append((state, action))
            episode_rewards.append(reward)
            state = next_state
            if method == 'dangerAvoidance':
                if state in holes:
                    if episode_actions[-1] not in danger_list:
                        danger_list.append(episode_actions[-1])
        G = 0
        if (episode+1)%50 == 0: # Update every 50 episodes
            reward_per_episode.append(accumulated_reward_50/50)
            if accumulated_reward_50/50 > -0.15: # When the average reward exceeds -0.1, considered converged
                if convergence_label == 0:
                    convergence_episode = episode + 1
                    convergence_action = len(accumulated_reward_list)
                    convergence_label = 1
            accumulated_reward_50 = 0

        for i in range(len(episode_actions) - 1, -1, -1): # Update the Q values
            state, action = episode_actions[i]
            G = discount_factor * G + episode_rewards[i]
            count[state[0], state[1], actions.index(action)] += 1 
            if (state, action) not in episode_actions[:i]: # First visited
                # Incremental Q value update
                Q[state[0], state[1], actions.index(action)] = Q[state[0], state[1], actions.index(action)]\
                + (G - Q[state[0], state[1], actions.index(action)]) / count[state[0], state[1], actions.index(action)]
            else: # Already visited
                count[state[0], state[1], actions.index(action)] -= 1 # Current state-action has been visited, thus ignore it
        end_time = time.time()
        processing_time = end_time - start_time
    return Q, reward_per_episode, accumulated_reward_list, processing_time, convergence_episode, convergence_action

# Define the SARSA with an ε-greedy behavior policy
def sarsa(holes, reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon, method = 'None'):
    ''' Use SARSA to solve the RL problem
    
    Parameters
    ----------
    
    holes (`list[tuple]`): list of tuples containing the coordinates of the holes
    reward_map (`ndarray`): the reward map
    actions (`list`): list of strings representing all actions
    num_episodes (`int`): the number of episodes
    discount_factor (`int`): the discount factor
    learning_rate (`int`): the hyperparameter learning rate which affects the update speed
    epsilon (`string`): the hyperparameter epsilon used for epsilong-greedy
    method (`string`): the method being used. Options: 'None', 'randomArgmax', 'q-reset', 'combined'

    Returns
    -------
    Q (`ndarray`): the trained Q-table of the robot
    reward_per_episode ('list'): list of reward per 50 episodes
    accumulated_reward_list ('list'): list of accumulated reward
    processing_time ('int'): the processing time
    convergence_episode ('int'): the number of episode needed to converge
    convergence_action ('int'): the number of action needed to converge
    '''
    if method == 'None' or method == 'randomArgmax':
        Q = np.zeros((10, 10, 4))
    elif method == 'q_reset' or method == 'combined' or method == 'dangerAvoidance' or method == 'exploringStart':
        Q = 0.1 * np.ones((10, 10, 4))
    else:
        print("InvalidMethod!")
        return
    # Initialization
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    convergence_episode = 1.5 * num_episodes
    convergence_action = 3000000
    convergence_label = 0
    danger_list = []
    start_time = time.time()
    for episode in range(num_episodes):
        episode_actions = []
        if method == 'exploringStart':
            state = (np.random.randint(0, 10), np.random.randint(0, 10)) # random initial state
            while state in holes or state == (9, 9): # if initial state is a hole or the frisbee, then re-randomize
                state = (np.random.randint(0, 10), np.random.randint(0, 10))
        else:
            state = (0, 0)
        while state not in holes and state != (9, 9):
            # Epsilon-greedy behavior policy
            if method == 'None' or method == 'q_reset' or method == 'exploringStart':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]

            elif method == 'randomArgmax' or method == 'combined':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)
                else:
                    if np.allclose(Q[state[0], state[1]],Q[state[0],state[1],0]):
                        action = np.random.choice(actions)
                    else:
                        action = actions[np.argmax(Q[state[0], state[1]])]
            elif method == 'dangerAvoidance':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)  
                elif np.random.rand() < 0.5:
                    action = actions[np.argmax(Q[state[0], state[1]])]
                    for pair in danger_list:
                        if state == pair[0]:
                            actions_without_danger = list(actions)
                            actions_without_danger.remove(pair[1])
                            action = np.random.choice(actions_without_danger) 
                            break
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]
            else:
                print("InvalidMethod!")
                return

            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            # Update
            accumulated_reward += reward
            accumulated_reward_50 += reward
            accumulated_reward_list.append(accumulated_reward)
            episode_actions.append((state, action))
            # Epsilon-greedy target policy
            if np.random.rand() < epsilon:
                next_action = np.random.choice(actions)
            else:
                next_action = actions[np.argmax(Q[next_state[0], next_state[1]])]
            Q[state[0], state[1], actions.index(action)] += learning_rate * \
                (reward + discount_factor * Q[next_state[0], next_state[1], actions.index(next_action)] \
                - Q[state[0], state[1], actions.index(action)])
            state = next_state
            if method == 'dangerAvoidance':
                if state in holes:
                    if episode_actions[-1] not in danger_list:
                        danger_list.append(episode_actions[-1])
        if (episode+1)%50 == 0:# Update every 50 episodes
            reward_per_episode.append(accumulated_reward_50/50)
            if accumulated_reward_50/50 > 0.7: # When the average reward exceeds 0.7, considered converged
                if convergence_label == 0:
                    convergence_episode = episode + 1
                    convergence_action = len(accumulated_reward_list)
                    convergence_label = 1
            accumulated_reward_50 = 0
    end_time = time.time() 
    processing_time = end_time - start_time
    return Q, reward_per_episode, accumulated_reward_list, processing_time, convergence_episode, convergence_action

# Define the Q-learning with an ε-greedy behavior policy
def q_learning(holes, reward_map, actions, num_episodes, discount_factor, learning_rate, epsilon, method = 'None'):
    ''' Use Q-learning to solve the RL problem
    
    Parameters
    ----------
    
    holes (`list[tuple]`): list of tuples containing the coordinates of the holes
    reward_map (`ndarray`): the reward map
    actions (`list`): list of strings representing all actions
    num_episodes (`int`): the number of episodes
    discount_factor (`int`): the discount factor
    learning_rate (`int`): the hyperparameter learning rate which affects the update speed
    epsilon (`string`): the hyperparameter epsilon used for epsilong-greedy
    method (`string`): the method being used. Options: 'None', 'randomArgmax', 'q-reset', 'combined'

    Returns
    -------
    Q (`ndarray`): the trained Q-table of the robot
    reward_per_episode ('list'): list of reward per 50 episodes
    accumulated_reward_list ('list'): list of accumulated reward
    processing_time ('int'): the processing time
    convergence_episode ('int'): the number of episode needed to converge
    convergence_action ('int'): the number of action needed to converge
    '''
    if method == 'None' or method == 'randomArgmax':
        Q = np.zeros((10, 10, 4))
    elif method == 'q_reset' or method == 'combined' or method == 'dangerAvoidance' or method == 'exploringStart':
        Q = 0.1 * np.ones((10, 10, 4))
    else:
        print("InvalidMethod!")
        return
    # Initialization
    reward_per_episode = []
    accumulated_reward = 0
    accumulated_reward_50 = 0
    accumulated_reward_list = []
    convergence_episode = 1.5 * num_episodes # If not converge, use 1.5 * num_episodes as the number to converge
    convergence_action = 10000000 # If not converge, use 10000000 as the number of actions to converge
    convergence_label = 0
    danger_list = []
    start_time = time.time()
    for episode in range(num_episodes):
        if method == 'exploringStart':
            state = (np.random.randint(0, 10), np.random.randint(0, 10)) # random initial state
            while state in holes or state == (9, 9): # if initial state is a hole or the frisbee, then re-randomize
                state = (np.random.randint(0, 10), np.random.randint(0, 10))
        else:
            state = (0, 0)
        episode_actions = []
        while state not in holes and state != (9, 9):
            # Epsilon-greedy behavior policy
            if method == 'None' or method == 'q_reset':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]

            elif method == 'randomArgmax' or method == 'combined':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)
                else:
                    if np.allclose(Q[state[0], state[1]],Q[state[0],state[1],0]):
                        action = np.random.choice(actions)
                    else:
                        action = actions[np.argmax(Q[state[0], state[1]])]

            elif method == 'dangerAvoidance' or method == 'exploringStart':
                if np.random.rand() < epsilon:
                    action = np.random.choice(actions)  
                elif np.random.rand() < 0.5:
                    action = actions[np.argmax(Q[state[0], state[1]])]
                    for pair in danger_list:
                        if state == pair[0]:
                            actions_without_danger = list(actions)
                            actions_without_danger.remove(pair[1])
                            action = np.random.choice(actions_without_danger)
                            break
                else:
                    action = actions[np.argmax(Q[state[0], state[1]])]
            else:
                print("InvalidMethod!")
                return
            next_state = transition(state, action)
            reward = reward_map[next_state[0], next_state[1]]
            accumulated_reward += reward
            accumulated_reward_50 += reward
            accumulated_reward_list.append(accumulated_reward)
            episode_actions.append((state, action))
            # Q value updates
            Q[state[0], state[1], actions.index(action)] += \
                learning_rate * (reward + discount_factor * np.max(Q[next_state[0], next_state[1]]) \
                - Q[state[0], state[1], actions.index(action)])
            state = next_state
            if method == 'dangerAvoidance':
                if state in holes:
                    if episode_actions[-1] not in danger_list:
                        danger_list.append(episode_actions[-1])
        if (episode+1)%50 == 0: # Update every 50 episodes
            reward_per_episode.append(accumulated_reward_50/50)
            if accumulated_reward_50/50 > 0.7: # When the average reward exceeds 0.7, considered converged
                if convergence_label == 0:
                    convergence_episode = episode + 1
                    convergence_action = len(accumulated_reward_list)
                    convergence_label = 1
            accumulated_reward_50 = 0
    end_time = time.time()
    processing_time = end_time - start_time
    return Q, reward_per_episode, accumulated_reward_list, processing_time, convergence_episode, convergence_action

# Define the policy extraction function
def get_policy(holes, Q):
    ''' Based on the Q-table, return the policy
    
    Parameters
    ----------
    
    holes (`list[tuple]`): list of tuples containing the coordinates of the holes
    Q (`ndarray`): the trained Q-table of the robot

    Returns
    -------
    policy (`ndarray`): the trained policy of the robot

    '''
    policy = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            policy[i, j] = np.argmax(Q[i, j])
    for hole in holes:
        policy[hole] = -1
    policy[9,9] = 10
    return policy

def vis_policy(policy):
    ''' Return a visualized policy
    
    Parameters
    ----------
    
    policy (`ndarray`): the trained policy of the robot

    Returns
    -------
    vis_policy (`ndarray`): the visualized trained policy of the robot

    '''
    vis_policy = np.zeros((10, 10), dtype=str)
    mapping = {0: '↑', 1: '↓', 2: '←', 3: '→', -1: 'O', 10: 'G'}
    for i in range(len(policy)):
        for j in range(len(policy[0])):
            vis_policy[i,j] = mapping[policy[i,j]]
    return vis_policy

def evaluate_performance(reward_map, policy, num_episodes):
    ''' Given the reward map and the policy, evaluate the performance of the model
    
    Parameters
    ----------
    
    reward_map (`ndarray`): the reward map
    policy (`ndarray`): the trained policy of the robot
    num_episodes (`int`): the number of episodes

    Returns
    -------
    avg_reward (`float`): the average reward

    '''
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