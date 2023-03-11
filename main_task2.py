import numpy as np
import matplotlib.pyplot as plt
from libraries.utility_extended import first_visit_monte_carlo_control
from libraries.utility_extended import sarsa
from libraries.utility_extended import q_learning
from libraries.utility_extended import get_policy
from libraries.utility_extended import vis_policy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='exploringStart', help="Options: 'None', 'randomArgmax', 'q_reset', 'combined', 'dangerAvoidance', 'exploringStart'")
args = parser.parse_args()

method = args.method # You can customize here. Options: 'None', 'randomArgmax', 'q_reset', 'combined', 'dangerAvoidance', 'exploringStart'

# Define the action-space
actions = ['up', 'down', 'left', 'right']

# Define the holes
holes = [(0, 5),(1, 1),(1, 4),(1, 8),(2, 4),(2, 5),(2, 8),(3, 3),(3, 5),(3, 9),(4, 0),(4, 1),(4, 4),(4, 7),(4, 8),(5, 1),(5, 7),(5, 9),(6, 3),(7, 4),(7, 9),(8, 4),(8, 6),(9, 2),(9, 6)]
# Define the reward function
reward_map = np.zeros((10, 10))
for r, c in holes:
    reward_map[r, c] = -1
reward_map[9, 9] = 1 # Reward of the goal is 1

# Train the Monte Carlo first-visit control
Q_mc, reward_per_episode_mc, accumulated_reward_mc, time_mc, convergingEpisode_mc, convergingAction_mc =\
     first_visit_monte_carlo_control(holes, reward_map, actions, 30000, 1, 0.5, method=method)
actions_mc = range(len(accumulated_reward_mc))

# Train the SARSA with an ε-greedy behavior policy
Q_sarsa, reward_per_episode_sarsa, accumulated_reward_sarsa, time_sarsa, convergingEpisode_sarsa, convergingAction_sarsa = \
    sarsa(holes, reward_map, actions, 30000, 0.9, 0.1, 0.1, method=method)
actions_sarsa = range(len(accumulated_reward_sarsa))

# Train the Q-learning with an ε-greedy behavior policy
Q_q, reward_per_episode_q, accumulated_reward_q, time_q, convergingEpisode_q, convergingAction_q = \
    q_learning(holes, reward_map, actions, 30000, 0.9, 0.1, 0.1, method=method)
actions_q = range(len(accumulated_reward_q))

# To align the three arrays with different lengths for visualization, expand those that are shorter with 0.
actions = max(len(actions_sarsa),len(actions_q),len(actions_mc))

if actions == len(actions_sarsa):
    accumulated_reward_q.extend([0]*(actions - len(actions_q)))
    accumulated_reward_mc.extend([0]*(actions - len(actions_mc)))
elif actions == len(actions_mc):
    accumulated_reward_sarsa.extend([0]*(actions - len(actions_sarsa)))
    accumulated_reward_q.extend([0]*(actions - len(actions_q)))
else:
    accumulated_reward_mc.extend([0]*(actions - len(actions_mc)))
    accumulated_reward_sarsa.extend([0]*(actions - len(actions_sarsa)))
actions = range(actions)

# Get the policies
policy_mc = get_policy(holes, Q_mc)
policy_sarsa = get_policy(holes, Q_sarsa)
policy_q = get_policy(holes, Q_q)
# Visualize the policies represented in arrows, which are more intuitive
arrow_policy_mc = vis_policy(policy_mc)
arrow_policy_sarsa = vis_policy(policy_sarsa)
arrow_policy_q = vis_policy(policy_q)

# Results printing
print(arrow_policy_mc)
print(f"Processing time for First-visit Monte Carlo Control: {time_mc}")
print(f"Number of episodes to converge for MC: {convergingEpisode_mc}")
print(arrow_policy_sarsa)
print(f"Processing time for SARSA: {time_sarsa}")
print(f"Number of episodes to converge for SARSA: {convergingEpisode_sarsa}")
print(arrow_policy_q)
print(f"Processing time for Q-learning: {time_q}")
print(f"Number of episodes to converge for Q: {convergingEpisode_q}")

# Plot the results
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(actions, accumulated_reward_mc, label = 'Accumulated reward of First-visit Monte Carlo')
ax1.plot(actions, accumulated_reward_sarsa, label = 'Accumulated reward of SARSA')
ax1.plot(actions, accumulated_reward_q, label = 'Accumulated reward of Q-learning')
ax1.legend()

ax2.plot(range(len(reward_per_episode_mc)), reward_per_episode_mc, label = 'Reward per episode of First-visit Monte Carlo')
ax2.plot(range(len(reward_per_episode_sarsa)), reward_per_episode_sarsa, label = 'Reward per episode of SARSA')
ax2.plot(range(len(reward_per_episode_q)), reward_per_episode_q, label = 'Reward per episode of Q-learning')
ax2.legend()
plt.show()