import numpy as np
import matplotlib.pyplot as plt
from libraries.utility import first_visit_monte_carlo_control
from libraries.utility import sarsa
from libraries.utility import q_learning
from libraries.utility import get_policy
from libraries.utility import vis_policy
from libraries.utility import evaluate_performance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='combined', help="Options: 'None', 'randomArgmax', 'q_reset', 'combined'")
args = parser.parse_args()

method = args.method # You can customize here

# Define the action-space
actions = ['up', 'down', 'left', 'right']

# Define the holes
holes = [(3, 3),(1, 3),(1, 1),(3, 0),(2, 3)]

# Define the reward map where the rewards of holes are -1, and the reward of the frisbee is +1
reward_map = np.zeros((4, 4))
for r, c in holes:
    reward_map[r, c] = -1
reward_map[3, 3] = 1

# method = 'combined' # Options: 'None', 'randomArgmax', 'q_reset', 'combined'

# Train the Monte Carlo first-visit control
Q_mc, reward_per_episode_mc, accumulated_reward_mc, time_mc, convergingEpisode_mc, convergingAction_mc =\
     first_visit_monte_carlo_control(holes, reward_map, actions, 5000, 0.9, 0.1, method=method)
actions_mc = range(len(accumulated_reward_mc))

# Train the SARSA with an ε-greedy behavior policy
Q_sarsa, reward_per_episode_sarsa, accumulated_reward_sarsa, time_sarsa, convergingEpisode_sarsa, convergingAction_sarsa =\
     sarsa(holes, reward_map, actions, 5000, 0.9, 0.1, 0.1, method=method)
actions_sarsa = range(len(accumulated_reward_sarsa))

# Train the Q-learning with an ε-greedy behavior policy
Q_q, reward_per_episode_q, accumulated_reward_q, time_q, convergingEpisode_q, convergingAction_q =\
     q_learning(holes, reward_map, actions, 5000, 0.9, 0.1, 0.1, method=method)
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


print(arrow_policy_mc)
print(f"Processing time for First-visit Monte Carlo Control: {time_mc}")

print(arrow_policy_sarsa)
print(f"Processing time for SARSA: {time_sarsa}")

print(arrow_policy_q)
print(f"Processing time for Q-learning: {time_q}")

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