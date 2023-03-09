import numpy as np
from libraries.utility import first_visit_monte_carlo_control
from libraries.utility import sarsa
from libraries.utility import q_learning
import pandas as pd
import os
# Define the state-space and action-space
# states = [(i, j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']

holes = [(3, 3),(1, 3),(1, 1),(3, 0),(2, 3)]

# Define the reward function
reward_map = np.zeros((4, 4))
reward_map[1, 1] = -1
reward_map[1, 3] = -1
reward_map[2, 3] = -1
reward_map[3, 0] = -1
reward_map[3, 3] = 1

time_mc_total = 0
time_sarsa_total = 0
time_q_total = 0

convergingEpisode_mc_total = 0
convergingEpisode_sarsa_total = 0
convergingEpisode_q_total = 0

convergingAction_mc_total = 0
convergingAction_sarsa_total = 0
convergingAction_q_total = 0

#Options: 'None', 'randomArgmax', 'q_reset', 'combined'
method = 'randomArgmax'

time = 30
episode = 10000
for i in range(time):
    _, _, _ , time_mc, convergingEpisode_mc, convergingAction_mc = first_visit_monte_carlo_control(holes, reward_map, actions, episode, 0.9, 0.1, method=method)
    _, _, _, time_sarsa, convergingEpisode_sarsa, convergingAction_sarsa = sarsa(holes, reward_map, actions, episode, 0.9, 0.1, 0.1, method=method)
    _, _, _, time_q, convergingEpisode_q, convergingAction_q = q_learning(holes, reward_map, actions, episode, 0.9, 0.1, 0.1, method=method)
    
    time_mc_total += time_mc
    time_sarsa_total += time_sarsa
    time_q_total += time_q

    convergingEpisode_mc_total += convergingEpisode_mc
    convergingEpisode_sarsa_total += convergingEpisode_sarsa
    convergingEpisode_q_total += convergingEpisode_q

    convergingAction_mc_total += convergingAction_mc
    convergingAction_sarsa_total += convergingAction_sarsa
    convergingAction_q_total += convergingAction_q

print(f"avg training time for mc: {time_mc_total/time}")
print(f"avg training time for sarsa: {time_sarsa_total/time}")
print(f"avg training time for q: {time_q_total/time}")

data = {
    'mc': {'avg_time': time_mc_total/time, 'episode': convergingEpisode_mc_total/time, 'action': convergingAction_mc_total/time},
    'sarsa': {'avg_time': time_sarsa_total/time, 'episode': convergingEpisode_sarsa_total/time, 'action': convergingAction_sarsa_total/time},
    'q': {'avg_time': time_q_total/time, 'episode': convergingEpisode_q_total/time, 'action': convergingAction_q_total/time}
}

dataframe = pd.DataFrame.from_dict(data, orient='index')

# Write the results into a CSV file
result_path = os.path.join("results")
if not os.path.exists(result_path):
    os.mkdir(result_path)
    print("Target directory: {} Created".format(result_path))
dataframe.to_csv("results/TestResult_{}_task1.csv".format(method), sep=',')
print("Results saved to:", "results/TestResult_{}_task1.csv".format(method))
