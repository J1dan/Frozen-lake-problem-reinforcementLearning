import numpy as np

reward_map = np.zeros((10, 10))
reward_map[1, 1], reward_map[1, 4], reward_map[1, 8] = -1, -1, -1
reward_map[2, 0], reward_map[2, 4], reward_map[2, 5], reward_map[2, 8] = -1, -1, -1, -1
reward_map[3, 3], reward_map[3, 5], reward_map[3, 9] = -1, -1, -1
reward_map[4, 0], reward_map[4, 1], reward_map[4, 4], reward_map[4, 7], reward_map[4, 8]  = -1, -1, -1, -1, -1
reward_map[5, 1], reward_map[5, 7], reward_map[5, 9] = -1, -1, -1 
reward_map[6, 3] = -1
reward_map[7, 6], reward_map[7, 9] = -1, -1
reward_map[8, 4], reward_map[8, 6] = -1, -1
reward_map[9, 2], reward_map[9, 6] = -1, -1
reward_map[9, 9] = 1
indices = np.where(reward_map == -1)
state = (1,1)

