import json
import numpy as np
import pandas as pd

var = 2

data_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/experiment_output/'
ts_filename = 'ts_reward_list_{}_per_var.json'.format(str(var))
ucb_filename = 'ucb_reward_list_{}_per_var.json'.format(str(var))
epsilon_filename = 'epsilon_reward_list_{}_per_var.json'.format(str(var))

ts_reward_list = json.load(open(data_path + ts_filename, 'r'))
ucb_reward_list = json.load(open(data_path + ucb_filename, 'r'))
epsilon_reward_list = json.load(open(data_path + epsilon_filename, 'r'))

num_of_iters = 10000
num_of_episodes = 10


def get_mean_and_std(reward_list):
    mean_list = []
    std_list = []
    for it in range(num_of_iters):
        reward_list_episodes = [reward_list[ep][it] for ep in range(num_of_episodes)]
        mean_list.append(np.mean(reward_list_episodes))
        std_list.append(np.std(reward_list_episodes)/(it+1))
    return mean_list, std_list


def get_cumulative_reward_list(reward_list):
    cumulative_reward_list = []
    for episode in reward_list:
        episode_list = []
        for row_num, each_round in enumerate(episode):
            if row_num == 0:
                episode_list.append(each_round)
            else:
                episode_list.append(each_round + episode_list[row_num - 1])
        cumulative_reward_list.append(episode_list)
    return cumulative_reward_list


ts_mean_list, ts_std_list = get_mean_and_std(get_cumulative_reward_list(ts_reward_list))
ucb_mean_list, ucb_std_list = get_mean_and_std(get_cumulative_reward_list(ucb_reward_list))
epsilon_mean_list, epsilon_std_list = get_mean_and_std(get_cumulative_reward_list(epsilon_reward_list))


mean_list = pd.DataFrame({'TS': ts_mean_list, 'UCB': ucb_mean_list, 'Epsilon_Greedy': epsilon_mean_list})
std_list = pd.DataFrame({'TS': ts_std_list, 'UCB': ucb_std_list, 'Epsilon_Greedy': epsilon_std_list})

mean_list.to_csv(data_path + 'mean_{}_var.csv'.format(str(var)), index=False)
std_list.to_csv(data_path + 'std_{}_var.csv'.format(str(var)), index=False)


