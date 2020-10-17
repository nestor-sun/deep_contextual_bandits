import numpy as np
import itertools
from itertools import permutations

context_list = np.zeros(9) # change to list of users in dataset
number_of_categories = 9 # change to number of categories in dataset

# print(context_list)

def expected_reward(arm,context,reward_history):
    if len(reward_history)==0:
        return 0
    else:
        r=0
        for ranking_reward in reward_history:
            r = r + ranking_reward[1]
        return r/len(reward_history)

def get_ranking(tuple_valuation):
    ranking = []
    for t in tuple_valuation:
        if t == int:
            ranking.append(t)
    return ranking

def submit_action(ranking):
    return np.random.random(1)

def average_ranking(ranking,t):
    return 1

# tuple_list = list(permutations([1,2]))
tuple_list = []
for i in range(1,number_of_categories+1):
    tuple_list.append(i)

print(tuple_list)

tuple_valuation = {}
reward_history = {}
for t in tuple_list:
    tuple_valuation[t] = 0
    reward_history[t] = []

print(tuple_valuation)


for context in context_list:
    for t in tuple_list:
        tuple_valuation[t] = expected_reward(t,context,reward_history[t])
    pass
    ranking = get_ranking(tuple_valuation)
    reward = submit_action(ranking)
    for t in tuple_list:
        reward_history[t].append([average_ranking(ranking,t),reward])

