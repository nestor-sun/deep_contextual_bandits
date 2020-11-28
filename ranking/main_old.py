import numpy as np
import itertools
from itertools import permutations

context_list = np.zeros(9) # change to list of users in dataset
number_of_categories = 9 # change to number of categories in dataset

print(context_list)

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
    for t in sorted(tuple_valuation, key=tuple_valuation.__getitem__):
        if isinstance(t, int):
            ranking.append(t)
    return ranking

def submit_action(ranking):
    return np.random.random(1)

def average_ranking(ranking,t):
    if isinstance(t, int):
        return ranking.index(t)+1
    else:
        tuple_ranks = []
        for o in t:
            tuple_ranks.append(ranking.index(o)+1)
        return np.mean(tuple_ranks)

# tuple_list = list(permutations([1,2]))
tuple_list = []
for i in range(1,number_of_categories+1):
    tuple_list.append(i)
for i in range(1,number_of_categories+1):
    for j in range(i+1,number_of_categories+1):
        tuple_list.append((i,j))

print(tuple_list)

tuple_valuation = {}
reward_history = {}
for t in tuple_list:
    tuple_valuation[t] = 0
    reward_history[t] = []

print(tuple_valuation)
print(reward_history)

for context in context_list:
    for t in tuple_list:
        tuple_valuation[t] = expected_reward(t,context,reward_history[t])
    print('tuple_valuation')
    print(tuple_valuation)

    ranking = get_ranking(tuple_valuation)

    print('ranking')
    print(ranking)
    reward = submit_action(ranking)
    print('reward')
    print(reward)
    for t in tuple_list:
        # print(t)
        reward_history[t].append([average_ranking(ranking,t),reward])
        # print(average_ranking(ranking,t))
    # break
    # print('reward_history')
    # print(reward_history)

