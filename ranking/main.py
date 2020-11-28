import numpy as np
import itertools
from itertools import permutations


# Top k-Ranking 

def get_ranking(k, context, history):

# context is user and/or search string

choosen_arms = np.zeros(arm_len)
ranking = np.zeros(k)

for i = range(k):
    ranking(i) = MAB(concat(context,choosen_arms), history) # chooses an arm from the context 

    choosen_arms(ranking(i))=1


return ranking

