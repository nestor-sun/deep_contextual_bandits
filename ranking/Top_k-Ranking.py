
# Top k-Ranking 

from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

class MAB:
    def __init__(self, arms_len, max_k, mode):
        self.arms_len = arms_len
        self.max_k = max_k
        self.mode = mode
        
        self.rewards = []
        self.arms_reward = np.zeros(self.arms_len)
        self.arms_expectation = .00001*np.random.random_sample((self.arms_len,)) 
        self.arms_expectation_counts = np.zeros(self.arms_len)
        
    def choose_arms(self):
        if self.mode == 'random':
            return np.random.permutation(self.arms_len)[0:self.max_k]
        elif self.mode == 'epsilon_greedyN':
            if np.random.random_sample() < .5/(1+np.sum(self.arms_expectation_counts)):
                return np.random.permutation(self.arms_len)[0:self.max_k]
            else:
                arms_expectation_sorted_index = np.flip(np.argsort(self.arms_expectation))
                return arms_expectation_sorted_index[0:self.max_k]
        elif self.mode == 'epsilon_greedy1':
            if np.random.random_sample() < .1:
                return np.random.permutation(self.arms_len)[0:self.max_k]
            else:
                arms_expectation_sorted_index = np.flip(np.argsort(self.arms_expectation))
                return arms_expectation_sorted_index[0:self.max_k]
        elif self.mode == 'epsilon_greedy2':
            if np.random.random_sample() < .5:
                return np.random.permutation(self.arms_len)[0:self.max_k]
            else:
                arms_expectation_sorted_index = np.flip(np.argsort(self.arms_expectation))
                return arms_expectation_sorted_index[0:self.max_k]
        elif self.mode == 'epsilon_greedy3':
            if np.random.random_sample() < .9:
                return np.random.permutation(self.arms_len)[0:self.max_k]
            else:
                arms_expectation_sorted_index = np.flip(np.argsort(self.arms_expectation))
                return arms_expectation_sorted_index[0:self.max_k]
        elif self.mode == 'max_expectation':
            arms_expectation_sorted_index = np.flip(np.argsort(self.arms_expectation))
            return arms_expectation_sorted_index[0:self.max_k]
        elif self.mode == 'optimal':
            if self.max_k == 1:
                if 5 < self.arms_len:
                    return np.array([5])
                else:
                    return self.arms_len-1
            elif self.max_k == 2:
                if 6 < self.arms_len:
                    return np.array([5, 6])
                else:
                    return np.array(range(self.arms_len-2, self.arms_len))
            elif self.max_k == 3:   
                if 6 < self.arms_len:
                    return np.array([4, 5, 6])
                else:
                    return np.array(range(self.arms_len-3, self.arms_len))
            elif self.max_k == 4:
                if 7 < self.arms_len:
                    return np.array([4, 5, 6, 7])
                else:
                    return np.array(range(self.arms_len-4, self.arms_len))
            elif self.max_k == 5:
                if 7 < self.arms_len:
                    return np.array([3, 4, 5, 6, 7])
                else:
                    return np.array(range(self.arms_len-5, self.arms_len))
            elif self.max_k == 6:
                if 8 < self.arms_len:
                    return np.array([3, 4, 5, 6, 7, 8])
                else:
                    return np.array(range(self.arms_len-6, self.arms_len))
        elif 'ucb' in self.mode:

            # arms_ucb = np.zeros(self.arms_len)
            # for i in range(self.arms_len):
            #     if self.arms_expectation_counts[i]>0:
            #         arms_ucb[i] = self.arms_expectation[i] + np.sqrt(2*np.log(1+np.sum(self.arms_expectation_counts))/(self.arms_expectation_counts[i]))
            #     else:
            #         arms_ucb[i] = np.power(10.,10.)

            arms_ucb = np.where(self.arms_expectation_counts==0,
                                np.power(10.,10.),
                                self.arms_expectation + np.sqrt(2*np.log(1+np.sum(self.arms_expectation_counts))/self.arms_expectation_counts))

            # if np.max(np.abs(arms_ucb2 - arms_ucb)) > np.power(10.,-8.):
            #     print('error arms_ucb2: %9.9f' % np.max(np.abs(arms_ucb2 - arms_ucb)))
            #     return None

            arms_ucb_sorted_index = np.flip(np.argsort(arms_ucb))
            return arms_ucb_sorted_index[0:self.max_k]
        else:
            print('choose_arms unknown mode error')
    
    def set_reward(self, arms, reward):
        self.rewards.append((arms, reward))
        for arm in arms:
            arm = int(arm)
            self.arms_reward[arm] += reward
            self.arms_expectation_counts[arm] += 1
            self.arms_expectation[arm] = self.arms_reward[arm]/self.arms_expectation_counts[arm]

def test_model_reward(MAB,reward_noise,epochs):

    reward_vector = np.zeros(epochs)

    for i in range(epochs):

        arms = MAB.choose_arms()
        error = np.zeros(arms.shape[0])
        for j in range(arms.shape[0]):
            arm = arms[j]
            error[j] = np.absolute(arm - 5)
        total_error = np.sum(error)
        total_error = total_error + np.abs(np.random.randn()*reward_noise)
        reward = -total_error
        MAB.set_reward(arms, reward)
        if i>0:
            reward_vector[i] = reward + reward_vector[i-1]
        else:
            reward_vector[i] = reward

    for i in range(epochs):
        reward_vector[i] = reward_vector[i]/(i+1)

    return reward_vector

def test_model_reward_wrapper(args):
    arms = args[0]
    k = args[1]
    method = args[2]
    reward_noise = args[3]
    epochs = args[4]
    MAB_method = MAB(arms,k,method)
    return test_model_reward(MAB_method,reward_noise,epochs)

def test_plot_reward(id,arms_range,reward_noise_range,epochs,k_range,stat_sample,title_str):

    fig, ax = plt.subplots(2, 3)
    fig.tight_layout(rect = (0, 0, .82, 1))

    plot_index = 0
    for arms in arms_range:
        for reward_noise in reward_noise_range:
            for k in k_range:
                plot_index += 1

                axes_subplot = plt.subplot(2, 3, plot_index)

                for method in ['max_expectation','epsilon_greedy1','epsilon_greedy2','epsilon_greedy3','epsilon_greedyN','random','ucb','optimal']:
                    reward_method = np.zeros((stat_sample,epochs))

                    parallel_args = []
                    for j in range(stat_sample):
                        parallel_args.append([arms,k,method,reward_noise,epochs])

                    # reward_method_wrapper = list(map(test_model_reward_wrapper, iter(parallel_args)))
                    with Pool(12) as p:
                        reward_method_wrapper = list(p.map(test_model_reward_wrapper, iter(parallel_args)))
                    
                    for j in range(stat_sample):
                        reward_method[j,:] = reward_method_wrapper[j]

                    reward_method_avg = np.mean(reward_method, axis=0)
                    reward_method_std = np.std(reward_method, axis=0)

                    plt.errorbar(range(epochs),reward_method_avg,yerr=reward_method_std/5.)

                plt.title(title_str[plot_index-1])
    fig.legend(['e.-g. 1'.replace('_',' '),
                'e.-g. 0.1'.replace('_',' '),
                'e.-g. 0.5'.replace('_',' '),
                'e.-g. 0.9'.replace('_',' '),
                'e.-g. 0.5/N'.replace('_',' '),
                'random','ucb','optimal'],loc="center right",borderaxespad=0.1)
    plt.savefig('experiment'+str(id)+'.png',dpi=300)
    plt.clf()
    # plt.show()

if __name__ == '__main__':
    DEBUG = False
    if DEBUG == True:
        print('DEBUGGING')

    if DEBUG == True:
        stat_sample = 2
    else:
        stat_sample = 50

    id = 1

    arms_range = [10]
    reward_noise_range = [1/2]
    epochs = 100
    k_range_max = 6
    k_range = range(1,k_range_max+1)
    title_str = []
    for k in k_range:
        title_str.append('Top '+str(k)+'-Ranking')
    test_plot_reward(id,arms_range,reward_noise_range,epochs,k_range,stat_sample,title_str)

    id = id + 1
    arms_range = [6]
    reward_noise_range = [1/2]
    epochs = 100
    k_range_max = 6
    k_range = range(1,k_range_max+1)
    title_str = []
    for k in k_range:
        title_str.append('Top '+str(k)+'-Ranking')
    test_plot_reward(id,arms_range,reward_noise_range,epochs,k_range,stat_sample,title_str)

    id = id + 1
    arms_range = [20]
    reward_noise_range = [1/2]
    epochs = 100
    k_range_max = 6
    k_range = range(1,k_range_max+1)
    title_str = []
    for k in k_range:
        title_str.append('Top '+str(k)+'-Ranking')
    test_plot_reward(id,arms_range,reward_noise_range,epochs,k_range,stat_sample,title_str)

    id = id + 1
    arms_range = [6, 10,15,25,50,100]
    reward_noise_range = [1/2]
    epochs = 100
    k_range = [3]
    title_str = []
    for a in arms_range:
        title_str.append('Arms: '+str(a))
    test_plot_reward(id,arms_range,reward_noise_range,epochs,k_range,stat_sample,title_str)

    id = id + 1
    arms_range = [10]
    reward_noise_range = [1,2,4,6,8,10]
    epochs = 200
    k_range = [3]
    title_str = []
    for n in reward_noise_range:
        title_str.append('Noise: '+str(n))
    test_plot_reward(id,arms_range,reward_noise_range,epochs,k_range,stat_sample,title_str)
