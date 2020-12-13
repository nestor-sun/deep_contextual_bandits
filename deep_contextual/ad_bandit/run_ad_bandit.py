from DeepFPL import DeepFPL
import data_utils

import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np

# Extract regret history
def extract_results(mab_alg):
    # mab_alg.regret_history # (n_episodes x n_iterations) (numpy array)
    _, n_iterations = mab_alg.regret_history.shape
    results = {"regret": {}, "avg_reward": {}}
    results["regret"]["mean"] = mab_alg.regret_history.mean(axis=0)
    results["regret"]["std"] = mab_alg.regret_history.std(axis=0)
    avg_rew_all = mab_alg.reward_history.cumsum(axis=1) / np.arange(1, n_iterations+1)
    results["avg_reward"]["mean"] = avg_rew_all.mean(axis=0)
    results["avg_reward"]["std"] = avg_rew_all.std(axis=0)
    
    return results
    
# Instantiate MAB with given keyword args
# dim_ads: dimensionality of ads
# dim_users: dimensionality of users
def run_mab_alg(n_episodes, n_iterations, dim_ads, dim_users, mab_alg_class, **kwargs):
    ad_feat = data_utils.read_ad_features(target_dim=dim_ads, w_cats=True) # (n_ads x n_ad_features)
    user_feat = data_utils.read_user_features(target_dim=dim_users) # (n_users x n_user_features)
    ad_ratings = data_utils.read_ad_ratings() # (n_users x n_ads)
        
    # Run MAB
    mab_alg = mab_alg_class(n_episodes, n_iterations,
                            ad_feat, user_feat, ad_ratings,
                            **kwargs)
    mab_alg.run()
    return mab_alg

# Batch run a series of algorithms to compare.
# output_folder: folder to output results to
def compare_algs(n_episodes, n_iterations, dim_ads, dim_users, alg_list, output_folder):
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    # Initialize result placeholders
    n_algs = len(alg_list)
    all_results = {"regret": {}, "avg_reward": {}}
    for key in all_results.keys():
        all_results[key]["mean"] = np.zeros((n_algs, n_iterations))
        all_results[key]["std"] = np.zeros((n_algs, n_iterations))
    # Run algorithms
    
    for idx, alg in enumerate(alg_list):
        alg_params = {key: val for key, val in alg.items() 
                      if key not in ["label", "class"]}
        mab_alg = run_mab_alg(n_episodes, n_iterations, dim_ads, dim_users,
                              alg["class"], **alg_params)
        new_results = extract_results(mab_alg)
        for key in all_results.keys():
            for field in all_results[key].keys():
                all_results[key][field][idx, :] = new_results[key][field]
        #regret[idx, :], stds[idx, :] = extract_regret(mab_alg)

    # Plot regret and reward
    for key in all_results.keys():
        fig, ax = plt.subplots()
        t_range = range(n_iterations)
        for idx, alg in enumerate(alg_list):
            this_mean = all_results[key]["mean"][idx, :]
            this_std = all_results[key]["std"][idx, :]
            ax.plot(t_range, this_mean, label=alg["label"])
            ax.fill_between(t_range, this_mean - this_std, 
                            this_mean + this_std, alpha=0.35)
        plt.xlabel("t")
        plt.ylabel(key.replace("_", " ").title())
        plt.legend()
        plt.show()
        
        # Save curve
        label_list = [alg["label"] for alg in alg_list]
        for field in all_results[key].keys():
            df = pd.DataFrame(all_results[key][field].T, columns=label_list)
            df.index.name = "t"
            df.to_csv(path / (field + ".csv"))
    return ax



#%% Sample Usage
n_episodes = 10
n_iterations = 10000
dim_ads = 16
dim_users = 40

# Running a single algorithm
#deep_fpl = run_mab_alg(n_episodes, n_iterations, dim_ads, dim_users, 
#                   DeepFPL, n_exp_rounds=10)
#reg, std = extract_regret(deep_fpl)

# Comparing multiple algorithms

for bandit_noise in [0.1, 2, 4]:
    alg_list = [{"label": "DeepFPL a={}".format(a),
                "class": DeepFPL,
                "n_exp_rounds": 50,
                "lr": 1e-2,
                "a": a,
                "bandit_noise": bandit_noise,
                "hidden_layers": [40, 40]} for a in [0, 2, 4]]
    noise_str = str(bandit_noise).replace(".", "_")
    folder_name = "bnoise_{}".format(noise_str)
    compare_algs(n_episodes, n_iterations, dim_ads, dim_users, alg_list, folder_name)
    plt.title("DeepFPL, bandit noise={}".format(bandit_noise))