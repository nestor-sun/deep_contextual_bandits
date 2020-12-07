from DeepFPL import DeepFPL
import data_utils

import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np

# Extract regret history
def extract_regret(mab_alg):
    # mab_alg.regret_history # (n_episodes x n_iterations) (numpy array)
    avg_reg = mab_alg.regret_history.mean(axis=0)
    std = mab_alg.regret_history.std(axis=0)
    return avg_reg, std
    
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
    
    # Run algorithms
    n_algs = len(alg_list)
    regret = np.zeros((n_algs, n_iterations))
    stds = np.zeros((n_algs, n_iterations))
    for idx, alg in enumerate(alg_list):
        alg_params = {key: val for key, val in alg.items() 
                      if key not in ["label", "class"]}
        mab_alg = run_mab_alg(n_episodes, n_iterations, dim_ads, dim_users,
                              alg["class"], **alg_params)
        regret[idx, :], stds[idx, :] = extract_regret(mab_alg)

    # Plot regret        
    fig, ax = plt.subplots()
    t_range = range(n_iterations)
    for idx, alg in enumerate(alg_list):
        this_regret = regret[idx, :]
        this_std = stds[idx, :]
        ax.plot(t_range, this_regret, label=alg["label"])
        ax.fill_between(t_range, this_regret - this_std, 
                        this_regret + this_std, alpha=0.35)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()
    
    # Save regret curve
    label_list = [alg["label"] for alg in alg_list]
    for name, array in zip(["regret", "stds"], [regret, stds]):
        df = pd.DataFrame(array.T, columns=label_list)
        df.index.name = "t"
        df.to_csv(path / (name + ".csv"))
    return ax



#%% Sample Usage
n_episodes = 2
n_iterations = 10000
dim_ads = 16
dim_users = 40

# Running a single algorithm
#deep_fpl = run_mab_alg(n_episodes, n_iterations, dim_ads, dim_users, 
#                   DeepFPL, n_exp_rounds=10)
#reg, std = extract_regret(deep_fpl)

# Comparing multiple algorithms
alg_list = [{"label": "DeepFPL_a={}".format(a),
            "class": DeepFPL,
            "n_exp_rounds": 50,
            "lr": 1e-3,
            "a": a,
            "hidden_layers": [40, 40]} for a in [0]]
compare_algs(n_episodes, n_iterations, dim_ads, dim_users, alg_list, "alg_testing")