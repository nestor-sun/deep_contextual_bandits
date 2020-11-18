import DeepFPL

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
def run_mab_alg(n_episodes, n_iterations, mab_alg_class, **kwargs):
    ad_feat = read_ad_features() # (n_ads x n_ad_features)
    user_feat = read_user_features() # (n_users x n_user_features)
    ad_ratings = read_ad_ratings() # (n_users x n_ads)
        
    # Run MAB
    mab_alg = mab_alg_class(n_episodes, n_iterations,
                            ad_feat, user_feat, ad_ratings,
                            **kwargs)
    mab_alg.run()
    return mab_alg

# Batch run a series of algorithms to compare.
# output_folder: folder to output results to
def compare_algs(n_episodes, n_iterations, alg_list, output_folder):
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True)
    
    # Run algorithms
    n_algs = len(alg_list)
    regret = np.zeros((n_algs, n_iterations))
    stds = np.zeros((n_algs, n_iterations))
    for idx, alg in enumerate(alg_list):
        alg_params = {key: val for key, val in alg.items() 
                      if key not in ["label", "class"]}
        mab_alg = run_mab_alg(n_episodes, n_iterations, alg["class"],
                              **alg_params)
        regret[idx, :], std[idx, :] = extract_regret(mab_alg)

    # Plot regret        
    fig, ax = plt.subplots()
    t_range = range(n_iterations)
    for idx, alg in enumerate(alg_list):
        this_regret = reg[idx, :]
        this_std = stds[idx, :]
        ax.plot(t_range, this_regret, label=alg["label"])
        ax.fill_between(this_regret, this_regret - this_std, 
                        this_regret + this_std, alpha=0.35)
    plt.legend()
    plt.show()
    
    # Save regret curve
    label_list = [alg["label"] for alg in alg_list]
    for name, array in zip(["regret", "stds"], [regret, stds]):
        df = pd.DataFrame(array, columns=label_list)
        df.index.name = "t"
        df.to_csv(path / (name + ".csv"))
    return ax



#%% Sample Usage
n_episodes = 10
n_iterations = 1000

# Running a single algorithm
deep_fpl = run_mab_alg(n_episodes, n_iterations, DeepFPL, n_exp_rounds=10)
reg, std = extract_regret(deep_fpl)

# Comparing multiple algorithms
alg_list = [{"label": "DeepFPL",
             "class": DeepFPL,
             "n_exp_rounds": 10},
            {"label": "AnotherAlg",
             "class": SomeAlg,
             "some_param": 7}]
compare_algs(n_episodes, n_iterations, alg_list, "alg_testing")