# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:01:49 2020

@author: Jonathan
"""

import pandas as pd
import matplotlib.pyplot as plt

def format_df(df, col_list):
    if "t" in df.columns:
        df.set_index("t", inplace=True)
    else:
        df.index.name = "t"
        df = df.divide(range(len(df)), axis="index")
    if col_list is not None:
        df = df[[col for col in df.columns if col in col_list]]
    return df

def consolidate_results(folders, col_list=None):
    means = []
    stds = []
    for folder in folders:
        means.append(format_df(pd.read_csv(folder + "/mean.csv"), col_list))
        stds.append(format_df(pd.read_csv(folder + "/std.csv"), col_list))
    means = pd.concat(means, axis=1)
    stds = pd.concat(stds, axis=1)
    
    # plot
    fig, ax = plt.subplots()
    t_range = range(len(means))
    for col_name in means.columns:
        this_mean = means[col_name].values
        this_std = stds[col_name].values
        ax.plot(t_range, this_mean, label=col_name)
        ax.fill_between(t_range, this_mean - this_std, 
                        this_mean + this_std, alpha=0.35)
    plt.xlabel("t")
    plt.ylabel("Cumulative mean reward, avg")
    plt.legend()
    plt.show()
        
consolidate_results(["mingwei", "bnoise_2"], 
                    col_list=["TS", "UCB", "Epsilon_Greedy", "DeepFPL a=0"])