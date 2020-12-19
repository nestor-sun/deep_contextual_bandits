import pandas as pd
import matplotlib.pyplot as plt


path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/experiment_output/'
mean_var_1 = pd.read_csv(path + 'mean_1_var.csv')
std_var_1 = pd.read_csv(path + 'std_1_var.csv')
mean_var_2 = pd.read_csv(path + 'mean_2_var.csv')
std_var_2 = pd.read_csv(path + 'std_2_var.csv')

fig, ax = plt.subplots()
t_range = range(len(mean_var_1))


for col_name in mean_var_1.columns:
    this_mean = mean_var_1[col_name].values
    this_std = std_var_1[col_name].values
    ax.plot(t_range, this_mean, label=col_name + '_var_1')
    ax.fill_between(t_range, this_mean - this_std,
                    this_mean + this_std, alpha=0.35)

    this_mean = mean_var_2[col_name].values
    this_std = std_var_2[col_name].values
    ax.plot(t_range, this_mean, label=col_name + '_var_2')
    ax.fill_between(t_range, this_mean - this_std,
                    this_mean + this_std, alpha=0.35)

plt.xlabel("t")
plt.ylabel("Cumulative mean reward, avg")
plt.legend()
plt.show()





