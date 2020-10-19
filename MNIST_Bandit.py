# MNIST Bandits
# Use generate_bandits to generate a series of bandit experiments and create a
# file.
#
# Use load_bandits to load the arms and rewards for a pregenerated set of
# bandits

from tensorflow import keras
import numpy as np


# Returns a generator for a bandit instance
#
# You can iterate using a bandit algorithm as follows:
#
# for arms, rewards in load_bandits("my_bandits.npz"):
#     # Run a trial
#     for round_arms, round_rewards in zip(arms, rewards):
#         # Pick an arm for a round and sample the reward. round_arms has shape
#         # (n_arms, 28, 28)
#         # Choose an arm to play
#         arm_idx = 7
#         # Get arm reward
#         sampled_reward = round_rewards[arm_idx]
#         break
#     break
#
# arms has shape [n_rounds, n_arms, 28, 28]
# rewards has shape [n_rounds, n_arms]
def load_bandits(filename):
    # Load
    with open(filename, 'rb') as f:
        image_indeces = np.load(f)
        rewards = np.load(f)
    
    # Yield bandits, one instance at a time
    x_train, y_train = load_data()
    for i in range(image_indeces.shape[0]):
        arms = x_train[image_indeces[i, :, :].squeeze(), :, :]
        arm_rewards = rewards[i, :, :].squeeze()
        yield arms, arm_rewards
        
        
# Generate a bandit instance and save to specified folder. Arguments:
# folder: path to save file
# n_trials: Number of bandit instances to simulate
# n_rounds: Number of rounds per bandit instance
# n_arms: Arms per round
# bernoulli_pos: Rewards for positive label
# bernoulli_neg: Rewards for negative labels
# replacement: Use replacement (per round)? If false, will take much longer to
#               generate
def generate_bandits(filename, n_trials=500, n_rounds=10000, n_arms=10,
                    bernoulli_pos=0.25, bernoulli_neg=0.75, w_replacement=True):
    # Generate random bandits
    x_train, y_train = load_data()
    labels = range(10)
    assert n_trials % len(labels) == 0
    n_ims = x_train.shape[0]
    
    # Select images (with replacement: can get repeat images in round)
    if w_replacement:
        image_indeces = np.random.randint(n_ims, size=(n_trials, n_rounds, n_arms))
    # No replacement (hangs)
    else:
        image_indeces = np.zeros((n_trials, n_rounds, n_arms))
        for i in range(n_trials):
            for j in range(n_rounds):
                image_indeces[i, j, :] = np.random.choice(n_ims, size=(n_arms),
                                                          replace=False)
    
    # Generate rewards
    classes = y_train[image_indeces]
    rewards = np.zeros(classes.shape)
    trials_per_label = round(n_trials / len(labels))
    for i in range(len(labels)):
        start_idx = i * trials_per_label
        end_idx = start_idx + trials_per_label
        classes_chunk = classes[start_idx : end_idx, :, :]
        mask = classes_chunk == i
        probs = np.zeros(classes_chunk.shape)
        probs[mask] = bernoulli_pos
        probs[~mask] = bernoulli_neg
        rewards[start_idx : end_idx, :, :] = np.random.binomial(1, probs)
    
    # Save
    with open(filename, 'wb') as f:
        np.save(f, image_indeces)
        np.save(f, rewards)
        
# Load MNIST data
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train
    