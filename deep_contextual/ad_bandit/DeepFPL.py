import torch
import numpy as np
from data_utils import concat_usrs_ads_ft

# Follow the perturbed leader
class DeepFPL():
    # n_episodes
    # n_iterations
    # ad_feat
    # n_rounds_exp # Number of exploratory rounds
    # a # Perturbation parameter
    # max_perturbation # Highest perturbation (multiples of a)
    # train_batch_size # Training batch size for model
    # lr # Learning rate
    # model_arch : model architecture (PyTorch class)
    #    
    def __init__(self, n_episodes, n_iterations, ad_feat, user_feat, ad_ratings,
                 n_rounds_exp=28, a=1.5, train_batch_size=32, lr=1e-3,
                 model_arch=None, max_perturbation=2):
        # Copy arguments as class properties
        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds and object's attributes
        del self.__dict__["self"] # don't need `self`
        
        # Other useful properties
        self.n_users, _ = user_feat.shape
        self.n_arms, _ = ad_feat.shape
        self.user_ad_feat = concat_usrs_ads_ft(user_feat, ad_feat)
        self.n_feat_user_ad = self.user_ad_feat.shape[2]
        self.regret_history = np.zeros((self.n_episodes, self.n_iterations))
        
        
    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Run an episode
        for ep_idx in range(self.n_episodes):
            # Placeholders for observations to be used for training
            inputs = np.zeros((self.n_iterations, self.n_feat_user_ad))
            rewards = np.zeros((self.n_iterations, 1))
            
            # Model initialization
            model = self.model_arch()
            model.to(device)
            criterion = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            # Time step
            for t in range(self.n_iterations):
                # Choose a random user
                user_idx = np.random.choice(self.n_users)
                
                # Update model based on perturbed history
                if t >= self.train_batch_size:
                    # Pick a random sample of observations and add noise
                    sample_idcs = np.random.randint(t, size=self.train_batch_size)
                    inp_train = inputs[sample_idcs]
                    perturbation = np.random.randn(self.train_batch_size) ### REVISIT
                    perturbation = self.a*np.minimum(np.maximum(perturbation, 
                                    -self.max_perturbation), self.max_perturbation)
                    rew_train = self.rewards[sample_idcs] + perturbation
                    
                    # Train model
                    model.train()
                    rew_pred = model(torch.as_tensor(inp_train, device=device))
                    loss = criterion(y_pred, torch.as_tensor(rew_train, device=device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Exploration phase
                if t < self.n_rounds_exp or t < self.train_batch_size:
                    played_arm = np.random.randint(self.n_arms)
                # Greedy phase
                else:
                    model.eval()
                    rewards_pred = model(torch.as_tensor(
                                        user_ad_feat[user_idx, :, :]), 
                                        device=device)
                    played_arm = torch.argmax(rewards_pred, dim=0)
                
                # Play the arm and update the history
                true_rating = self.ratings[user_idx, played_arm]
                reward = some_random_rew(true_rating) ### REVISIT
                self.rewards[t] = reward
                max_reward = self.ratings[user_idx, :].max()
                self.regret_history[ep_idx, t] = max_reward - true_rating ### REVISIT
                inputs[t, :] = self.user_ad_feat[user_idx, played_arm, :]
                    
        # Cumulative regret
        self.regret_history = np.cumsum(self.regret_history, axis=1)