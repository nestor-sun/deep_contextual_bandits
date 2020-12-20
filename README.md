# Deep Contextual Bandits

## Ranking
For ranking plots, run 
```python ranking/Top_k-Ranking.py```

The plots will be saved in your directory.


## DeepFPL
Code for DeepFPL located in: deep_contextual_bandits/deep_contextual/ad_bandit/

 - run_ad_bandit.py (driver script)
 - data_utils.py (data utilities)
 - Plotter.py (results plotter)
 - DeepFPL.py (DeepFPL algorithm)

## Mushroom Bandit and Feature extraction with ResNet50 

Code for comparing Epsilon-greedy network with a random strategy
 - Run pytorch-Mushroom.py (dependencies pytorch and pandas). It will plot the edible mushrooms eaten as a function of the number of rounds. Number of mushrooms eaten per round can be changed by changing the variable K. Requires access to the mushroom dataset from UCI ML repositry.
 - resNet_img.py extracts image features for an input image using pretrained ResNet50 model and saves the features in a local directory.
