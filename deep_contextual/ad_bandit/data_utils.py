import numpy as np
import pathlib
from sklearn.decomposition import PCA

data_folder = "data"

# Return concatenated ad features and user features
# Input shapes (n_users, n_user_feat), (n_ads, n_ad_feat)
# Output shape (n_users, n_ads, n_user_feat + n_ad_feat)
#
# Tensor for which entry [i, j, :] represents the ith user features concatenated
# with the ad features (in that order)
def concat_usrs_ads_ft(user_feat, ad_feat):
    n_users, n_user_features = user_feat.shape
    n_ads, n_ad_features = ad_feat.shape
    
    concat_feat = np.zeros((n_users, n_ads, n_ad_features + n_user_features))
    concat_feat[:, :, :n_user_features] = user_feat[:, None, :]
    concat_feat[:, :, n_user_features:] = ad_feat[None, :, :]
    return concat_feat

## Test for ad_features_concat
#n_ads = 150
#n_ad_features = 400
#n_users = 80
#n_user_features = 10

#ad_feat = np.random.rand(n_ads, n_ad_features)
#user_feat = np.random.rand(n_users, n_user_features)
#ad_ratings = np.random.rand(n_users, n_ads)

#concat_feat = concat_usrs_ads_ft(user_feat, ad_feat)

## Data reading utilities
def load_array(filename, target_dim=None):
    path = pathlib.Path(data_folder) / filename
    data = np.load(path)
    
    if target_dim is not None:
        _, orig_dim = data.shape
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(data)
        recons = pca.inverse_transform(reduced)
        recons_error = ((recons - data)**2).mean()
        
        print("Reducing dimensionality for {}".format(filename))
        print("Going from {} to {}".format(orig_dim, target_dim))
        print("Reconstruction error: {}".format(recons_error))
        data = reduced
    return data

#(n_ads x n_ad_features)
def read_ad_features(w_cats=True, target_dim=None):
    if w_cats:
        filename = "ad_feat_with_one_hot_encoding.npy"
    else:
        filename = "ad_feat_without_one_hot_encoding.npy"
    return load_array(filename, target_dim=target_dim)

# (n_users x n_user_features)
def read_user_features(target_dim=None): 
    return load_array("user_feat.npy", target_dim=target_dim)

# (n_users x n_ads)
def read_ad_ratings():
    return load_array("ad_ratings.npy")