import numpy as np

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