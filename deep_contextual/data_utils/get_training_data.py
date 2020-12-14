import json
import numpy as np
from sklearn.decomposition import PCA

part_1_or_part_2 = 2

if part_1_or_part_2 == 1:
    user_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/user/'
    img_feat_filename = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/img_feat.json'
    data_top_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/'
else:
    user_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/user/'
    img_feat_filename = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/img_feat.json'
    data_top_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/'


def one_hot(category):
    num_category = 20
    init = [0 for _ in range(num_category)]
    init[int(category)] = 1
    return init


img_feats = json.load(open(img_feat_filename))
feature_list = []
rating_list = []
max_rating_for_each_user = {}

user_cat_img_index = []

for usr in range(1, 61):
    if part_1_or_part_2 == 2:
        usr = usr + 60
        if usr <= 9:
            user_id = 'U000' + str(usr)
        elif usr >= 100:
            user_id = 'U0' + str(usr)
        else:
            user_id = 'U00' + str(usr)

    # print(usr)
    else:
        if usr <= 9:
            user_id = 'U000' + str(usr)

        else:
            user_id = 'U00' + str(usr)


    feat_filename = user_id + '_feat.json'
    rating_filename = user_id + '_rating.json'

    user_feat = json.load(open(user_path + feat_filename, 'r'))
    user_ratings = json.load(open(user_path + rating_filename, 'r'))
    max_rating_for_each_user[usr] = max(user_ratings.values())

    for category in range(10):
        if part_1_or_part_2 == 2:
            category += 10

        for image in range(15):
            # print(usr, category, image)
            img_feat = img_feats[str(category+1)][str(image+1)]
            one_hot_category = one_hot(str(category))
            try:
                user_rating = user_ratings[str(category+1) + '_' + str(image+1)]
            except:
                continue
            # row = [user_feat + one_hot_category + img_feat, user_rating]
            feature = user_feat + one_hot_category + img_feat
            feature_list.append(feature)
            rating_list.append(user_rating)

            user_cat_img_index.append([usr, category, image+1])

            # print(row)
            # data.append(row)

feature_list = np.array(feature_list)
pca = PCA(n_components=60)
new_feature_list = pca.fit_transform(feature_list)
print(new_feature_list.shape)

data = []
for feature, rating in zip(new_feature_list,  rating_list):
    # print(len(list(feature)), rating)
    data.append([list(feature), rating])


if part_1_or_part_2 == 1:
    json.dump(max_rating_for_each_user, open(data_top_path + 'max_rating_for_each_user.json', 'w'))
    json.dump(data, open(data_top_path + 'training.json', 'w'))
else:
    testing_dict = {}
    for index, feat in zip(user_cat_img_index, new_feature_list):
        # print(index, feat)
        user = index[0]
        category = index[1] + 1
        image = index[2]

        if user in testing_dict.keys():
            testing_dict[user][str(category) + '_' + str(image)] = list(feat)
        else:
            testing_dict[user] = {}
            testing_dict[user][str(category) + '_' + str(image)] = list(feat)

    json.dump(max_rating_for_each_user, open(data_top_path + 'max_rating_for_each_user.json', 'w'))
    json.dump(data, open(data_top_path + 'testing.json', 'w'))
    json.dump(testing_dict, open(data_top_path + 'testing_dict.json', 'w'))



# ad_feat_with_one_hot = np.zeros(150, len(img_feats[str(1)][str(1)]) + 20)
# ad_feat_without_one_hot = np.zeros(150, len(img_feats[str(1)][str(1)]))

#
# ad_feat_with_one_hot = np.empty((0, 512+20))
# ad_feat_without_one_hot = np.empty((0, 512))
#
# for category in range(10):
#     for image in range(15):
#         img_feat = img_feats[str(category + 1)][str(image + 1)]
#         one_hot_category = one_hot(str(category))
#
#         ad_feat_without_one_hot = np.append(ad_feat_without_one_hot, np.array([img_feat]), axis=0)
#         ad_feat_with_one_hot = np.append(ad_feat_with_one_hot, np.array([img_feat + one_hot_category]), axis=0)
#
# output_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/for_jonathan/'
# with_one_hot_name = 'ad_feat_with_one_hot_encoding.npy'
# without_one_hot_name = 'ad_feat_without_one_hot_encoding.npy'
#
# np.save(output_path + with_one_hot_name, ad_feat_with_one_hot)
# np.save(output_path + without_one_hot_name, ad_feat_without_one_hot)
#
# user_feat = np.empty((0, 150))
# ad_ratings = np.empty((0, 150))
#
# for usr in range(1, 61):
#     # print(usr)
#     if usr <= 9:
#         user_id = 'U000' + str(usr)
#
#     else:
#         user_id = 'U00' + str(usr)
#
#     feat_filename = user_id + '_feat.json'
#     rating_filename = user_id + '_rating.json'
#
#     user_feats = json.load(open(user_path + feat_filename, 'r'))
#     user_ratings = json.load(open(user_path + rating_filename, 'r'))
#     ratings = []
#     for category in range(10):
#         for image in range(15):
#             try:
#                 user_rating = user_ratings[str(category)][image]
#                 ratings.append(user_rating)
#             except:
#                 ratings.append(1)
#
#     # print(len(ratings))
#     user_feat = np.append(user_feat, np.array([user_feats]), axis=0)
#     ad_ratings = np.append(ad_ratings, np.array([ratings]), axis=0)
#
# user_feat_name = 'user_feat.npy'
# ad_rating_name = 'ad_ratings.npy'
#
# np.save(output_path + user_feat_name, user_feat)
# np.save(output_path + ad_rating_name, ad_ratings)
#
#
