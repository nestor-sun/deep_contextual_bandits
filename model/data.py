import os
import json
import numpy as np


ads_top_dir = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/Ads/'
user_top_dir = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/user/'


def get_data():
    user_list = []
    for u in range(1, 61):
        if u < 10:
            user_list.append('U000' + str(u))
        else:
            user_list.append('U00' + str(u))

    dataset = []
    for ads_cat in [folder for folder in os.listdir(ads_top_dir) if '.ini' not in folder]:
        print(ads_cat)
        ads_cat_dir = ads_top_dir + ads_cat + '/'
        for img_name in [img for img in os.listdir(ads_cat_dir) if 'png' in img]:
            for user in user_list:
                # img = Image.open(ads_cat_dir + img_name)
                # pix = transforms(img)
                # print(pix)
                rating = json.load(open(user_top_dir + user + '_rating.json', 'r'))
                user_feat = json.load(open(user_top_dir + user + '_feat.json', 'r'))
                # print(user, img, ads_cat)
                try:
                    label = rating[ads_cat][int(img_name.replace('.png', ''))-1]
                    dataset.append([[ads_cat, int(img_name.replace('.png', ''))-1], user_feat, label])
                except:
                    continue
    return dataset


output_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/'
data = get_data()
json.dump(data, open(output_path + 'training.json', 'w'))

