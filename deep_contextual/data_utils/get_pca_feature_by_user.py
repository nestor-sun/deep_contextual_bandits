import json

input_file = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/testing_dict.json'
output_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/user/'

test_dict = json.load(open(input_file, 'r'))

for usr in range(61, 121):
    print(usr)
    if usr <= 9:
        user_id = 'U000' + str(usr)
    elif usr >= 100:
        user_id = 'U0' + str(usr)
    else:
        user_id = 'U00' + str(usr)

    user_feat_dict = test_dict[str(usr)]

    json.dump(user_feat_dict, open(output_path + user_id + '_pca_feat.json', 'w'))


