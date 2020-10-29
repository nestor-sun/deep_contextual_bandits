import json
import os

ads16_top_path = 'C:/data/ads_16/ADS16_Benchmark_part1/ADS16_Benchmark_part1/Corpus/Corpus/'
user_file_name = 'U0001/'
rating_filename = 'U0001-RT.csv'

with open(ads16_top_path + user_file_name + rating_filename, 'r') as f:
    rating_list = []
    for line_num, line in enumerate(f):
        if line_num != 0:
            fields = line.strip().split(';')
            rating_list.append(fields)

rating_dict = {}
for cat, rating in zip(rating_list[0], rating_list[1]):
    ratings = list(map(int, rating.replace('"', '').split(',')))
    rating_dict[cat.replace('"', '')] = ratings

output_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/'

# if not os.path.exists(output_path + user_file_name):
#     os.makedirs(output_path + user_file_name)
json.dump(rating_dict, open(output_path + user_file_name.replace('/', '') + '.json', 'w'))

