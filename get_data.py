import json

user_path = 'C:/Users/Varun Suryan/Desktop/ML Guarnatess/Project/ADS16_Benchmark_part1/user/'


def one_hot(category):
    num_category = 20
    init = [0 for _ in range(num_category)]
    init[int(category)] = 1
    return init

data = []

for usr in range(41, 61):
    print(usr)
    if usr <= 9:
        user_id = 'U000' + str(usr)
    
    else:
        user_id = 'U00' + str(usr)
    
    feat_filename = user_id + '_feat.json'
    rating_filename = user_id + '_rating.json'

    data_feat = json.load(open(user_path + feat_filename, 'r'))
    data_rating = json.load(open(user_path + rating_filename, 'r'))
    
    for categories in range(20):
        for image in range(15):
            row = [[str(categories), image], data_feat, data_rating[str(categories)][image], one_hot(str(categories))]
            data.append(row)


json.dump(data, open('C:/Users/Varun Suryan/Desktop/ML Guarnatess/Project/' + 'data.json', 'w'))