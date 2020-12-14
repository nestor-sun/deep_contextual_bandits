from os import listdir
import re
import json

data1_top_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/'
user1_path = data1_top_path + 'Corpus/'

data2_top_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/'
user2_path = data2_top_path + 'Corpus/'


folder1 = [file for file in listdir(user1_path) if 'ini' not in file]
folder2 = [file for file in listdir(user2_path) if 'ini' not in file]


output1_path = data1_top_path + 'user/'
output2_path = data2_top_path + 'user/'


types_of_jobs = ['Housewife/Househusband', 'Contract employment','Student', 'Temporary', 'Odd job', 'Unemploied', \
                 'Self-employment']
working_hour_list =['Full Time', 'Part Time']
home_country_list = ['United States of America',  'Great Britain', 'Czech Republic', 'United Kingdom', 'Slovenia', 'Saudi Arabia',
 'Singapore', 'Canada', 'Romania', 'Phillipines', 'Italy', 'India']
fave_sport_list = ['I do not like Sports', 'Nothing', 'Team sports', 'Individual sports', 'Precision sports',
                   'Indoor sports', 'Hunting sports', 'Endurance sports', 'Skating sports', 'Other', 'Water sports',
                   'Equestrian sports', 'Olympic sports', 'Motor sports', 'Winter sports']


web_list = ['Media  sites', 'Grocery  beverages sites', 'Pet supplies sites', 'Toys  games sites',
            'Console  video games sites', 'Sports  outdoor sites', 'Stationery  office supplies sites',
            'Clothing  shoes sites', 'Health  Beauty sites', 'Home stuffs sites', 'Consumer electronics sites',
            'Outdoor Living sites', 'Jewellery  watches sites', 'Computer software sites', 'Automotive sites',
            'Tools  hardware sites', 'Dating sites', 'Musical instruments  recording equipment sites', 'Betting sites']

music_list = ['Classical Music', 'Easy Listening', 'Jazz', 'Dance Music', 'Electronic Music', 'Indie Pop', 'Pop',
              'Rock', 'Hip Hop - Rap', 'Alternative Music', 'Asian Pop', 'Singer - Songwriter', 'European Music',
              'Country Music', 'Latin Music', 'Blues', 'RB - Soul', 'Inspirational', 'World Music - Beats', 'Reggae',
              'New Age', 'Opera']

movie_list = ['Action', 'Thriller', 'Drama', 'Comedy', 'Mystery', 'Documentary', 'Animation', 'Musical', 'Family',
              'Sci-Fi', 'Adventure', 'Crime and Gangster', 'Epic - Historical', 'Fantasy', 'Horror', 'Romance',
              'Western', 'Biography', 'Erotic', 'Sport', 'War']

tv_list = ['Comedy', 'Drama', 'Sport', 'News', 'Learning', 'Weather', 'Entertainment', 'Music', "Children's", 'Factual',
           'Religion  Ethics']

book_list = ['Mystery', 'Romance', 'Science fiction', 'Science', 'Biographies', 'Satire', 'Trilogies', 'Series',
             'Action and Adventure', 'Horror', 'Travel', 'Journals', 'Fantasy', 'Cookbooks', 'Drama', 'Erotic fiction',
             'Religious', 'Guide', 'Comics', 'Math', "Children's literature", 'Self help', 'Prayer books', 'Diaries',
             "Children's", 'Poetry', 'History', 'Autobiographies']


def get_user_ratings(rt):
    with open(rt, 'r') as f:
        rating_list = []
        for line_num, line in enumerate(f):
            if line_num != 1:
                fields = line.replace('Cat', '').strip().split(';')
                rating_list.append(fields)

    rating_dict = {}
    for cat, rating in zip(rating_list[0], rating_list[1]):
        ratings = list(map(int, rating.replace('"', '').split(',')))
        rating_dict[cat.replace('"', '')] = ratings

    new_dict = {}
    for cat, rating_list in rating_dict.items():
        for img, rating in enumerate(rating_list):
            new_index = str(int(cat)+1) + '_' + str(img+1)
            # print(new_index, rating)
            new_dict[new_index] = rating

    return new_dict


def get_b5(file):
    b5 = []
    with open(file, 'r') as f:
        for row_num, row in enumerate(f):
            if row_num == 0:
                continue
            answer = int(row.strip().split(';')[-1].replace('"', ''))
            b5.append(answer)
    return b5


def get_demo(inf_file):
    feat = []
    info = open(inf_file).readlines()[1].strip().split(';')
    gender = info[2].replace('"', '')

    if gender == 'F':
        feat.append(0)
    else:
        feat.append(1)

    age = int(info[3].replace('"', ''))
    feat.append(age)

    type_of_job = info[5].replace('"', '')
    job_feat = [0 for i in types_of_jobs]
    try:
        job_feat[types_of_jobs.index(type_of_job)] = 1
    except:
        pass
    feat += job_feat

    working_hour = info[6].replace('"', '')
    work_hour_feat = [0, 0]
    work_hour_feat[working_hour_list.index(working_hour)] = 1

    feat+=work_hour_feat

    income = int(info[7].replace('"', ''))
    feat.append(income)

    home_country = info[8].replace('"', '')
    home_country_feat = [0 for i in home_country_list]
    home_country_feat[home_country_list.index(home_country)] = 1

    feat += home_country_feat

    # fave_sport = info[-1].replace('"', '')
    replace_sports = re.sub(r'\([^)]*\)', '', info[-1].replace('"', ''))
    fave_sports = replace_sports.strip().split(',')

    fave_sport_feat = [0 for i in fave_sport_list]
    for fave_sport in fave_sports:
        fave_sport = fave_sport.replace('â€Ž', '').strip()
        fave_sport_feat[fave_sport_list.index(fave_sport)] = 1
    feat += fave_sport_feat
    return feat


def get_pref(pref_file):
    info = open(pref_file).readlines()[1].strip().replace('&amp;', '').split(';')

    websites = re.sub(r'\([^)]*\)', '', info[0].replace('"', '')).split(',')
    music = re.sub(r'\([^)]*\)', '', info[1].replace('"', '')).split(',')
    movies = re.sub(r'\([^)]*\)', '', info[2].replace('"', '')).split(',')
    tvs = re.sub(r'\([^)]*\)', '', info[3].replace('"', '')).split(',')
    books = re.sub(r'\([^)]*\)', '', info[4].replace('"', '')).split(',')

    web_feat = [0 for i in web_list]
    for web in websites:
        web = web.strip()
        try:
            web_feat[web_list.index(web)] = 1
        except:
            continue

    music_feat = [0 for i in music_list]
    for mus in music:
        mus = mus.strip()
        music_feat[music_list.index(mus)] = 1

    movie_feat = [0 for i in movie_list]
    for mov in movies:
        mov = mov.strip()
        movie_feat[movie_list.index(mov)] = 1

    tv_feat = [0 for i in tv_list]
    for tv in tvs:
        tv =tv.strip()
        tv_feat[tv_list.index(tv)] = 1

    book_feat = [0 for i in book_list]
    for book in books:
        book = book.strip()
        try:
            book_feat[book_list.index(book)] = 1
        except:
            continue
    return web_feat + music_feat + movie_feat + tv_feat + book_feat


# for folder in folder1:
    # print(folder)
    # b5_file = user1_path + folder + '/' + folder + '-B5.csv'
    # neg_file = user1_path + folder + '/' + folder + '-NEG.csv'
    # pos_file = user1_path + folder + '/' + folder + '-POS.csv'
    # inf_file = user1_path + folder + '/' + folder + '-INF.csv'
    # pref_file = user1_path + folder + '/' + folder + '-PREF.csv'

    # rt = user1_path + folder + '/' + folder + '-RT.csv'

    # b5 = get_b5(b5_file)
    # demo = get_demo(inf_file)
    # pref = get_pref(pref_file)
    #
    # feat = b5 + demo + pref
    # print(len(feat), feat)
    # ratings = get_user_ratings(rt)
    # print(ratings)
    # json.dump(feat, open(output1_path + folder + '_feat.json', 'w'))
    # json.dump(ratings, open(output1_path + folder + '_rating.json', 'w'))

for folder in folder2:
    print(folder)
    b5_file = user2_path + folder + '/' + folder + '-B5.csv'
    neg_file = user2_path + folder + '/' + folder + '-NEG.csv'
    pos_file = user2_path + folder + '/' + folder + '-POS.csv'
    inf_file = user2_path + folder + '/' + folder + '-INF.csv'
    pref_file = user2_path + folder + '/' + folder + '-PREF.csv'

    rt = user2_path + folder + '/' + folder + '-RT.csv'

    b5 = get_b5(b5_file)
    demo = get_demo(inf_file)
    pref = get_pref(pref_file)

    feat = b5 + demo + pref
    print(len(feat), feat)
    ratings = get_user_ratings(rt)
    # print(ratings)
    json.dump(feat, open(output2_path + folder + '_feat.json', 'w'))
    json.dump(ratings, open(output2_path + folder + '_rating.json', 'w'))
