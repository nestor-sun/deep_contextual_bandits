from os import listdir
import re

data1_top_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/'
user1_path = data1_top_path + 'Corpus/'

data2_top_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/'
user2_path = data1_top_path + 'Corpus/'


folder1 = [file for file in listdir(user1_path) if 'ini' not in file]
folder2 = [file for file in listdir(user2_path) if 'ini' not in file]

web_list = []
music_list = []
movie_list = []
tv_list = []
book_list = []


for folder in folder1:

    inf_file = user1_path + folder + '/' + folder + '-INF.csv'
    pref_file = user1_path + folder + '/' + folder + '-PREF.csv'

    info = open(pref_file).readlines()[1].strip().replace('&amp;', '').split(';')

    websites = re.sub(r'\([^)]*\)', '', info[0].replace('"', '')).split(',')
    music = re.sub(r'\([^)]*\)', '', info[1].replace('"', '')).split(',')
    movies = re.sub(r'\([^)]*\)', '', info[2].replace('"', '')).split(',')
    tvs = re.sub(r'\([^)]*\)', '', info[3].replace('"', '')).split(',')
    books = re.sub(r'\([^)]*\)', '', info[4].replace('"', '')).split(',')

    print(music)


    for web in websites:
        web = web.strip()
        if web not in web_list:
            web_list.append(web)

    for mus in music:
        mus = mus.strip()
        if mus not in music_list:
            music_list.append(mus)

    for mov in movies:
        mov = mov.strip()
        if mov not in movie_list:
            movie_list.append(mov)

    for tv in tvs:
        tv =tv.strip()
        if tv not in tv_list:
            tv_list.append(tv)

    for book in books:
        book = book.strip()
        if book not in book_list:
            book_list.append(book)




for folder in folder2:

    inf_file = user2_path + folder + '/' + folder + '-INF.csv'
    pref_file = user2_path + folder + '/' + folder + '-PREF.csv'

    info = open(pref_file).readlines()[1].strip().replace('&amp;', '').split(';')

    websites = re.sub(r'\([^)]*\)', '', info[0].replace('"', '')).split(',')
    music = re.sub(r'\([^)]*\)', '', info[1].replace('"', '')).split(',')
    movies = re.sub(r'\([^)]*\)', '', info[2].replace('"', '')).split(',')
    tvs = re.sub(r'\([^)]*\)', '', info[3].replace('"', '')).split(',')
    books = re.sub(r'\([^)]*\)', '', info[4].replace('"', '')).split(',')

    print(music)

    for web in websites:
        web = web.strip()
        if web not in web_list:
            web_list.append(web)

    for mus in music:
        mus = mus.strip()
        if mus not in music_list:
            music_list.append(mus)

    for mov in movies:
        mov = mov.strip()
        if mov not in movie_list:
            movie_list.append(mov)

    for tv in tvs:
        tv = tv.strip()
        if tv not in tv_list:
            tv_list.append(tv)

    for book in books:
        book = book.strip()
        if book not in book_list:
            book_list.append(book)



