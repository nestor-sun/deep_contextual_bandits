import sys
sys.path.append('deep_contextual/model')
import torch
from dataloader import Data
import pandas as pd


path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/deep_contextual/model/'
model = torch.load(path + 'model.dat')


testing_data_file = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part2/testing.json'
testing_data = Data(testing_data_file)

data = []
for i, data_row in enumerate(testing_data):
    feat = data_row[0]
    y = data_row[1]

    if torch.cuda.is_available():
        feat = feat.cuda()
        y = y.cuda()

    output = model(feat)
    # print(output)
    data.append({'prediction': float(output), 'target': float(y)})

df = pd.DataFrame(data)
print(df.corr())


