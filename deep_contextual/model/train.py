import sys
sys.path.append('deep_contextual/model')

from deep_contextual_bandits import DeepContextualBandits
from dataloader import Data
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
import pandas as pd


learning_rate = 0.0001
model = DeepContextualBandits(60)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


training_data_file = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/training.json'
train_data = Data(training_data_file)
# loader = DataLoader(train_data, batch_size=12, shuffle=True, num_workers=2)

# print(train_data[0][0], train_data[0][1])

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


def train():
    train_losses = []
    model.train()
    for i, data_row in enumerate(train_data):
        feat = data_row[0]
        # print(len(feat))
        y = data_row[1]

        feat = Variable(feat)
        y = Variable(y).double()

        # converting the data into GPU format
        if torch.cuda.is_available():
            feat = feat.cuda()
            y = y.cuda()

        output_train = model(feat)
        output_train = output_train.double()
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        loss_train = criterion(output_train, y)
        train_losses.append(loss_train)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
    return float(sum(train_losses))


loss_list = []
for epoch in range(250):
    loss = train()/len(train_data)
    loss_list.append(loss)
    print('Epoch', epoch, 'Loss', loss)

output_path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/deep_contextual/model/'
json.dump([float(i) for i in loss_list], open(output_path + 'loss_list.json', 'w'))
torch.save(model, output_path + 'model.dat')

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


