import sys
sys.path.append('D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/code/model')

from deep_contextual_bandits import DeepContextualBandits
from dataloader import Data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable


learning_rate = 0.001
model = DeepContextualBandits(150)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


img_top_dir = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/Ads/'
training_data_file = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/training.json'
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), \
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_data = Data(img_top_dir, training_data_file, transform)
loader = DataLoader(train_data, batch_size=12, shuffle=True, num_workers=2)

# model = models.resnet50(pretrained=True)
# img = train_data[0]['img']
# user = train_data[0]['user_feat']
#
# model(img.unsqueeze(0))


if torch.cuda.is_available():
    model.cuda()
    criterion = criterion.cuda()


def train():
    train_losses = []
    model.train()
    for i, data in enumerate(loader):
        img = Variable(train_data[i]['img'])
        user_feat = Variable(train_data[i]['user_feat'])
        y = Variable(train_data[i]['y']).double()

        # converting the data into GPU format
        if torch.cuda.is_available():
            img = img.cuda()
            user_feat = user_feat.cuda()
            y = y.cuda()

        output_train = model(img.unsqueeze(0), user_feat)
        output_train = output_train.double()
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        loss_train = criterion(output_train, y)
        train_losses.append(loss_train)
        train_losses.append(loss_train)
        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
    return sum(train_losses)


for epoch in range(200):
    loss = train()
    print('Epoch', epoch, 'Loss', loss)

