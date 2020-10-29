from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import torch


class Data(Dataset):
    def __init__(self, img_top_dir, training_data_file, transform=None):
        self.img_top_dir = img_top_dir
        self.training_data = json.load(open(training_data_file))
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_top_dir + str(self.training_data[index][0][0]) + '/' + str(self.training_data[index][0][1]+1) + '.png'
        pic = Image.open(img_name).convert('RGB')
        return {'img': self.transform(pic), 'user_feat': torch.tensor(self.training_data[index][1]), \
                'y': torch.tensor(self.training_data[index][2])}

    def __len__(self):
        return len(self.training_data)


# img_top_dir = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/Ads/'
# training_data_file = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/training.json'
# transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), \
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# train = Data(img_top_dir, training_data_file, transform)
#
# print(train[1])
# print(len(train))
