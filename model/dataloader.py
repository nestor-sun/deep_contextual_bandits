from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import torch


class Data(Dataset):
    def __init__(self, training_data_file):
        self.training_data = json.load(open(training_data_file))

    def __getitem__(self, index):
        return torch.tensor(self.training_data[index][0]), torch.tensor(self.training_data[index][1])

    def __len__(self):
        return len(self.training_data)


# img_top_dir = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/Ads/'
# training_data_file = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/training.json'
# transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), \
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# train = Data(training_data_file)
# feat, y = train[1]
# print(len(feat))

file = 'file.json'
data = Data(file)

for data_row in range(len(data)):
    print(data[data_row])


