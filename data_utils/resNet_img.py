import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import json

resnet = models.resnet34(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])

for para in resnet.parameters():
    para.requires_grad = False

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = transform(image).float()
    return image


data_top_path = '/data/ads_16/'


full_data = {}
for category in range(11, 21):
    data = {}
    for img in range(1, 16):
        image = image_loader(data_top_path + 'ADS16_Benchmark_part2/Ads/{}/{}.png'.format(category, img))
        image = image.unsqueeze(0)
        image = resnet(image).flatten().tolist()

        data[img] = image

    full_data[category] = data

json.dump(full_data, open(data_top_path + 'ADS16_Benchmark_part2/img_feat.json', 'w'))

