from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


path = 'D:/study/gumd/UMD_course/machine learning guarantees and analyses/deep_contextual_bandits/data/ads_16/ADS16_Benchmark_part1/Ads/1/'
file1 = '14.png'
file2 = '13.png'

img2 = Image.open(path + file1).convert('RGB')
img1 = Image.open(path + file2).convert('RGB')
draw = ImageDraw.Draw(img)
# img.show()
transform = transforms.Compose([transforms.ToTensor()])
transformed1 = transform(img1)
transformed2 = transform(img2)

trans1 = transforms.ToPILImage()

back1 = trans1(transformed1)
back1.show()

back2 = trans1(transformed2)
back2.show()

