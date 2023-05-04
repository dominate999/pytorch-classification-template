import torch

from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from minst_model import minst_model

# 使用PIL读取
img_pil = Image.open('img_2.png')         # PIL.Image.Image对象
transform1 = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor()
])

r, g, b = img_pil.split()
plt.imshow(r)
plt.show()

img = transform1(r)
img = img.unsqueeze(0)

print(img.shape)


model = minst_model()

model.load_state_dict(torch.load("minst_model_epoch_3_net.pth"))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
model.eval()  # 一定要记住在评估模式的时候调用model.eval()来固定dropout和批次归一化。否则会产生不一致的推理结果。

output = model(img)
print(output)

result = output.argmax(dim=1)

print(result)
