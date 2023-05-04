import json
import os
from random import random

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.modules import module

from torch.utils.data import DataLoader
from train_fn import train, evaluate, get_dataLoader, visual_result
from minst_model import minst_model


# 自定义dataloader加载数据方式
# 适用情况，自定义的数据不规整，使用pytorch自带的会报错，这时就需要使用自定义的collate，否则将数据进行规整再使用
def my_collate(batch):
    # 依据你自己的数据集来实现下面的内容
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


SEED = 1210


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # 对numpy模组进行随机数设定
    torch.manual_seed(seed)  # 对torch中的CPU部分进行随机数设定
    torch.cuda.manual_seed(seed)  # 对torch中的GPU部分进行随机数设定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)  # 设置几乎所有的随机种子 随机种子，可使得结果可复现

# 超参数设置
epochs = 3  # 定义训练轮次
batch_size = 32  # batch 大小设置
learning_rate = 1e-3  # 学习率
weight_decay = 1e-4  # （权重衰减）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
train_size_rate = 0.8  # 训练集占总数据集的比例
# nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
nw = 0  # 并行使用几个gpu工作 ,windows环境下一般为 0  linux环境下可以有多个
print('Using {} dataloader workers every process'.format(nw))

# 模型的相关设置
model = minst_model()  # 使用定义的模型
model_name = "minst_model"  # 模型的名称
model = model.to(device)  # 使用不同的device
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 二分类交叉熵损失函数
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 传入参数进入到数据集中
parameters = [

]
# dataset 自定义数据集加载
# dataset = dataset(parameters)
# 使用已定义的数据集
train_data = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))
test_data = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
# 查看输入数据的形状
data, target = train_data.__getitem__(0)
print(data.shape)
# 切割数据集 ，使用dataloader加载数据
# train_loader, test_loader = get_dataLoader(batch_size, train_size_rate, None, nw, dataset)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


# 开始训练
for epoch in range(epochs):
    train_loss, train_acc, train_prec, train_recall, train_f1 = train(model,
                                                                      train_loader,
                                                                      optimizer,
                                                                      criterion,
                                                                      device,
                                                                      epoch)
    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(model,
                                                                    test_loader,
                                                                    criterion,
                                                                    device,
                                                                    epoch)
    # 每个epoch都保存一次模型的参数
    save_path = './model_epoch/{}_epoch_{}_net.pth'.format(model_name, epoch+1)
    # 模型保存的位置
    torch.save(model.state_dict(), save_path)
    print("{}_epoch_{}:模型已保存!".format(model_name, epoch+1))
    # 加载模型的参数
    # model.load_state_dict(torch.load(save_path))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    # model.eval()  # 一定要记住在评估模式的时候调用model.eval()来固定dropout和批次归一化。否则会产生不一致的推理结果。
    # visualize the loss as the network trained 可视化每一个epoch的训练的效果
    visual_result(train_loss, test_loss, epoch, "batch=32", "loss", model_name, 'loss')
    # visual_result(train_acc, test_acc, epoch, "accuracy", model_name, 'acc')
    # visual_result(train_prec, test_prec, epoch, "precision", model_name, 'prec')
    # visual_result(train_recall, test_recall, epoch, "recall", model_name, 'recall')
    visual_result(train_f1, test_f1, epoch, "batch=32", "f1", model_name, 'f1')
