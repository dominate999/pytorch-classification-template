import json

from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, parameters, transform=None, target_transform=None):
        super(dataset, self).__init__()
        self.datas = []
        # 这里对传入的parameters做处理
        # 将你需要的数据处理后，存放到datas中
        # datas中的每一个数据都是一条数据，格式为 特征值 目标值
        # 这里也可对数据进行预处理
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data, target = self.datas[index]
        return data, target

    def __len__(self):
        return len(self.datas)
