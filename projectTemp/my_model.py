from torch import nn


class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        # 一般这里把需要使用的卷积、池化等函数在这里给到全局变量

    def forward(self, data):
        # 类的功能实现部分，input就是这个类的输入，对于模型类来说，这个输入是个图片数据
        # 这里使用上面的全局变量，整合成一个完整的神经网络，就是网络的前向过程
        # 而网络的反向传播在模型运行之外，torch有直接的函数帮助我们计算反向传播，后面讲解
        # ouput就是网络模型的输出，一般是个列表
        return data


