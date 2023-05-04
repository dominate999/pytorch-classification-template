import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm


def get_dataLoader(batch_size, train_size_rate, collate_fn, nw, dataset):
    train_size = int(train_size_rate * len(dataset))
    print("训练集的数据大小为:", train_size)
    test_size = len(dataset) - train_size
    print("测试集的数据大小为:", test_size)
    #  使用torch random_split对数据集进行切割
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    if collate_fn is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, collate_fn=collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   )
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    return train_loader, test_loader


def sklearn_fn(epoch, predict, true, average='macro'):
    accuracy = accuracy_score(true, predict)
    precision = precision_score(true, predict, average=average)
    recall = recall_score(true, predict, average=average)
    f1 = f1_score(true, predict, average=average)
    print(
        "[EPOCH&BATCH] Epoch:{} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(epoch + 1,
                                                                                                 accuracy,
                                                                                                 precision,
                                                                                                 recall,
                                                                                                 f1))
    print(classification_report(true, predict))


def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    train_loss_list = []  # 每次训练的loss 用于展示数据
    train_loss = 0.0  # 训练损失值，所有的loss累加的值
    train_preds = []  # 预测值 prediction -s 总数
    train_trues = []  # 真值,总数
    train_batch_preds = []  # 每个batch训练的预测值
    train_batch_trues = []  # 每次batch训练的真值
    visual_acc = []
    visual_precision = []
    visual_recall = []
    visual_f1 = []
    train_bar = tqdm(dataloader)  # 进度条显示数据
    for step, data in enumerate(train_bar):
        tokens, targets = data  # 获取模型中的数据 特征、目标值 均为一个batch数组
        # print(tokens.shape)
        tokens = tokens.to(device)
        targets = targets.to(device)  # 根据device选择设备 GPU or CPU
        optimizer.zero_grad()  # 优化器清零
        outputs = model(tokens)  # 得到预测值
        loss = criterion(outputs, targets)  # 使用损失函数进行比对
        loss.backward()  # 反向传播
        optimizer.step()  # 使用优化器
        train_loss += loss.item()  # 累加统计损失值 ，注意一定是需要使用 .item(),具体原因自行百度！
        train_loss_list.append(loss.item())  # 以数组的形式，添加到训练损失总值中
        train_outputs = outputs.argmax(dim=1)  # 比较并输出一组元素中最大值所在的索引 argmax(1) 横向比较

        train_batch_preds.extend(train_outputs.detach().cpu().numpy())  # 统计每个batch的值
        train_batch_trues.extend(targets.detach().cpu().numpy())

        train_preds.extend(train_batch_preds)  # 转换为数组的形式，并统计总预测值
        train_trues.extend(train_batch_trues)  # 转换为数组的形式，并统计总真值
        # 使用sklearn 对数据进行分析，输出结果
        sklearn_accuracy = accuracy_score(train_trues, train_preds)
        visual_acc.append(sklearn_accuracy)   # 统计每个Batch的准确值
        sklearn_precision = precision_score(train_trues, train_preds, average='macro')
        visual_precision.append(sklearn_precision)
        sklearn_recall = recall_score(train_trues, train_preds, average='macro')
        visual_recall.append(sklearn_recall)
        sklearn_f1 = f1_score(train_trues, train_preds, average='macro')
        visual_f1.append(sklearn_f1)
        if step % (math.floor(len(dataloader) / 10)) == 0:  # 所有batch切分成10部分输出, 向下取整
            sklearn_fn(epoch, train_batch_preds, train_batch_trues, average='macro')
            train_batch_preds = []
            train_batch_trues = []
        train_bar.desc = "[train__eppch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
            epoch + 1, train_loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1)
    sklearn_fn(epoch, train_preds, train_trues, average='macro')

    return train_loss_list, visual_acc, visual_precision, visual_recall, visual_f1


def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    test_preds = []
    test_trues = []
    test_batch_preds = []
    test_batch_trues = []
    test_loss = 0.0
    test_loss_list = []
    i = 0
    visual_acc = []
    visual_precision = []
    visual_recall = []
    visual_f1 = []
    test_bar = tqdm(dataloader)
    with torch.no_grad():  # 这句话就将这里面的语句不去关注梯度信息
        for step, data in enumerate(test_bar):
            test_tokens, test_targets = data
            test_tokens = test_tokens.to(device)
            test_targets = test_targets.to(device)
            outputs = model(test_tokens)
            loss = criterion(outputs, test_targets)  # 使用损失函数进行比对
            test_loss += loss.item()
            test_loss_list.append(loss.item())

            test_outputs = outputs.argmax(dim=1)

            test_batch_preds.extend(test_outputs.detach().cpu().numpy())
            test_batch_trues.extend(test_targets.detach().cpu().numpy())

            test_preds.extend(test_batch_preds)
            test_trues.extend(test_batch_trues)

            test_accuracy = accuracy_score(test_trues, test_preds)
            visual_acc.append(test_accuracy)
            test_precision = precision_score(test_trues, test_preds, average='macro')
            visual_precision.append(test_precision)
            test_recall = recall_score(test_trues, test_preds, average='macro')
            visual_recall.append(test_recall)
            test_f1 = f1_score(test_trues, test_preds, average='macro')
            visual_f1.append(test_f1)
            if step % (math.floor(len(dataloader) / 10)) == 0:  # 所有batch切分成10部分输出, 向下取整
                sklearn_fn(epoch, test_batch_preds, test_batch_trues, average='macro')
                test_batch_preds = []
                test_batch_trues = []
            test_bar.desc = "[test__epoch__bar] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
                epoch + 1, test_loss, test_accuracy, test_precision, test_recall, test_f1)
        sklearn_fn(epoch, test_preds, test_trues, average='macro')

    return test_loss_list, visual_acc, visual_precision, visual_recall, visual_f1


# 结果可视化
def visual_result(train_data, test_data, epoch, label_name_x, label_name_y, fig_name, result_type):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_data) + 1), train_data, label='Train {}'.format(label_name_y))
    # 对于测试集输出的结果 绘制散点图
    scale = len(train_data)/len(test_data)
    test_range = range(1, len(test_data)+1)
    test_range = [i * scale for i in test_range]
    plt.plot(test_range, test_data, label='Test {}'.format(label_name_y))

    plt.xlabel(label_name_x)
    plt.ylabel(label_name_y)
    plt.ylim(0, max(train_data))  # consistent scale
    plt.xlim(0, len(train_data) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('./visual/{}_plot_{}_{}.png'.format(result_type, fig_name, epoch + 1), bbox_inches='tight')
