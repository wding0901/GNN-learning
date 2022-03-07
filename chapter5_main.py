#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from DataSource.CoraData import CoraData
from NeuralNetwork.GcnNet import GcnNet


"""
从numpy元素创建一个tensor张量，双方共享同一块内存空间
"""
def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


# 训练主体函数
def train(epochs):
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):

        # 直接调用model时触发__call__方法，调用了里面的forward函数
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        train_acc, _, _ = test(tensor_train_mask)  # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)  # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history


# In[9]:


# 测试函数
def test(mask):
    model.eval()
    with torch.no_grad():
        # tensor_x 是 2708 * 1433 的矩阵，2708个图，每个图1433个特征
        # logits 是2708 * 7的矩阵，因为图有7种标签，每行7个值分别代表是该类标签的概率
        logits = model(tensor_adjacency, tensor_x)

        # 这边取mask是因为训练集和测试集的数据放在一起，通过mask里面的下标区分
        test_mask_logits = logits[mask]

        """
        tensor.max(k) 是指在第K维进行比较。第一个返回值是最大值，第二个是下标
        通俗的说max(0)表示行进行比较，即找出每一列的最大值，以及最大值的下标
        max(1)表示列进行比较，即找出每一行的最大值，及其下标
        这边test_mask_logits每行的7个值代表7种类型的概率，每行最大值的下标就表示预测的分类类型
        """
        predict_y = test_mask_logits.max(1)[1]

        # torch.eq对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


# In[13]:


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


def tsne_show(test_logits):
    tsne = TSNE()
    out = tsne.fit_transform(test_logits)
    fig = plt.figure()
    for i in range(7):
        indices = test_label == i
        x, y = out[indices].T
        plt.scatter(x, y, label=str(i))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 超参数定义
    learning_rate = 0.1
    weight_decay = 5e-4
    epochs = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据，并转换为torch.Tensor
    dataset = CoraData().data
    node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

    tensor_x = tensor_from_numpy(node_feature, device)
    tensor_y = tensor_from_numpy(dataset.y, device)
    tensor_train_mask = tensor_from_numpy(dataset.train_mask, device)
    tensor_val_mask = tensor_from_numpy(dataset.val_mask, device)
    tensor_test_mask = tensor_from_numpy(dataset.test_mask, device)
    normalize_adjacency = CoraData.normalization(dataset.adjacency)  # 规范化邻接矩阵

    num_nodes, input_dim = node_feature.shape

    indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                           normalize_adjacency.col]).astype('int64')).long()
    values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values,
                                                (num_nodes, num_nodes)).to(device)

    # 模型定义：Model, Loss, Optimizer
    # 这里只是初始化模型，并没有forward
    model = GcnNet(input_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    loss, val_acc = train(epochs)
    test_acc, test_logits, test_label = test(tensor_test_mask)
    print("Test accuarcy: ", test_acc.item())

    plot_loss_with_acc(loss, val_acc)

    # 绘制测试数据的TSNE降维图
    tsne_show(test_logits)

