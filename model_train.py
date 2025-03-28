import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import copy
import time
import pandas as pd

'''数据加载'''
def train_val_data_process():   # 定义训练集和验证集的处理函数
    train_data=FashionMNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.Resize(size=32),transforms.ToTensor()]),
                            download=False)
    train_data,val_data=Data.random_split(train_data,[round(len(train_data)*0.8),   # 随机划分训练集和验证集
                                                      round(len(train_data)*0.2)])  # 验证集
    train_dataloader=Data.DataLoader(dataset=train_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)   # 训练集的DataLoader(划分)
    val_dataloader=Data.DataLoader(dataset=val_data,
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=2)   # 验证集的DataLoader
    return train_dataloader,val_dataloader

'''模型训练过程'''
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):   # 定义模型训练的过程
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 选择设备
    
    # 定义优化器和损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)  # 定义优化器，使得模型在训练过程中更新权重
    # adam移动平均值，使得模型在训练过程中更加平滑，防止梯度爆炸，加速梯度下降
    criterion=nn.CrossEntropyLoss()   # 定义损失函数，交叉熵   

    model=model.to(device)   # 将模型加载到设备上

    best_model_wts=copy.deepcopy(model.state_dict())   # 保存最佳模型参数

    # 初始化参数：
    best_acc=0.0   # 最佳准确率
    train_loss_all=[]  # 训练集损失函数列表
    val_loss_all=[]
    train_acc_all=[]   # 训练集准确率列表
    val_acc_all=[]
    since=time.time()   # 记录训练开始时间

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))   # 打印当前epoch,此时每个epoch是从1开始的，到num_epochs结束
        print('-'*10)   # 打印分割线

        # 初始化参数：
        train_loss=0.0    # 训练集损失函数
        train_corrects=0  # 训练的准确度
        val_loss=0.0
        val_corrects=0

        train_num=0      #训练集样本数
        val_num=0      #验证集样本数

        '''训练阶段'''
        for step,(b_x,b_y) in enumerate(train_dataloader):    # 遍历训练集，b_x为一个batch的输入数据，b_y为一个batch的标签数据
        # 此时b_x为128*28*28*1的四维张量，b_y为128个标签,其中的enumerate函数返回的是一个enumerate对象，
        # 该对象包含两个元素，第一个元素是索引，第二个元素是b_x和b_y
            b_x=b_x.to(device)   # 将训练集的输入数据加载到设备上
            b_y=b_y.to(device)   # 将训练集的标签数据加载到设备上

            model.train()        # 开启训练模式
            output=model(b_x)    # 前向传播,输入为一个batch,输出为一个batch中的对应预测

            pre_lab=torch.argmax(output,dim=1)   # 查找每行中最大概率的行标，即预测的标签（类似softmax操作）
            loss=criterion(output,b_y)   # 计算每个batch的损失函数

            optimizer.zero_grad()   # 梯度清零初始化
            loss.backward()         # 反向传播
            optimizer.step()        # 更新权重,根据反向传播的梯度信息来更新网络参数，从而降低loss函数计算值的作用

            train_loss+=loss.item()*b_x.size(0)   # 累加训练集的损失函数
            train_corrects+=torch.sum(pre_lab==b_y.data)   # 如果预测正确，则累加训练集的正确数+1
            train_num+=b_x.size(0)   # 累加训练集的样本数

        '''验证阶段'''
        for step,(b_x,b_y) in enumerate(val_dataloader):   # 遍历验证集
            b_x=b_x.to(device)   # 将验证集的输入数据加载到设备上
            b_y=b_y.to(device)   # 将验证集的标签数据加载到设备上

            model.eval()         # 开启验证模式

            output=model(b_x)    # 前向传播,输入为一个batch,输出为一个batch中的对应预测
            pre_lab=torch.argmax(output,dim=1)   # 查找每行中最大概率的行标，即预测的标签（类似softmax操作）
            loss=criterion(output,b_y)   # 计算每个batch的损失函数

            val_loss+=loss.item()*b_x.size(0)   # 累加验证集的损失函数
            val_corrects+=torch.sum(pre_lab==b_y.data)   # 如果预测正确，则累加验证集的正确数+1
            val_num+=b_x.size(0)   # 累加验证集的样本数

        train_loss_all.append(train_loss/train_num)   # 计算并保存每次迭代的loss值和准确率
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double().item()/train_num)      # item()函数将tensor转换为python的float类型，
        val_acc_all.append(val_corrects.double().item()/val_num)            # double()函数将tensor转换为python的float类型
        

        print('{}Train Loss: {:.4f}  Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1])) 
        print('{}Val Loss: {:.4f} Vac Acc: {:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        # 保存最佳模型参数,权重
        # 寻找最佳准确度
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]   # 更新最佳准确度
            best_model_wts=copy.deepcopy(model.state_dict())     # 保存最佳模型参数,deepcopy()函数用于深度复制模型参数

        # 打印训练时间
        time_use=time.time()-since
        print('训练耗时{:.0f}m{:.0f}s'.format(time_use//60,time_use%60))

    # 选择最优参数，
    # 加载最高准确率下的模型参数
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),'./best_model_params.pth')   # 保存模型参数
    
    # torch.save(best_model_wts,'./人工智能/神经网络/LeNet-5/best_model_params.pth')

    train_process=pd.DataFrame(data={
        'epoch':range(num_epochs),
        'train_loss_all':train_loss_all,
        'val_loss_all':val_loss_all,
        'train_acc_all':train_acc_all,
        'val_acc_all':val_acc_all
    })   # 保存训练过程数据

    return train_process   # 返回训练过程数据
        
# 根据训练过程数据绘制训练曲线
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))  # 设置画布大小   
    plt.subplot(1,2,1)          # 绘制训练集损失率曲线,(1,2,1)表示1行2列的第一个子图
    plt.plot(train_process['epoch'],train_process.train_loss_all,'ro-',label='train_loss')
    plt.plot(train_process['epoch'],train_process.val_loss_all,'bs-',label='val_loss')
    plt.legend()      # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1,2,2)          # 绘制训练集准确率曲线,(1,2,2)表示1行2列的第二个子图
    plt.plot(train_process['epoch'],train_process.train_acc_all,'ro-',label='train_acc')
    plt.plot(train_process['epoch'],train_process.val_acc_all,'bs-',label='val_acc')
    plt.legend()      # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('acc')

if __name__=='__main__':
    # 加载模型
    LeNet=LeNet()
    # 加载数据
    train_dataloader,val_dataloader=train_val_data_process()
    # 训练模型
    train_process=train_model_process(LeNet,train_dataloader,val_dataloader,num_epochs=50)
    # 绘制训练曲线
    matplot_acc_loss(train_process)
    plt.show()



