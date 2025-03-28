from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
train_data=FashionMNIST(root='./人工智能/神经网络/LeNet-5/data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                        download=False)
train_loader=Data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)    # 将数据改为64个一批，并使用0个线程来加载数据，线程用于提高加载速度。
for step,(b_x,b_y) in enumerate(train_loader):
    if step>0:
        break
# 获取一个batch的数据
batch_x=b_x.squeeze().numpy()  # 将四维张量移除第一维，并改为numpy数组
batch_y=b_y.numpy()           # 转换为numpy数组     因为tensor时神经网络中的格式，所以需要转换为numpy方便后面绘图
class_label=train_data.classes   # 获取类别标签
print(class_label)

# 可视化
import matplotlib.pyplot as plt
# 创建一个图形对象，设置图形大小为12x5英寸
plt.figure(figsize=(12,5))
# 遍历batch_y的长度，为每个元素创建一个子图
for ii in range(len(batch_y)):
    # 在4行16列的子图布局中选择第ii+1个位置创建子图
    plt.subplot(4,16,ii+1)
    # 显示batch_x中第ii个图像，使用灰度颜色映射
    plt.imshow(batch_x[ii,:,:],cmap=plt.cm.gray)
    # 设置子图标题为class_label中对应的标签，字体大小为10
    plt.title(class_label[batch_y[ii]],size=10)
    # 关闭坐标轴显示
    plt.axis('off')
    # 调整子图之间的间距，水平间距为0.05
    plt.subplots_adjust(wspace=0.05)
# 显示所有子图
plt.show()



