import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):   # 初始化LeNet-5模型，定义网络层，激活函数，参数
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)   
        # 卷积层1，输入通道1，输出通道6，卷积核大小5，padding为2
        self.sig=nn.Sigmoid()       # sigmoid激活函数
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)   
        # 池化层2，池化核大小2，步长为2
        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        # 卷积层3，输入通道6，输出通道16，卷积核大小5，padding为0
        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)   
        # 池化层4，池化核大小2，步长为2
        self.flatten=nn.Flatten()
        # 将其平展成一维
        self.f5=nn.Linear(in_features=16*5*5,out_features=120)
        # 全连接层1，输入特征16*5*5，输出特征120
        self.f6=nn.Linear(in_features=120,out_features=84)
        # 全连接层2，输入特征120，输出特征84
        self.f7=nn.Linear(in_features=84,out_features=10)
        # 全连接层3，输入特征84，输出特征10
    def forward(self,x):
        # x=self.c1(x)
        # x=self.sig(x)
        # x=self.s2(x)
        # x=self.c3(x)
        # x=self.sig(x)
        # x=self.s4(x)
        # x=self.flatten(x)
        # x=self.f5(x)
        # x=self.sig(x)
        # x=self.f6(x)
        # x=self.sig(x)
        # x=self.f7(x)
        x=self.sig(self.c1(x))    # 卷积的时候使用激活函数
        x=self.s2(x)
        x=self.sig(self.c3(x))
        x=self.s4(x)
        x=self.flatten(x)
        x=self.f5(x)
        x=self.f6(x)
        x=self.f7(x)
        return x

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LeNet().to(device)
    print(summary(model,(1,28,28)))   # 打印模型结构










