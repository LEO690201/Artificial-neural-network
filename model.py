import torch
from torch import nn
from torchsummary import summary
class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.ReLU=nn.ReLU()
        # 路线1：1x1卷积
        self.p1_1=nn.Conv2d(in_channels=in_channels,
                          out_channels=c1,
                          kernel_size=1)
        # 路线2：1x1卷积+3x3卷积
        self.p2_1=nn.Conv2d(in_channels=in_channels,
                          out_channels=c2[0],
                          kernel_size=1)
        self.p2_2=nn.Conv2d(in_channels=c2[0],
                          out_channels=c2[1],
                          kernel_size=3,
                          padding=1)
        # 路线3：1x1卷积+5x5卷积
        self.p3_1=nn.Conv2d(in_channels=in_channels,
                          out_channels=c3[0],
                          kernel_size=1)
        self.p3_2=nn.Conv2d(in_channels=c3[0],
                          out_channels=c3[1],
                          kernel_size=5,
                          padding=2)
        # 路线4：3x3最大池化+1x1卷积
        self.p4_1=nn.MaxPool2d(kernel_size=3,
                             stride=1,
                             padding=1)
        self.p4_2=nn.Conv2d(in_channels=in_channels,
                          out_channels=c4,
                          kernel_size=1)
    def forward(self,x):
        p1=self.ReLU(self.p1_1(x))
        p2=self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3=self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4=self.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)  # 在通道维度上进行拼接，合并
class GoogLeNet(nn.Module):
    def __init__(self,Inception):
        super(GoogLeNet,self).__init__()
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3),                   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        )
        self.b2=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=192,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        )
        self.b3=nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        )
        self.b4=nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(128,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3, 
                         stride=2, 
                         padding=1)
        )
        self.b5=nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),   # 全局池化层,(1,1)表示输出的大小为1x1的特征图，操作是把所有的数据进行加和平均
            nn.Flatten(),  
            nn.Dropout(0.4),
            nn.Linear(1024,10)
        )
        for m in self.modules():  # 用于初始化权重，默认使用kaiming_uniform_，防止梯度消失或爆炸
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        return x
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=GoogLeNet(Inception).to(device)
    print(summary(model,(1,224,224)))