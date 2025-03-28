import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet

def test_data_process():   # 定义训练集和验证集的处理函数
    test_data=FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                            download=False)
    test_dataloader=Data.DataLoader(dataset=test_data,
                                     batch_size=1,      # 可以一张一张测试
                                     shuffle=True,
                                     num_workers=0)   # 训练集的DataLoader(划分),num_workers=0表示使用0个进程来加载数据
    return test_dataloader

# test_dataloader=test_data_process()   # 调试是否有问题

def test_model_process(model,test_dataloader):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 选择设备
    model.to(device)   # 将模型加载到设备上

    # 初始化参数
    test_corrects=0.0
    test_num=0
    
    with torch.no_grad():   # 关闭梯度计算,测试中仅仅进行前向传播
        for test_data_x,test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x=test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y=test_data_y.to(device)
            
            model.eval()   # 设置模型为验证模式，此时的特点为：不启用dropout和batchnorm
            output=model(test_data_x)  # 前向传播,输入为测试数据集，输出为对每个样本的预测
            
            pre_lab=torch.argmax(output,dim=1)  # 预测的标签

            test_corrects+=torch.sum(pre_lab==test_data_y.data)  # 统计预测正确的个数
            test_num+=test_data_y.size(0)   # 统计测试集的样本个数

    test_acc=test_corrects/test_num   # 计算测试集的准确率
    print('测试集准确率为：{:.2f}%'.format(test_acc*100))

if __name__=='__main__':
    # 定义模型,将模型实例化
    model=AlexNet()
    model.load_state_dict(torch.load('./best_model_params.pth'))  # 加载最优模型参数

    test_loader=test_data_process()   # 加载测试集数据
    test_model_process(model,test_loader)   # 测试模型


# 87.52%


    # # 测试集推理过程可视化
    # device='cuda' if torch.cuda.is_available() else 'cpu'   # 选择设备
    # model.to(device)   # 将模型加载到设备上
    # classes=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # with torch.no_grad():
    #     for b_x,b_y in test_loader:
    #         b_x=b_x.to(device)
    #         b_y=b_y.to(device)
            
    #         # 设置为验证模式
    #         model.eval()
    #         output=model(b_x)
    #         pre_lab=torch.argmax(output,dim=1)
    #         result=pre_lab.item()  # 将其转换为数字
    #         label=b_y.item()
    #         print('预测值： ',classes[result],'~~~~~~ 真实值： ',classes[label])







