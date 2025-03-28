import torch
from torchvision import transforms
from PIL import Image
from model import AlexNet

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((32,32)),  
                                    transforms.ToTensor()       
    ])
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = transform(image)  
    image = image.unsqueeze(0)  # 增加一个维度，模拟batch维度
    return image

def test_model_process(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device)  
    model.eval()  
    image = image.to(device)
    with torch.no_grad(): 
        output = model(image) 
        pre_lab = torch.argmax(output, dim=1)  
        return pre_lab.item()  

if __name__ == '__main__':
    # 定义模型并加载最优模型参数
    model = AlexNet()
    model.load_state_dict(torch.load('./best_model_params.pth'))
    # 加载并预处理图片
    image_path = './my_photo3.png'  # 替换为你的图片路径
    image = load_and_preprocess_image(image_path)
    # 测试模型
    predicted_label = test_model_process(model, image)
    classes=['T 恤/上衣', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    print(f'预测的标签为：{classes[predicted_label]}，标签序号为：{predicted_label}')