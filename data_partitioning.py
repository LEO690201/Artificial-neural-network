import os
import random
import shutil

def mkfile(file):                  # 递归创建目录
    if not os.path.exists(file):  # 判断是否存在目录
        os.makedirs(file)         # 创建目录
# 获取data文件夹下所有文件名
file_path='./cat_dog'
flower_class=[cla for cla in os.listdir(file_path)]      # 获取data文件夹下所有类别名
# 创建训练集train文件夹并且由类名创建5个子文件夹
mkfile('data_cat_dog/train')
for cla in flower_class:
    mkfile('data_cat_dog/train/'+cla)
# 创建验证集val文件夹并且由类名创建5个子文件夹
mkfile('data_cat_dog/test')
for cla in flower_class:
    mkfile('data_cat_dog/test/'+cla)
# 划分比例：
split_rate=0.1
# 遍历所有类别的全部图像并按比例划分为训练集和验证集
for cla in flower_class:
    cla_path=file_path+'/'+cla+'/'   # 某一类别的子目录
    images=os.listdir(cla_path)
    num=len(images)
    eval_index=random.sample(images,k=int(num*split_rate))
    for index,image in enumerate(images):
        # eval_index中保存验证集val的图像名称
        if image in eval_index:
            image_path=cla_path+image
            new_path='data_cat_dog/test/'+cla
            shutil.copy(image_path,new_path)
        else:
            image_path=cla_path+image
            new_path='data_cat_dog/train/'+cla
            shutil.copy(image_path,new_path)
        print('\r[{}] processing [{}/{}]'.format(cla,index+1,num),end="")
    print()
print('processing done!')
