import os
import numpy as np
import PIL.Image as Image
folder_path='./data_cat_dog'
# 初始化文件夹中的图片文件
total_pixels=0
sum_normalized_pixels_values=np.zeros(3)
for root,dirs,files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg','.png','.jpeg','bmp')):
            image_path=os.path.join(root,filename)
            image=Image.open(image_path)
            image_array=np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array=image_array/255.0
            # print(image_path)
            # print(normalize_image_array.shape)
            # 积累归一化后的像素值与像素数量
            total_pixels+=normalized_image_array.size
            sum_normalized_pixels_values+=np.sum(normalized_image_array,(0,1))
# 计算平均值与标准差
mean=sum_normalized_pixels_values/total_pixels
sum_squared_diff=np.zeros(3)
for root,dirs,files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg','.png','.jpeg','bmp')):
            image_path=os.path.join(root,filename)
            image=Image.open(image_path)
            image_array=np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array=image_array/255.0
            # print(normalized_image_array.shape)
            # print(mean.shape)
            # print(image_path)
            try:
                diff=(normalized_image_array-mean)**2
                sum_squared_diff+=np.sum(diff,(0,1))
            except:
                print('error')
            # diff=(normalized_image_array-mean)**2
            # sum_squared_diff+=np.sum(diff,(0,1))
variance=sum_squared_diff/total_pixels
print('mean:',mean)
print('variance:',variance)