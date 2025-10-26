
from sklearn.metrics import roc_curve, auc


# import pandas as pd
# from os import *
import torch.nn.functional as F
import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


from resnet import resnet50

import numpy as np


import matplotlib.pyplot as plt

from torch.nn import DataParallel

import torch.nn.init as nn_init

from os import path
from PIL import Image
import numpy as np
import pandas as pd



from dataloader.image_transforms import Image_Transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fusion_multi_model import Build_MultiModel_szzyy_pair_onlyimg_convnext

import numpy as np
from sklearn.metrics import confusion_matrix
from pycm import *
import argparse

def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False




path_join = path.join





class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, im_dir, im_names, im_labels, im_extra,im_path,im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        self.im_path_head=im_path

        self.im_extra=im_extra
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):

        return len(self.im_labels)

    def __getitem__(self, idx):



        image_list=self.im_names[idx].split(';')
        img_list = [os.path.join(self.im_dir, string) for string in image_list]

        images = []

        for image_path in img_list:
            try:
                im = Image.open(image_path).convert('RGB')
                im = self.im_transforms(im)
                images.append(im)
            except:
                print('Error: Failed to open or verify image file {}'.format(image_path))



        return images, self.im_labels[idx], self.im_path_head[idx],self.im_extra[idx]


 

def load_data(label_path, train_lists, img_path,classes,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []




    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['IMAGE'].to_numpy()

        im_labels=df['TYPE'].to_numpy()
        im_path=df['ID'].to_numpy()
        df[classes].iloc[:, 0] /= 100

        im_extra = torch.tensor(df[classes].to_numpy(), dtype=torch.float)

    
        train_sets.append(CustomDataset(img_path, im_names, im_labels ,im_extra, im_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))

        #这里的问题，因为原来的数据分了很多歌批次，不同的csv。这个就一个csv。原来是trainloader 数组拼出来。现在就取0就好了

    return train_loaders[0]


parser = argparse.ArgumentParser()


parser.add_argument("--task_id", "-id", type=str, default="1", help="5fold id")
parser.add_argument("--model_type", "-mt", type=str, default="avg", help="fusion type")    
args = parser.parse_args()



modeltype=args.model_type
print(modeltype)

taskname=args.task_id
print(taskname)

thispath='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/test_res/ConvNeXt/'+modeltype+'/'+taskname+'/'

weightpath='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/test_res/ConvNeXt/'+modeltype+'/'+taskname+'/'+taskname+'_best.pth'



label_path='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/5fold/'+taskname


IMAGE_PATH = '/vepfs/gaowh/sz_zyy_new/'

TRAIN_LISTS = [taskname+'_train.csv']
TEST_LISTS = [taskname+'_test.csv']

CLASSES = ['AGE','GXY','TNB','GXZ','TXBGAS','GNSXZ','SMOKE','DRINK']


# IMAGE_PATH = '/vepfs/gaowh/sz_zyy_data/'

# TRAIN_LISTS = ['pair_train.csv']
# TEST_LISTS = ['pair_test.csv']

# CLASSES = ['AGE','GXY','TNB','GXZ','TXBGAS','GNSXZ','SMOKE','DRINK']







train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32

# Create training and test loaders
validate_loader = load_data(label_path, TEST_LISTS, IMAGE_PATH,CLASSES, batch_size, val_transforms,'test')

train_loader = load_data(label_path, TRAIN_LISTS, IMAGE_PATH,CLASSES, batch_size, train_transforms,'train')





model = Build_MultiModel_szzyy_pair_onlyimg_convnext(backbone=modeltype)




if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])

 
model.to(device)

model.load_state_dict(
    torch.load(weightpath))

# 设置模型为评估模式
model.eval()


# 存储预测概率和真实标签
predictions = []
labels = []

# 对测试集进行推断
with torch.no_grad():
    for val_images,val_labels,_,im_extra in validate_loader:
        outputs = model(val_images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        
        predictions.extend(probs[:, 1].tolist())  # 获取正例的预测概率
        labels.extend(val_labels.tolist())

# 计算TPR和FPR
fpr, tpr, thresholds = roc_curve(labels, predictions)



# 创建一个DataFrame对象
data = pd.DataFrame({'Labels': labels, 'Predictions': predictions})

# 保存为CSV文件
data.to_csv(thispath+taskname+'.csv', index=False)  # 指定保存路径和文件名，并设置index=False以避免保存索引列

threshold = 0.5


predictions = np.array(predictions)


binary_predictions = np.where(predictions > threshold, 1, 0)

conf_matrix = ConfusionMatrix(actual_vector=labels, predict_vector=binary_predictions)

filename = thispath+ "confusion_matrix.txt"

delimiter = "\t"  # 分隔符可以自行设置

# 获取混淆矩阵的 NumPy 数组表示
# matrix = conf_matrix.matrix

# 保存混淆矩阵为文本文件
# np.savetxt(filename, matrix, delimiter=delimiter)

matrix_str = str(conf_matrix)

# 保存混淆矩阵为文本文件
with open(filename, "w") as f:
    f.write(matrix_str)


# 计算AUC
auc_score = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
# plt.show()
plt.savefig(thispath+'roc_curve.png')  # 指定保存路径和文件名
