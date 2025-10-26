
from sklearn.metrics import roc_curve, auc


# import pandas as pd
# from os import *
import torch.nn.functional as F
import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from fusion_multi_model import Build_MultiModel_smile
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from fusion_multi_model import Build_MultiModel_smile_embedding


import numpy as np


import matplotlib.pyplot as plt

from torch.nn import DataParallel

import torch.nn.init as nn_init

from os import path
from PIL import Image
import numpy as np
import pandas as pd


# from fusion_multi_model import Build_MultiModel_szzyy_pair_onlyimg

from dataloader.image_transforms import Image_Transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

class CustomDataset_Multi(torch.utils.data.Dataset):
    def __init__(self, img_id,im_labels_dia,im_labels_offset,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15,img_path,im_transforms=None):





        self.im_dir = img_path
        self.im_path_head=img_id

        self.im_labels_dia = im_labels_dia
        self.im_labels_offset = im_labels_offset

        self.im_1 = im_1
        self.im_2 = im_2
        self.im_3 = im_3
        self.im_4 = im_4
        self.im_5 = im_5
        self.im_6 = im_6
        self.im_7 = im_7
        self.im_8 = im_8
        self.im_9 = im_9
        self.im_10 = im_10
        self.im_11 = im_11
        self.im_12 = im_12
        self.im_13 = im_13
        self.im_14 = im_14
        self.im_15 = im_15

# im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15



# im_jiaomozhijing, im_a, im_k2, im_k1, im_sex, im_k2zhou, im_zuibodianjiaomohoudu, im_RST_um


        
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):

        return len(self.im_labels_offset)

    def __getitem__(self, idx):

        # seed = 123
        headname=str(self.im_path_head[idx])
        input1 = os.path.join(self.im_dir,headname,headname+'_input2.jpg') 
        dw= os.path.join(self.im_dir,headname,headname+'_dw.png')
        xy = os.path.join(self.im_dir,headname,headname+'_xy.png')


        try:
            input1_img = Image.open(input1).convert('RGB')
            input1_img = self.im_transforms(input1_img)
        except:
            print('Error: Failed to open or verify image file {}'.format(input1))


        try:
            dw_img = Image.open(dw).convert('RGB')
            dw_img = self.im_transforms(dw_img)
        except:
            print('Error: Failed to open or verify image file {}'.format(dw))


        try:
            xy_img = Image.open(xy).convert('RGB')
            xy_img = self.im_transforms(xy_img)
        except:
            print('Error: Failed to open or verify image file {}'.format(xy))


        # return input1_img,dw_img,xy_img , self.im_labels_dia[idx] ,self.im_labels_offset[idx] , headname, self.im_jinshi[idx],self.im_sanguang[idx] ,self.im_sanguangzhou[idx],self.im_toujinghoudu[idx] ,self.im_s[idx],    self.im_jiaomozhijing[idx], self.im_a[idx],    self.im_k2[idx],self.im_k1[idx],    self.im_sex[idx],self.im_k2zhou[idx],    self.im_zuibodianjiaomohoudu[idx],self.im_RST_um[idx]
        return input1_img, dw_img, xy_img, \
            self.im_labels_dia[idx], self.im_labels_offset[idx], \
            headname, \
            self.im_1[idx], self.im_2[idx], self.im_3[idx], self.im_4[idx], self.im_5[idx], self.im_6[idx], self.im_7[idx], self.im_8[idx], self.im_9[idx], self.im_10[idx], self.im_11[idx], self.im_12[idx], self.im_13[idx], self.im_14[idx], self.im_15[idx]


def load_data_multi(label_path, train_lists, img_path,CLASSES,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []


    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
# Area_mm2,Diameter_mm,Offset_mm
        im_labels_dia=df['dia_target'].to_numpy()
        im_labels_offset=df['offset_target'].to_numpy()


        # im_sanguang=df['sanguang'].to_numpy()
        # im_toujinghoudu=df['toujinghoudu'].to_numpy()


        # im_s=df['S'].to_numpy()
        # im_sanguangzhou=df['sanguangzhou'].to_numpy()
        # im_jinshi=df['jinshi'].to_numpy()

        # im_jiaomozhijing=df['jiaomozhijing'].to_numpy()

        # im_a=df['A'].to_numpy()
        # im_k2=df['K1'].to_numpy()
        # im_k1=df['K2'].to_numpy()
        # im_sex=df['sex'].to_numpy()
        # im_k2zhou=df['K2zhou'].to_numpy()
        # im_zuibodianjiaomohoudu=df['zuibodianjiaomohoudu'].to_numpy()
        # im_RST_um=df['RST_um'].to_numpy()
        # im_c=df['C'].to_numpy()

        im_8 = df['Corneal Diameter(mm)'].to_numpy()

        im_9 = df['Input Cylinder(D)'].to_numpy()
        im_13 = df['Steep Keratometry'].to_numpy()
        im_2 = df['Cylinder(D)'].to_numpy()
        im_3 = df['Axis of Cylinder'].to_numpy()
        im_12 = df['Flat Keratometry'].to_numpy()


        im_1 = df['Sphere(D)'].to_numpy()
        im_5 = df['Input Sphere(D)'].to_numpy()
        im_7 = df['Age'].to_numpy()
        im_10 = df['Input Axis'].to_numpy()



        im_4 = df['Lenticule Thickness(um)'].to_numpy()
        im_6 = df['Gender'].to_numpy()
        im_14 = df['Axis of Steep Keratometry'].to_numpy()
        im_15 = df['Residual Stromal Thickness(um)'].to_numpy()

        im_11 = df['Central Corneal Thickness(um)'].to_numpy()



        img_id=df['id'].to_numpy()



    
        train_sets.append(CustomDataset_Multi(img_id,im_labels_dia,im_labels_offset ,  im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15, img_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        # print('Size for {0} = {1}'.format(train_list, len(hb_im_names)))

        #这里的问题，因为原来的数据分了很多歌批次，不同的csv。这个就一个csv。原来是trainloader 数组拼出来。现在就取0就好了

    return train_loaders[0]








parser = argparse.ArgumentParser()


parser.add_argument("--task_id", "-id", type=str, default="1", help="5fold id")
parser.add_argument("--fusion_type", "-mt", type=str, default="avg", help="fusion type")    

parser.add_argument('--fusionmode',default=3, type=int, help="0 1 2 3")
parser.add_argument('--taskname',default="dia_target_2", type=str, )
parser.add_argument('--target_type',default="dia_target", type=str, help="dia_target  offset_target")

parser.add_argument("--backbone", "-bk", type=str, default="ResNet50", help="backbone  ResNet34 ResNet50 ResNet50BAM  ResNet50CBAM  ResNet50SE  Alexnet")    

args = parser.parse_args()

target_type=args.target_type

print(target_type)

backbone=args.backbone
print(backbone)

fusion_type=args.fusion_type
print(fusion_type)

taskname=args.taskname
print(taskname)

# 0.6066
key="qulv-embed"

thispath='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/dia_target_3-embedd-qulv_valauc'

weightpath='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/dia_target_3-embedd-qulv_valauc/rzzz_best.pth'



label_path='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/'

# cli test val
dataset="cli"



IMAGE_PATH = '/data/gaowh/data/files/EOZ_smile/'




# cli_val test val  
TEST_LISTS = ['cli_val.csv']

CLASSES = ['AGE','GXY','TNB','GXZ','TXBGAS','GNSXZ','SMOKE','DRINK']






train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 8

# Create training and test loaders

validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH,CLASSES,batch_size, val_transforms,'test')






model = Build_MultiModel_smile_embedding(backbone=backbone,fusionmode=args.fusionmode)

 



if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0])

 
model.to(device)

model.load_state_dict(
    torch.load(weightpath))

# 设置模型为评估模式
model.eval()


acc = 0.0  # accumulate accurate number / epoch

test_loss = 0.0  # accumulate validation loss
test_steps = 0  # count the number of validation steps
predict_all = []
gt_all = []
predict_probs_all = []
Accuracy_list_val = []
ids=[]

with torch.no_grad():
    test_bar = tqdm(validate_loader)
    for test_data in test_bar:
        input1_img,dw_img,xy_img,target_dia,target_offset,head,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15 = test_data



        if target_type=="dia_target":

            target = target_dia.to(device)
        else:
            target = target_offset.to(device)

        # 将特征转换为Double类型
        im_1 = im_1.to(device).double()
        im_2 = im_2.to(device).double()
        im_3 = im_3.to(device).double()
        im_4 = im_4.to(device).double()
        im_5 = im_5.to(device).double()
        im_6 = im_6.to(device).double()
        im_7 = im_7.to(device).double()
        im_8 = im_8.to(device).double()
        im_9 = im_9.to(device).double()
        im_10 = im_10.to(device).double()
        im_11 = im_11.to(device).double()
        im_12 = im_12.to(device).double()
        im_13 = im_13.to(device).double()
        im_14 = im_14.to(device).double()
        im_15 = im_15.to(device).double()



        test_labels = target.view(-1).to(device)

        outputs = model(input1_img,dw_img,xy_img,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15)
        probs = torch.softmax(outputs, dim=1)

        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, test_labels).sum().item()

        gt_all.extend(test_labels.cpu().tolist())
        predict_all.extend(predict_y.cpu().tolist())

        # 确保提取所有类别的概率
        predict_probs_all.extend(probs[:, 1].tolist())  # 假设类别1是正类
        ids.extend(head)  # 假设类别1是正类


# 计算验证集准确率
test_accurate = acc / len(validate_loader.dataset)
Accuracy_list_val.append(test_accurate)

# 计算多分类AUC
auc = roc_auc_score(gt_all, predict_probs_all)
print("test AUC: {:.4f}".format(auc))

# 生成分类报告
report = classification_report(gt_all, predict_all)
print(report)

# 计算宏F1分数
f1 = f1_score(gt_all, predict_all, average='macro')
print("Test Macro F1: {:.4f}".format(f1))



# 计算混淆矩阵
cm = confusion_matrix(np.array(gt_all), np.array(predict_all))
print("Test Confusion Matrix:")
print(cm)

print('test_accuracy: %.4f' % (test_accurate))



data = {
    'id':ids,
    "Groundtruth": gt_all,
    "Probability": predict_probs_all
}
df = pd.DataFrame(data)

# 保存为 CSV 文件



csv_file = thispath+"/"+key+"_"+dataset+".csv"  # 你可以修改文件名
df.to_csv(csv_file, index=False)

