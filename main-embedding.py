import os
import parser
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import argparse
# from pandas_ml import ConfusionMatrix

import torch.nn.functional as F

# from model import resnet50

# from models.getModel import get_encoder
from resnet import resnet50

import random

import numpy as np
from sklearn.metrics import confusion_matrix


# from semodel import resnet50

import matplotlib.pyplot as plt

from torch.nn import DataParallel

import torch.nn.init as nn_init

from os import path
from PIL import Image
import numpy as np
import pandas as pd


import datetime
import time
from utils_loss import FocalLoss

from fusion_multi_model import Build_MultiModel_smile_embedding

from dataloader.image_transforms import Image_Transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 计算回归指标的函数
def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    corr = r2_score(targets, predictions)
    return mse, mae, rmse, corr

def writefile(name, list):
    # print(list)

    f = open(name+'.txt', mode='w')  # 打开文件，若文件不存在系统自动创建。
    # f.write(Loss_list)  # write 写入
    for i in range(len(list)):
        s = str(list[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
        f.write(s)
    f.close()



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


#这个程序要重新改，改成从csv分别读取test和train信息







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
        im_labels_dia=df['dia_target'].to_numpy()
        im_labels_offset=df['offset_target'].to_numpy()
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
    return train_loaders[0]







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_type", "-mt", type=str, default="avg", help="fusion type  avg linear")    
    parser.add_argument("--fold", "-fold", type=str, default="4", help="")    
    parser.add_argument('--gpuid',default=4, type=int)
    parser.add_argument('--target_type',default="dia_target", type=str, help="dia_target  offset_target")
    parser.add_argument('--fusionmode',default=3, type=int, help="0 1 2 3")
    parser.add_argument("--backbone", "-bk", type=str, default="ResNet50", help="backbone  ResNet34 ResNet50 ResNet50BAM  ResNet50CBAM  ResNet50SE  Alexnet")
    args = parser.parse_args()
    taskname='smile'
    fold=args.fold
    backbone=args.backbone
    gpuid=args.gpuid
    savename= args.target_type+"_"+str(args.fusionmode)+"-embedd-qulv_valauc"
    path='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Smile-dia/'+savename
    mkdir(path)
    save_path_best = path+'/'+taskname+'_best.pth'


    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    target_type=args.target_type

    print(target_type)
    print(args.fusionmode)



    label_path='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Smile-dia/label'
    IMAGE_PATH = '/data/gaowh/data/files/EOZ_smile/'
    TRAIN_LISTS = ['train.csv']
    VAL_LISTS = ['val.csv']
    TEST_LISTS = ['test.csv']
    CLIVAL_LISTS = ['cli_val.csv']
    CLASSES = []
    batch_size = 8
    print(batch_size)
    # 4 load dataset
    train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


    # Create training and test loaders
    train_loader = load_data_multi(label_path, TRAIN_LISTS, IMAGE_PATH,CLASSES,batch_size, train_transforms,'train')
    test_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH,CLASSES,batch_size, val_transforms,'test')
    val_loader = load_data_multi(label_path, VAL_LISTS, IMAGE_PATH,CLASSES,batch_size, val_transforms,'test')
    clival_loader = load_data_multi(label_path, CLIVAL_LISTS, IMAGE_PATH,CLASSES,batch_size, val_transforms,'test')


    dfres = pd.read_csv(path_join(label_path, TEST_LISTS[0]))
    val_num=dfres.shape[0]

    dftrain = pd.read_csv(path_join(label_path, TRAIN_LISTS[0]))

    train_num=dftrain.shape[0]

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    net = Build_MultiModel_smile_embedding(backbone=backbone,fusionmode=args.fusionmode)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[gpuid])

    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001, amsgrad=True, weight_decay=0.0001)

    Loss_list = []
    Loss_list_val = []

    Accuracy_list = []
    Accuracy_list_val = []

    epochs =25
    best_auc = 0.0
    train_steps = len(train_loader)


    start_time=time.time()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0  #
        val_loss=0.0

        train_bar = tqdm(train_loader)

        for count, (input1_img,dw_img,xy_img,target_dia,target_offset,head,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15) in enumerate(train_bar):

            optimizer.zero_grad()
            if target_type=="dia_target":

                target = target_dia.to(device)
            else:
                target = target_offset.to(device)
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
            target=target.view(-1)
            logits = net(input1_img,dw_img,xy_img,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15)

            loss = loss_function(logits, target.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
            predict_y = torch.max(logits, dim=1)[1]

            train_acc += torch.eq(predict_y, target.to(device)).sum().item()
        train_accurate = train_acc/train_num
        Accuracy_list.append(train_accurate)


        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        test_loss = 0.0  # accumulate validation loss
        test_steps = 0  # count the number of validation steps
        predict_all = []
        gt_all = []
        predict_probs_all = []

        with torch.no_grad():
            test_bar = tqdm(val_loader)
            for test_data in test_bar:
                input1_img,dw_img,xy_img,target_dia,target_offset,head,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15 = test_data

                if target_type=="dia_target":

                    target = target_dia.to(device)
                else:
                    target = target_offset.to(device)
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

                outputs = net(input1_img,dw_img,xy_img,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15)
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels).sum().item()
                loss_val = loss_function(outputs, test_labels)
                test_loss += loss_val.item()
                test_steps += 1

                gt_all.extend(test_labels.cpu().tolist())
                predict_all.extend(predict_y.cpu().tolist())

                predict_probs_all.extend(probs[:, 1].tolist())

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)

        test_accurate = acc / len(test_loader.dataset)

        valauc = roc_auc_score(gt_all, predict_probs_all)
        print("val AUC: {:.4f}".format(valauc))

        report = classification_report(gt_all, predict_all)
        print(report)

        f1 = f1_score(gt_all, predict_all, average='macro')
        print("val Macro F1: {:.4f}".format(f1))
        cm = confusion_matrix(np.array(gt_all), np.array(predict_all))
        print("val Confusion Matrix:")
        print(cm)
        print('val accuracy: %.3f' % (test_accurate))

        acc = 0.0  # accumulate accurate number / epoch

        test_loss = 0.0  # accumulate validation loss
        test_steps = 0  # count the number of validation steps
        predict_all = []
        gt_all_test = []
        predict_probs_all_test = []

        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                input1_img,dw_img,xy_img,target_dia,target_offset,head,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15 = test_data

                if target_type=="dia_target":

                    target = target_dia.to(device)
                else:
                    target = target_offset.to(device)
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

                outputs = net(input1_img,dw_img,xy_img,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15)
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels).sum().item()
                loss_val = loss_function(outputs, test_labels)
                test_loss += loss_val.item()
                test_steps += 1

                gt_all_test.extend(test_labels.cpu().tolist())
                predict_all.extend(predict_y.cpu().tolist())

                predict_probs_all_test.extend(probs[:, 1].tolist())

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)
        test_accurate = acc / len(test_loader.dataset)


        testauc = roc_auc_score(gt_all_test, predict_probs_all_test)
        print("test AUC: {:.4f}".format(testauc))

        report = classification_report(gt_all_test, predict_all)
        print(report)

        f1 = f1_score(gt_all_test, predict_all, average='macro')
        print("Test Macro F1: {:.4f}".format(f1))
        cm = confusion_matrix(np.array(gt_all_test), np.array(predict_all))
        print("Test Confusion Matrix:")
        print(cm)

        print('test_accuracy: %.3f' % (test_accurate))


        acc = 0.0  # accumulate accurate number / epoch

        test_loss = 0.0  # accumulate validation loss
        test_steps = 0  # count the number of validation steps
        predict_all = []
        gt_all_val = []
        predict_probs_all_val = []

        with torch.no_grad():
            test_bar = tqdm(clival_loader)
            for test_data in test_bar:
                input1_img,dw_img,xy_img,target_dia,target_offset,head,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15 = test_data

                if target_type=="dia_target":

                    target = target_dia.to(device)
                else:
                    target = target_offset.to(device)

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

                outputs = net(input1_img,dw_img,xy_img,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15)
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels).sum().item()
                loss_val = loss_function(outputs, test_labels)
                test_loss += loss_val.item()
                test_steps += 1

                gt_all_val.extend(test_labels.cpu().tolist())
                predict_all.extend(predict_y.cpu().tolist())

                predict_probs_all_val.extend(probs[:, 1].tolist())  # 假设类别1是正类

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)

        test_accurate = acc / len(clival_loader.dataset)


        cliauc = roc_auc_score(gt_all_val, predict_probs_all_val)
        print("clival_loader AUC: {:.4f}".format(cliauc))

        report = classification_report(gt_all_val, predict_all)
        print(report)

        f1 = f1_score(gt_all_val, predict_all, average='macro')
        print("clival_loader Macro F1: {:.4f}".format(f1))
        cm = confusion_matrix(np.array(gt_all_val), np.array(predict_all))
        print("clival_loader Confusion Matrix:")
        print(cm)

        print('clival_loader accuracy: %.3f' % (test_accurate))
        data_clival = {
            "gt_all": gt_all_val,
            "predict_probs_all": predict_probs_all_val
        }
        df_clival = pd.DataFrame(data_clival)

        data_test = {
            "gt_all": gt_all_test,
            "predict_probs_all": predict_probs_all_test
        }
        df_test = pd.DataFrame(data_test)

        currentauc=testauc+cliauc
        if valauc > best_auc:
            best_auc = valauc
            print("bestauc")

            torch.save(net.state_dict(), save_path_best)
            csv_file = path+"/roc_data_test.csv"
            df_test.to_csv(csv_file, index=False)

            csv_file = path+"/roc_data_clival.csv"
            df_clival.to_csv(csv_file, index=False)



    print('Finished Training')








if __name__ == '__main__':
    main()