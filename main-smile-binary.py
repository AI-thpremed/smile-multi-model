import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import argparse
from sklearn.metrics import confusion_matrix
from os import path
from PIL import Image
import numpy as np
import pandas as pd
import time
from fusion_multi_model import Build_MultiModel_smile
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





path_join = path.join


class CustomDataset_Multi(torch.utils.data.Dataset):
    def __init__(self, img_id,im_labels_dia,im_labels_offset, img_path,im_transforms=None):
        self.im_dir = img_path
        self.im_path_head=img_id

        self.im_labels_dia = im_labels_dia
        self.im_labels_offset = im_labels_offset
        
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


        return input1_img,dw_img,xy_img , self.im_labels_dia[idx] ,self.im_labels_offset[idx] , headname


def load_data_multi(label_path, train_lists, img_path,CLASSES,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []


    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_labels_dia=df['dia_target'].to_numpy()
        im_labels_offset=df['dia_target'].to_numpy()


        img_id=df['id'].to_numpy()

        train_sets.append(CustomDataset_Multi(img_id,im_labels_dia,im_labels_offset , img_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))

    return train_loaders[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_type", "-mt", type=str, default="avg", help="fusion type  avg linear")    
    parser.add_argument("--fold", "-fold", type=str, default="4", help="")    
    parser.add_argument('--gpuid',default=0, type=int)
    parser.add_argument('--target_type',default="dia_target", type=str, help="dia_target  offset_target")
    parser.add_argument('--fusionmode',default=2, type=int, help="0 1 2 3")
    parser.add_argument("--backbone", "-bk", type=str, default="ResNet50", help="backbone  ResNet34 ResNet50 ResNet50BAM  ResNet50CBAM  ResNet50SE  Alexnet")
    args = parser.parse_args()
    taskname='dia'
    fold=args.fold
    backbone=args.backbone
    gpuid=args.gpuid
    savename= args.target_type+"_"+str(args.fusionmode)
    path='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/'+savename

    mkdir(path)

    save_path_best = path+'/'+taskname+'_best.pth'


    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    target_type=args.target_type

    print(target_type)
    print(args.fusionmode)

    label_path='/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label'

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
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    net = Build_MultiModel_smile(backbone=backbone,fusionmode=args.fusionmode)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[gpuid])

    net.to(device)


    loss_function = nn.CrossEntropyLoss()


    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)

    optimizer = optim.Adam(params, lr=0.0001, amsgrad=True, weight_decay=0.0001)

    epochs =25
    best_auc = 0.0
    train_steps = len(train_loader)

    start_time=time.time()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0


        train_acc = 0.0  #





        train_bar = tqdm(train_loader)


        for count, (input1_img,dw_img,xy_img,target_dia,target_offset,head) in enumerate(train_bar):

            optimizer.zero_grad()
            if target_type=="dia_target":

                target = target_dia.to(device)
            else:
                target = target_offset.to(device)


            target=target.view(-1)

            logits = net(input1_img,dw_img,xy_img)

            loss = loss_function(logits, target.to(device))

            # logits = net(images.to(device))  #这个就是结果
            loss = loss_function(logits, target.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            predict_y = torch.max(logits, dim=1)[1]

            train_acc += torch.eq(predict_y, target.to(device)).sum().item()



        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        test_loss = 0.0  # accumulate validation loss
        test_steps = 0  # count the number of validation steps
        predict_all = []
        gt_all = []
        predict_probs_all = []
        val_ids = []

        with torch.no_grad():
            test_bar = tqdm(val_loader)
            for test_data in test_bar:
                input1_img,dw_img,xy_img,target_dia,target_offset,head = test_data

                if target_type=="dia_target":

                    target = target_dia.to(device)
                else:
                    target = target_offset.to(device)


                test_labels = target.view(-1).to(device)

                outputs = net(input1_img,dw_img,xy_img)
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels).sum().item()
                loss_val = loss_function(outputs, test_labels)
                test_loss += loss_val.item()
                test_steps += 1

                gt_all.extend(test_labels.cpu().tolist())
                predict_all.extend(predict_y.cpu().tolist())
                val_ids.extend(head)  

                # 确保提取所有类别的概率
                predict_probs_all.extend(probs[:, 1].tolist())  # 假设类别1是正类

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)

        # # 计算验证集准确率
        test_accurate = acc / len(val_loader.dataset)
        # Accuracy_list_val.append(test_accurate)

        # 计算多分类AUC
        valauc = roc_auc_score(gt_all, predict_probs_all)
        print("val AUC: {:.4f}".format(valauc))

        # 生成分类报告
        report = classification_report(gt_all, predict_all)
        print(report)

        # 计算宏F1分数
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
        test_ids = []

        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                input1_img,dw_img,xy_img,target_dia,target_offset,head = test_data

                if target_type=="dia_target":

                    target = target_dia.to(device)
                else:
                    target = target_offset.to(device)


                test_labels = target.view(-1).to(device)

                outputs = net(input1_img,dw_img,xy_img)
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels).sum().item()
                loss_val = loss_function(outputs, test_labels)
                test_loss += loss_val.item()
                test_steps += 1

                gt_all_test.extend(test_labels.cpu().tolist())
                predict_all.extend(predict_y.cpu().tolist())
                test_ids.extend(head)  

                # 确保提取所有类别的概率
                predict_probs_all_test.extend(probs[:, 1].tolist())  # 假设类别1是正类

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)

        # # 计算验证集准确率
        test_accurate = acc / len(test_loader.dataset)


        # 计算多分类AUC
        auc = roc_auc_score(gt_all_test, predict_probs_all_test)
        print("test AUC: {:.4f}".format(auc))

        # 生成分类报告
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
        cli_ids = []

        with torch.no_grad():
            test_bar = tqdm(clival_loader)
            for test_data in test_bar:
                input1_img,dw_img,xy_img,target_dia,target_offset,head = test_data

                if target_type=="dia_target":

                    target = target_dia.to(device)
                else:
                    target = target_offset.to(device)


                test_labels = target.view(-1).to(device)

                outputs = net(input1_img,dw_img,xy_img)
                probs = torch.softmax(outputs, dim=1)

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels).sum().item()
                loss_val = loss_function(outputs, test_labels)
                test_loss += loss_val.item()
                test_steps += 1

                gt_all_val.extend(test_labels.cpu().tolist())
                predict_all.extend(predict_y.cpu().tolist())
                cli_ids.extend(head)  


                predict_probs_all_val.extend(probs[:, 1].tolist())  # 假设类别1是正类

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)

        # # 计算验证集准确率
        test_accurate = acc / len(clival_loader.dataset)


        auc = roc_auc_score(gt_all_val, predict_probs_all_val)
        print("clival_loader AUC: {:.4f}".format(auc))

        report = classification_report(gt_all_val, predict_all)
        print(report)

        f1 = f1_score(gt_all_val, predict_all, average='macro')
        print("clival_loader Macro F1: {:.4f}".format(f1))

        cm = confusion_matrix(np.array(gt_all_val), np.array(predict_all))
        print("clival_loader Confusion Matrix:")
        print(cm)

        print('clival_loader accuracy: %.3f' % (test_accurate))

        data_val = {
            'id':val_ids,

            "Groundtruth": gt_all,
            "Probability": predict_probs_all
        }
        df_val = pd.DataFrame(data_val)


        data_test = {
            'id':test_ids,

            "Groundtruth": gt_all_test,
            "Probability": predict_probs_all_test
        }
        df_test = pd.DataFrame(data_test)

        data_clival ={
            'id':cli_ids,
        
            "Groundtruth": gt_all_val,
            "Probability": predict_probs_all_val
        }
        df_clival = pd.DataFrame(data_clival)

 

        if valauc > best_auc:
            best_auc = valauc
            print("bestauc")

            torch.save(net.state_dict(), save_path_best)

            csv_file = path+"/dia"+str(args.fusionmode)+"_val.csv"
            df_val.to_csv(csv_file, index=False)


            csv_file = path+"/dia"+str(args.fusionmode)+"_test.csv"
            df_test.to_csv(csv_file, index=False)

            csv_file = path+"/dia"+str(args.fusionmode)+"_cli.csv"
            df_clival.to_csv(csv_file, index=False)



    print('Finished Training')








if __name__ == '__main__':
    main()