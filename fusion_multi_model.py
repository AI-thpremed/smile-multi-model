import torch
import torch.nn as nn
# from fusion.segating import SEGating
# from fusion.average import project,SegmentConsensus
# from fusion.segating import SEGating
# from fusion.segating import SEGating
# from fusion.nextvlad import NextVLAD


# from models.cv_models.swin_transformer import swin
# from models.cv_models.swin_transformer_v2 import swinv2
# # from models.cv_models.resnest import resnest50, resnest101
# from models.cv_models.convnext import convnext_tiny, convnext_small, convnext_base
# from resnet import resnet50,resnet152

from ConvNeXt import convnext_tiny as create_model

from resnet_att import resnet50bam, resnet50cbam,resnet50se,resnet50,resnet152,resnet34



# from models.FIT_Net import FITNet

# from model3d import resnet

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import random
from typing import List, Tuple, Optional, Union, Literal
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from typing import Optional, Dict, Any
from torch import Tensor
import numpy as np
from dataloader.image_transforms import Image_Transforms
from torchvision import transforms, datasets
import os
from PIL import Image
from torch.nn import BatchNorm1d

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
 
# img+img share backbone
class Build_MultiModel_szzyy(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()



        self.num_classes = num_classes


        self.backbone=backbone


        self.model = resnet50()
        self.model.fc = Identity()


        self.lg2 = torch.nn.Linear(in_features=(2048+8), out_features=self.num_classes)


        # self.vector=self.lg2.weight



        print('init model:', backbone)



    def forward(self, img,extradata):

        all=[]

        ydoutput = self.model(img)
        
        all.append(ydoutput)
        all.append(extradata)


        all_output = torch.cat(all, -1) # b, c1+c2
                # 在预测过程中保存向量值
   
        vector = all_output

        self.vector = vector.detach()

        res = self.lg2(all_output)


        return res,vector





# MLP示例
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        output = self.output_layer(x)
        return output

# 注意力机制示例
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_scores = self.sigmoid(self.linear(x))
        weighted_features = torch.mul(x, attention_scores)
        fused_feature = torch.sum(weighted_features, dim=1, keepdim=True)
        return attention_scores, fused_feature



class Build_MultiModel_szzyy_pair_mlpatt(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()



        self.num_classes = num_classes



        self.backbone=backbone


        self.model = resnet50()
        self.model.fc = Identity()


        self.lg1 = torch.nn.Linear(in_features=(2048+8), out_features=self.num_classes)

        self.lg2 = torch.nn.Linear(in_features=(2048+2048), out_features=2048)


            
        self.mlp = MLP(input_dim=(2048+8), hidden_dim=512, output_dim=2)



        self.attention_model = Attention(input_dim=2056)

        self.binary_classifier = nn.Linear(1, 2)


        print('init model:', backbone)




    def forward(self, img, extradata):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)

        avg_feats = torch.stack(output_list, dim=1)
        summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
        avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
        all.append(avg_feats_mean)
        all.append(extradata)
        all_output = torch.cat(all, -1) # b, c1+c2

        if self.backbone=="mlp":

            res = self.mlp(all_output)

        else:
            attention_scores, fused_feature = self.attention_model(all_output)
            res = self.binary_classifier(fused_feature.view(-1, 1))





        return res


 
class Build_MultiModel_szzyy_pair(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()



        self.num_classes = num_classes



        self.backbone=backbone


        self.model = resnet50()
        self.model.fc = Identity()


        self.lg1 = torch.nn.Linear(in_features=(2048+8), out_features=self.num_classes)

        self.lg2 = torch.nn.Linear(in_features=(2048+2048), out_features=2048)





        print('init model:', backbone)




    def forward(self, img, extradata):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)


        if self.backbone=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            all.append(avg_feats_mean)
            all.append(extradata)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg1(all_output)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            temp = self.lg2(all_output)
            all.append(temp)
            all.append(extradata)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg1(all_output)



        return res



class Build_MultiModel_szzyy_pair_multi(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()



        self.num_classes = num_classes



        self.backbone=backbone


        self.model = resnet50()
        self.model.fc = Identity()


        self.lg1 = torch.nn.Linear(in_features=(2048), out_features=self.num_classes)

        self.lg2 = torch.nn.Linear(in_features=(2048+2048), out_features=2048)





        print('init model:', backbone)




    def forward(self, img, extradata):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)


        if self.backbone=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征

            modified_data = extradata / 10 + 1

            all_output = avg_feats_mean * modified_data



            res = self.lg1(all_output)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            temp = self.lg2(all_output)
            all.append(temp)
            all.append(extradata)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg1(all_output)



        return res


class Build_MultiModel_szzyy_pair_onlyimg(nn.Module):
    def __init__(self, backbone='ResNet50',fusion_type='avg', input_dim=2048, num_classes=3, pretrained_modelpath='None'):
        super().__init__()



        self.num_classes = num_classes
        self.backbone=backbone
        self.fusion_type=fusion_type


        if backbone=='ResNet50':
            self.model = resnet50()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        elif backbone=='ResNet34':
            self.model=resnet34()

            

        elif backbone=='ResNet50BAM':
            self.model=resnet50bam()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet50CBAM':
            self.model=resnet50cbam()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        elif backbone=='ResNet50SE':
            self.model=resnet50se()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        else:


            self.model = resnet152()

        self.model.fc = Identity()



        if self.backbone=='ResNet34':
            self.lg1 = torch.nn.Linear(in_features=(512+512), out_features=self.num_classes)
            self.lg2 = torch.nn.Linear(in_features=(512), out_features=self.num_classes)

        else:
            self.lg1 = torch.nn.Linear(in_features=(2048+2048), out_features=self.num_classes)
            self.lg2 = torch.nn.Linear(in_features=(2048), out_features=self.num_classes)




        print('init model:', backbone)



    def forward(self, img):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)

        if self.fusion_type=="avg" or self.fusion_type=="3img2" :
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            res = self.lg2(avg_feats_mean)

        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(all_output)


        return res

def convert_to_float(arr):
    float_arr = np.empty(arr.shape, dtype=float)
    for i in range(len(arr)):
        try:
            float_arr[i] = float(arr[i])
        except ValueError:
            float_arr[i] = 0  
    return float_arr

class TabularDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame, target: Optional[str]=None, 
                 target_dtype: Union[Literal['regression', 'classification'], torch.dtype]=torch.long,
                 categorical_features: Optional[List[str]]=None,
                 continuous_features: Optional[List[str]]=None,im_transforms=None):
        """
        PyTorch Dataset for tabular data.

        Parameters:
        - dataframe: pd.DataFrame. Input data.
        - target: str. Target column name.
        - target_dtype: Union[Literal['regression', 'classification'], torch.dtype]. Target data type.
        - categorical_features: Optional[List[str]]. List of categorical feature column names.
        - continuous_features: Optional[List[str]]. List of continuous feature column names.
        """
        if categorical_features is None and continuous_features is None:
            raise ValueError('At least one of categorical_features and continuous_features must be provided')
        
        if target_dtype == 'classification':
            self.target_dtype = torch.long
        elif target_dtype == 'regression':
            self.target_dtype = torch.float
        elif isinstance(target_dtype, torch.dtype):
            self.target_dtype = target_dtype
        else:
            raise ValueError('target_dtype must be either "categorical" or "continuous"')
        
        self.dataset_length = len(dataframe)
        self.vocabulary = {}
        for column in categorical_features:
            if column not in dataframe.columns:
                raise ValueError(f'{column} not found in dataframe')
            unique_values = sorted(dataframe[column].unique().tolist())
            self.vocabulary[column] = {value: i for i, value in enumerate(unique_values)}
        
        for column in continuous_features:
            if column not in dataframe.columns:
                raise ValueError(f'{column} not found in dataframe')
        
        if categorical_features is not None:
            self.categorical_data = dataframe[categorical_features]
        else:
            self.categorical_data = None
        if continuous_features is not None:
            self.continuous_data = dataframe[continuous_features].to_numpy()
        else:
            self.continuous_data = None
        if target is not None:
            self.target = torch.tensor(dataframe[target].to_numpy(), dtype=self.target_dtype)
        else:
            self.target = torch.randn(self.dataset_length)
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])
        self.im_names = dataframe['image_names'].to_numpy()
        self.im_dir='/data/gaowh/data/files/szzyy_rzza'


    def get_vocabulary(self):
        return self.vocabulary
    
    def check_image_existence(self,image_path):
        return os.path.exists(image_path)
    def get_random_images(self, image_names, num_images):
        random_images = []
        for image_name in image_names:
            image_path = os.path.join(self.im_dir, image_name)
            if self.check_image_existence(image_path):
                random_images.append(image_path)
            if len(random_images) >= num_images:
                break

        while len(random_images) < num_images:
            random_images.append(random.choice(random_images))

        return random_images


    def __len__(self):
        return self.dataset_length
        
    def __getitem__(self, idx) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if self.categorical_data is not None:
            categorical_data = [self.vocabulary[col][self.categorical_data[col].iloc[idx]] for col in self.categorical_data.columns]
            categorical_data = torch.tensor(categorical_data, dtype=torch.long)
        else:
            categorical_data = torch.empty(0, dtype=torch.long)

        if self.continuous_data is not None:
            continuous_data = torch.tensor(self.continuous_data[idx], dtype=torch.float32)
        else:
            continuous_data = torch.empty(0, dtype=torch.float32)

        xxx = self.im_names[idx].split(';')
        img_list = self.get_random_images(xxx, 2)
        images = []

        default_image = Image.new('RGB', (224, 224))  # 创建一个默认图像
        for image_path in img_list:
            try:
                im = Image.open(image_path).convert('RGB')
                im = self.im_transforms(im)
            except:
                print('Error: Failed to open or verify image file {}'.format(image_path))
                im = self.im_transforms(default_image)  # 使用默认图像
            images.append(im)

        return images, self.target[idx], categorical_data, continuous_data

        

class ColumnEmbedding(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]], embedding_dim: int):
        """
        Column embedding layer for categorical features.

        Parameters:
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features 
                      (e.g. {
                                'column_name_1': 
                                {
                                    'category_1_1': index_1, 
                                    'category_1_2': index_2,
                                }, 
                                'column_name_2': 
                                {
                                    'category_2_1': index_1, 
                                    'category_2_2': index_2,
                                    'category_2_3': index_3,
                                }
                            }).
        - embedding_dim (int): Embedding dimension.
        """
        super(ColumnEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict({
            column: nn.Embedding(len(vocab), embedding_dim)
            for column, vocab in vocabulary.items()
        })
        
    def forward(self, x: torch.Tensor, column: str) -> torch.Tensor:
        return self.embeddings[column](x)
    


class CatEncoder(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, int]], embedding_dim: int):
        super(CatEncoder, self).__init__()
        """
        Categorical feature encoder.

        Parameters:
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features
        - embedding_dim (int): Embedding dimension.
        """
        self.vocabulary = vocabulary
        self.column_embedding = ColumnEmbedding(vocabulary, embedding_dim)
    
    def forward(self, x):
        x = [self.column_embedding(x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        x = x.squeeze(-1)  # 移除最后一个维度

        return x

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.batch_norm = BatchNorm1d(num_numerical_types)

    def forward(self, x):
        x = self.batch_norm(x)  
        return x

class NumEncoder(nn.Module):
    def __init__(self, num_features: int, embedding_dim:int):
        """
        Continuous feature encoder.

        Parameters:
        - num_features (int): Number of continuous features.
        - embedding_dim (int): Embedding dimension.
        """
        super(NumEncoder, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(num_features)])
        self.numerical_embedder = NumericalEmbedder(1, 2)
        
    def forward(self, x):
        # x = self.numerical_embedder(x)

        # x = [linear(x[:, i].unsqueeze(1)) for i, linear in enumerate(self.linears)]
        # x = torch.stack(x, dim=1)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout_rate: float):
        """
        Transformer encoder.

        Parameters:
        - d_model (int): Dimension of the model.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - dropout_rate (float): Dropout rate.
        """
        super(Transformer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=dim_feedforward, 
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ), 
            num_layers=num_layers,
            norm=nn.LayerNorm([d_model])
        )

    def forward(self, x):
        return self.transformer(x)


class Build_MultiModel_szzyy_pair_alldata(nn.Module):
    def __init__(self,vocabulary: Dict[str, Dict[str, int]], backbone='ResNet50',fusion_type='avg', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()

        self.catencoders =  CatEncoder(vocabulary, 1)
        self.numencoders=NumEncoder(2, 1)

        self.num_classes = num_classes
        self.backbone=backbone
        self.fusion_type=fusion_type


        if backbone=='ResNet50':
            self.model = resnet50()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        elif backbone=='ResNet34':
            self.model=resnet34()

            

        elif backbone=='ResNet50BAM':
            self.model=resnet50bam()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet50CBAM':
            self.model=resnet50cbam()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        elif backbone=='ResNet50SE':
            self.model=resnet50se()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        else:


            self.model = resnet152()

        self.model.fc = Identity()



        if self.backbone=='ResNet34':
            self.lg1 = torch.nn.Linear(in_features=(512+512), out_features=self.num_classes)
            self.lg2 = torch.nn.Linear(in_features=(512), out_features=self.num_classes)

        else:
            self.lg1 = torch.nn.Linear(in_features=(2048+2048), out_features=self.num_classes)
            self.lg2 = torch.nn.Linear(in_features=(2048+11), out_features=self.num_classes)



        print('init model:', backbone)



    def forward(self, img,imcat,imnum):
        batch_size = imcat.size(0)
        

        max_value = torch.max(imcat)
        min_value = torch.min(imcat)
        imcat_normalized = (imcat - min_value) / (max_value - min_value)

        tabular_features = torch.cat((imcat_normalized,imnum), dim=1)
        # tabular_features2 = self.lg4(tabular_features)

        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)

        if self.fusion_type=="avg" or self.fusion_type=="3img2" :
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            # res = self.lg2(avg_feats_mean)
            combined_tensor = torch.cat((tabular_features, avg_feats_mean), dim=1)

            res = self.lg2(combined_tensor)

        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(all_output)



        return res


class mlpfusion(nn.Module):
    def __init__(self, feature_size = 256): 
        super(mlpfusion, self).__init__()
        self.fc1 = Linear(feature_size*2, 1) 
        self.fc2 = Linear(feature_size*2, 1)

        self.sigmoid= nn.Sigmoid()

    def forward(self, encoder_output_list):
        # pdb.set_trace()
        batch_size = encoder_output_list[0].size()[0]
        xall = torch.cat(encoder_output_list, -1) # b, c1+c2

     
        weight1 = self.fc1(xall)
        weight2 = self.fc2(xall)


        weight1 = self.sigmoid(weight1)
        weight2 = self.sigmoid(weight2)

        return weight1, weight2




class SegmentConsensus(nn.Module):
    def __init__(self, in_features=2048, out_features=256):
        print("cv fusion: average...", flush=True)
        super(SegmentConsensus, self).__init__()
        self.linear_logits = torch.nn.Linear(
            in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear_logits(x)
        return x



        
# img+img share backbone
class Build_MultiModel_ShareBackbone_mlp(nn.Module):
    def __init__(self,backbone='ResNet34', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*2
        # self.num_classes = num_classes
        self.use_gate = use_gate
        # self.gate = SegmentConsensus(self.input_dim,2048*5)
        # self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)
        # self.lg2 = nn.Sequential(nn.Linear(256, 2),nn.SoftMax())
        self.lg2 = nn.Sequential(nn.Linear(256, 2), nn.LogSoftmax(dim=1))

        self.backbone=backbone

        if backbone=='ResNet50':
            self.model = resnet50()
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        elif backbone=='ResNet34':
            self.model=resnet34()

            

        elif backbone=='ResNet50BAM':
            self.model=resnet50bam()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)

        elif backbone=='ResNet50CBAM':
            self.model=resnet50cbam()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)


        elif backbone=='ResNet50SE':
            self.model=resnet50se()

            
            net_dict = self.model.state_dict()
            predict_model = torch.load("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_standard/resnet50-pre.pth")
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
            net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
            self.model.load_state_dict(net_dict)
        else:
            self.model = resnet152()



        self.model.fc = Identity()
        # self.lgbase = torch.nn.Linear(in_features=2048, out_features=256)

        if self.backbone=='ResNet34':
            self.lgbase = torch.nn.Linear(in_features=512, out_features=256)

        else:
            self.lgbase = torch.nn.Linear(in_features=2048, out_features=256)


        self.mlpfusion=mlpfusion()

        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')


    def forward(self, x):

        encoder_output_list = []

        for i in range(len(x)):
            image = x[i]
            temp = self.model(image)
            temp=self.lgbase(temp)
            encoder_output_list.append(temp)


        weight1, weight2=self.mlpfusion(encoder_output_list)


        encoder_output_list[0] = encoder_output_list[0] * weight1
        encoder_output_list[1] = encoder_output_list[1] * weight2



        avg_feats = torch.stack(encoder_output_list, dim=1)
        summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
        output = summed_feats / image.size(1)  # 计算平均特征



        output = self.lg2(output)

        return output







class Build_MultiModel_szzyy_pair_onlyimg_convnext(nn.Module):
    def __init__(self, backbone='convnext', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone=backbone
        self.model = create_model(num_classes=num_classes)
        self.model.head = Identity()
        self.lg1 = torch.nn.Linear(in_features=(768+768), out_features=self.num_classes)
        self.lg2 = torch.nn.Linear(in_features=(768), out_features=self.num_classes)
        print('init model:', backbone)
    def forward(self, img):
        all = []
        output_list = []
        # 循环处理每个图像
        for i in range(len(img)):
            image = img[i]
            temp = self.model(image)
            output_list.append(temp)
        if self.backbone=="avg":
            avg_feats = torch.stack(output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            avg_feats_mean = summed_feats / len(output_list)  # 计算平均特征
            res = self.lg2(avg_feats_mean)
        else:
            all_output = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(all_output)


        return res








class Build_MultiModel_smile(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, fusionmode=0,num_classes=2, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone
        self.fusionmode = fusionmode
        self.model_input = resnet50()
        self.model_input.fc = Identity()
        self.model_dw = resnet50()
        self.model_dw.fc = Identity()
        self.model_xy = resnet50()
        self.model_xy.fc = Identity()



        self.lg1 = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        self.lg2_1 = torch.nn.Linear(in_features=2048*2, out_features=256)

        self.lg2_2 = torch.nn.Linear(in_features=256, out_features=self.num_classes)

        self.lg3_1 = torch.nn.Linear(in_features=2048*3, out_features=256)

    def forward(self, input1_img,dw_img,xy_img):
        output_list = []

        #单纯吸引图
        if self.fusionmode==0:
            temp = self.model_xy(xy_img)
            output_list.append(temp)
            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(res)

        #吸引+对位
        elif self.fusionmode==1:
            temp = self.model_xy(xy_img)
            output_list.append(temp)
            temp = self.model_dw(dw_img)
            output_list.append(temp)


            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg2_1(res)
            res = self.lg2_2(res)


        elif self.fusionmode==2:
            temp = self.model_xy(xy_img)
            output_list.append(temp)
            temp = self.model_dw(dw_img)
            output_list.append(temp)
            temp = self.model_input(input1_img)
            output_list.append(temp)

            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg3_1(res)
            res = self.lg2_2(res)




        elif self.fusionmode==3:
            features = self.model_input(input1_img)
            output_list.append(features)
            res = torch.cat(output_list, -1)
            res = self.lg1(res)

        return res






class Build_MultiModel_smile_embedding(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, fusionmode=0,num_classes=2, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*20
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.backbone=backbone


        self.fusionmode = fusionmode


        self.model_input = resnet50()
        self.model_input.fc = Identity()
        self.model_dw = resnet50()
        self.model_dw.fc = Identity()
        self.model_xy = resnet50()
        self.model_xy.fc = Identity()
        self.lg1 = torch.nn.Linear(in_features=2048, out_features=self.num_classes)

        self.lg2_1 = torch.nn.Linear(in_features=2048*2, out_features=256)

        self.lg2_2 = torch.nn.Linear(in_features=896, out_features=self.num_classes).double()

        self.lg3_1 = torch.nn.Linear(in_features=2048*3, out_features=256)

        self.lg4 = torch.nn.Linear(in_features=2048, out_features=256)
        self.numembedding=NumEncoderTransformer(10,64)

    def forward(self, input1_img,dw_img,xy_img,im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9, im_10, im_11, im_12, im_13, im_14, im_15):
        output_list = []


        im_1 = im_1.unsqueeze(1)
        im_2 = im_2.unsqueeze(1)
        im_3 = im_3.unsqueeze(1)
        im_4 = im_4.unsqueeze(1)
        im_5 = im_5.unsqueeze(1)
        im_6 = im_6.unsqueeze(1)
        im_7 = im_7.unsqueeze(1)
        im_8 = im_8.unsqueeze(1)
        im_9 = im_9.unsqueeze(1)
        im_10 = im_10.unsqueeze(1)
        im_11 = im_11.unsqueeze(1)
        im_12 = im_12.unsqueeze(1)
        im_13 = im_13.unsqueeze(1)
        im_14 = im_14.unsqueeze(1)
        im_15 = im_15.unsqueeze(1)

        continuous_features = torch.cat((im_1, im_2, im_3, im_5, im_7, im_8, im_9, im_10, im_12, im_13), dim=1)

        con_fea=self.numembedding(continuous_features)
        #单纯吸引图
        if self.fusionmode==0:
            temp = self.model_xy(xy_img)
            output_list.append(temp)
            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg1(res)

        #吸引+对位
        elif self.fusionmode==1:
            temp = self.model_xy(xy_img)
            output_list.append(temp)
            temp = self.model_dw(dw_img)
            output_list.append(temp)


            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg2_1(res)
            res = self.lg2_2(res)

        elif self.fusionmode==2:
            temp = self.model_xy(xy_img)
            output_list.append(temp)
            temp = self.model_dw(dw_img)
            output_list.append(temp)
            temp = self.model_input(input1_img)
            output_list.append(temp)

            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg3_1(res)

            continuous_features = torch.cat((res,con_fea), dim=1)

            res = self.lg2_2(continuous_features)



        elif self.fusionmode==3:
            temp = self.model_input(input1_img)
            output_list.append(temp)
            res = torch.cat(output_list, -1) # b, c1+c2
            res = self.lg4(res)



            continuous_features = torch.cat((res,con_fea), dim=1)

            res = self.lg2_2(continuous_features)



        return res




class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super(NumericalEmbedder, self).__init__()
        # 确保 BatchNorm1d 使用 Double 类型
        self.batch_norm = nn.BatchNorm1d(num_numerical_types).double()

    def forward(self, x):
        # 确保输入数据转换为 Double 类型
        x = x.double()
        x = self.batch_norm(x)
        return x



class NumEncoderTransformer(nn.Module):
    def __init__(self, num_continuous_features: int, embedding_dim: int):
        super(NumEncoderTransformer, self).__init__()

        self.num_continuous_features = num_continuous_features
        self.mlpclassifier1=nn.Linear(num_continuous_features,max(2*num_continuous_features+1,embedding_dim)).double()
        self.mlpclassifier2=nn.Linear(max(2*num_continuous_features+1,embedding_dim),embedding_dim * num_continuous_features).double()
        self.numerical_embedder = NumericalEmbedder(1, num_continuous_features)



    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        x = self.numerical_embedder(x)
        x = x.squeeze(-1) 
        x = self.mlpclassifier1(x)
        x = self.mlpclassifier2(x)
        return x


