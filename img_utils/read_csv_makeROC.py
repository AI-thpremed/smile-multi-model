import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
from sklearn.metrics import roc_curve, auc

# 读取 CSV 文件
data = pd.read_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/3_multi_CrossEntropyLoss_dot3.csv")

thispath="/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/"

# 设置阈值
threshold = 0.5

# 将预测结果转换为二进制值
predictions = np.array(data["Predictions"])
binary_predictions = np.where(predictions > threshold, 1, 0)

# 计算 ROC 曲线的 FPR 和 TPR
fpr, tpr, _ = roc_curve(data["Labels"], predictions)

# 计算 ROC 曲线下的面积（AUC）
auc_score = auc(fpr, tpr)

# 绘制 ROC 曲线
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

# 保存 ROC 曲线图像
plt.savefig(thispath+"new_roc_curve.png")

# 计算混淆矩阵
conf_matrix = ConfusionMatrix(actual_vector=data["Labels"].tolist(), predict_vector=binary_predictions.tolist())

# 将混淆矩阵转换为字符串
matrix_str = str(conf_matrix)

# 保存混淆矩阵为文本文件
with open(thispath+str(threshold)+"testnew_confusion_matrix.txt", "w") as f:
    f.write(matrix_str)
