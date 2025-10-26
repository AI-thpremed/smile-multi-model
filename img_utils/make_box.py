import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv(r'/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/new.csv')
thispath='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/'
# 根据标签和预测值生成箱型图
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='Predictions', by='Labels', ax=ax)

# 设置图表标题和标签
ax.set_title('Predictions by Labels')
ax.set_xlabel('Labels')
ax.set_ylabel('Predictions')

# 保存图表为PNG格式的文件
plt.savefig(thispath+'predictions_by_labels.png')
