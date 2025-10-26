import pandas as pd

# 读取原始 CSV 文件
df = pd.read_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/3_multi_CrossEntropyLoss.csv")

# 获取所有不重复的 ID
ids = df["ID"].unique()

# 初始化一个空列表用于存储新的数据
new_data = []

# 遍历每个 ID
for id in ids:
    # 找到所有对应的行
    rows = df.loc[df["ID"] == id]
    # 如果Predictions的平均值小于0.3，将Predictions设为0，否则计算平均值
    predictions_avg = 0 if rows["Predictions"].mean() < 0.3 else rows["Predictions"].mean()
    # 获取Labels列的第一个值
    labels = rows["Labels"].iloc[0]
    # 创建新的行数据
    new_row = {"ID": id, "Labels": labels, "Predictions": predictions_avg}
    # 将新行数据添加到新数据列表中
    new_data.append(new_row)

# 将新数据保存为 CSV 文件
new_df = pd.DataFrame(new_data)
new_df.to_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/3_multi_CrossEntropyLoss_dot3.csv", index=False)
