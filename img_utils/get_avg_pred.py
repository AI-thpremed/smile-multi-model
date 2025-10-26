# import pandas as pd

# # 读取原始 CSV 文件
# df = pd.read_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/3_multi_CrossEntropyLoss.csv")

# # 按照 ID 列进行分组，并对 Predictions 列进行求和和平均操作
# grouped = df.groupby("ID").agg({"Labels": "first", "Predictions": ["sum", "mean"]})

# # 将多级列名转换为单级列名
# grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

# # 将结果保存为新的 CSV 文件
# grouped.to_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/3_multi_CrossEntropyLoss/new.csv", index=False)




import pandas as pd

# 读取原始 CSV 文件
df = pd.read_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/2_multi/2_multi.csv")

# 获取所有不重复的 ID
ids = df["ID"].unique()

# 初始化一个空列表用于存储新的数据
new_data = []

# 遍历每个 ID
for id in ids:
    # 找到所有对应的行
    rows = df.loc[df["ID"] == id]
    # 计算 Predictions 的平均值
    predictions_avg = rows["Predictions"].sum() / rows["Predictions"].count()
    # 获取 Labels 列的第一个值
    labels = rows["Labels"].iloc[0]
    # 创建新的行数据
    new_row = {"ID": id, "Labels": labels, "Predictions": predictions_avg}
    # 将新行数据添加到新数据列表中
    new_data.append(new_row)

# 将新数据保存为 CSV 文件
new_df = pd.DataFrame(new_data)
new_df.to_csv("/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/2_multi/new.csv", index=False)
