import pandas as pd
from sklearn.model_selection import train_test_split


def shuffle_and_split(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 随机打乱数据
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # 按照8:2的比例分割成训练集和测试集
    train, test = train_test_split(df_shuffled, test_size=0.2, random_state=42)

    # 保存训练集和测试集到CSV文件
    train.to_csv('/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/train.csv', index=False)
    test.to_csv('/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/test.csv', index=False)

    print("Train and test sets have been created and saved.")


# 指定CSV文件路径
csv_file_path = "/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/updated_csv3_final_2.csv"
shuffle_and_split(csv_file_path)