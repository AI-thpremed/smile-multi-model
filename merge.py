import pandas as pd

# 定义文件路径
test_file = '/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/test3.csv'
train_file = '/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/train3.csv'
output_file = '/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/label/combined3.csv'

# 读取CSV文件
try:
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
except Exception as e:
    print(f"读取文件时出错：{e}")
    exit()

# 检查列头是否一致
if list(test_df.columns) != list(train_df.columns):
    print("警告：两个文件的列头不一致，无法合并。")
    print("test3.csv 的列头：", test_df.columns)
    print("train3.csv 的列头：", train_df.columns)
else:
    # 合并两个数据框
    combined_df = pd.concat([test_df, train_df], ignore_index=True)
    print("文件已成功合并。")

    # 保存到新的CSV文件
    combined_df.to_csv(output_file, index=False)
    print(f"合并后的数据已保存到：{output_file}")