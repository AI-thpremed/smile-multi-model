import pandas as pd
from sklearn.metrics import roc_auc_score

# 读取CSV文件
file_path = '/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_smile/dia_target_0/dia0_cli.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 确保数据中包含 'gt_all' 和 'predict_probs_all' 列
# if 'gt_all' not in data.columns or 'predict_probs_all' not in data.columns:
#     raise ValueError("CSV文件中必须包含 'gt_all' 和 'predict_probs_all' 列")


# Groundtruth	Probability
# 获取真实标签和预测概率
y_true = data['Groundtruth']
y_scores = data['Probability']

# 计算AUROC
auroc = roc_auc_score(y_true, y_scores)

print(f"AUROC: {auroc:.4f}")