import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/stroke_label/szzyy_stroke_db.csv")

# Split the data into train and test sets with an 80:20 ratio
train_data, test_data = train_test_split(data, test_size=0.2)

# Save the train and test sets back to CSV files
train_data.to_csv("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/stroke_label/train.csv", index=False)
test_data.to_csv("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/stroke_label/test.csv", index=False)
