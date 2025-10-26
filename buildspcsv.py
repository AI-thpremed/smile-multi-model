import pandas as pd
import os

# Path to the CSV file
csv_file = "/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/5fold/Fold2_train.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Function to find image files
def find_image_files(id_value):
    directory = "/data/gaowh/data/files/szzyy/sp"
    image_list = []
    for filename in os.listdir(directory):
        if filename.startswith(str(id_value) + '_'):
            image_list.append(filename)
    return ';'.join(image_list)

df['new_imagelist'] = df['id'].apply(find_image_files)

df.to_csv("/data/gaowh/work/24process/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnert_szzyy/sp_label/train.csv", index=False, encoding='utf-8')
