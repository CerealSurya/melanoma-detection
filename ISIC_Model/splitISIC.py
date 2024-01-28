import os
import random
import pandas as pd

df = pd.read_csv("ISIC_2020_Training_GroundTruth.csv")
df = df.reset_index()

"Splitting training data into benign / malignant"
for i in df.index: 
    image = df.loc[i].at['image_name'] + '.jpg'
    try:
        status = 'benign'
        if df.loc[i].at['benign_malignant'] == 'malignant':
            status = 'malignant'
        os.rename(f"./combinedDataset/train/{image}", f"./combinedDataset/train/{status}/original_{image}")
    except:
        print(f"error occured on {image}")