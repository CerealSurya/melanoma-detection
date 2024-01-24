import os
import random
import pandas as pd


df = pd.read_csv("HAM10000_metadata.csv")
df = df.reset_index()

#GOal: move 4412 images from hamdataset to ISIC
#!Might need to move all melanomic images as to upscale the malignant minority class??
num = 0
while(num < 4412): #ISIC_2907414.jpg
    rand = int(random.random() * 7) #random 7 digit number+
    status = ''
    try:
        for i in df.index:
            if df.loc[i].at['image_id'] == f'ISIC_{rand}':
                if df.loc[i].at['dx'] == 'mel':
                    status = 'malignant'
                else:
                    status = 'benign'
                    
        if status != '':
            os.rename(f"./combinedDataset/ISIC_2020_Test_Input/ISIC_{rand}.jpg", f"./combinedDataset/train/{status}/HAM_{rand}.jpg")
        else:
            print("Could not find img in HAM metadata")
            num -= 1
    except:
        print(f"error occured")
        num -= 1
    num+=1
