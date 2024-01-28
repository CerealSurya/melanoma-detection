import os
import random
import pandas as pd


df = pd.read_csv("HAM10000_metadata.csv")
df = df.reset_index()

"Upscaling minority malignant dataset as much as posisble to reduce bias"
for i in df.index: #1113 melanomic images in HAM10000 
    if df.loc[i].at['dx'] == 'mel':
        image = df.loc[i].at['image_id'] + '.jpg'
        try:
            num = 1
            if image in os.listdir("./combinedDataset/HAM10000_images_part_2"):
                num = 2
            os.rename(f"./combinedDataset/HAM10000_images_part_{num}/{image}", f"./combinedDataset/train/malignant/HAM_{image[5:]}")
        except:
            print(f"error occured on {image}")

"Moving 4372 images from HAM10000 to our validation to maintain 10% of data as validation"
num = 0
while(num < 4372): #ISIC_2907414.jpg
    rand = int(random.random() * 7) #random 7 digit number+
    status = ''
    try:
        for i in df.index:
            if df.loc[i].at['image_id'] == f'ISIC_{rand}':
                if df.loc[i].at['dx'] == 'mel':
                    status = 'malignant'
                else:
                    status = 'benign' #should all be benign!
                    
        if status != '':
            daNum = 1
            if image in os.listdir("./combinedDataset/HAM10000_images_part_2"):
                daNum = 2
            os.rename(f"./combinedDataset/HAM10000_images_part_{daNum}/ISIC_{rand}.jpg", f"./combinedDataset/test/{status}/HAM_{rand}.jpg")
        else:
            print("Could not find img in HAM metadata")
            num -= 1
    except:
        print(f"error occured")
        num -= 1
    num+=1
