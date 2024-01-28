import os
import random
import pandas as pd


df = pd.read_csv("HAM10000_metadata.csv")
df = df.reset_index()
"Upscaling minority malignant dataset as much as posisble to reduce bias"
# for i in df.index: #1113 melanomic images in HAM10000 
#     if df.loc[i].at['dx'] == 'mel':
#             image = df.loc[i].at['image_id'] + '.jpg'
#             try:
#                 num = 1
#                 if image in os.listdir("./combinedDataset/HAM10000_images_part_2"):
#                     num = 2
#                 os.rename(f"./combinedDataset/HAM10000_images_part_{num}/{image}", f"./combinedDataset/train/malignant/HAM_{image[5:]}")
#             except:
#                 print(f"error occured on {image}")

"Moving 4372 images from HAM10000 to our validation to maintain 10% of data as validation"
num = 0
# while(num < 4372): 
#     #less than 34320, greater than 24306
#     rand = int((random.random() * (34320 - 24306)) + 24306)
#     rand = "00" + str(rand) #random 7 digit number
#     #try:
#     if f'ISIC_{rand}.jpg' in os.listdir("./combinedDataset/HAM10000_images_part_1"):
#         os.rename(f"./combinedDataset/HAM10000_images_part_1/ISIC_{rand}.jpg", f"./combinedDataset/test/benign/HAM_{rand}.jpg")        
#     elif f'ISIC_{rand}.jpg' in os.listdir("./combinedDataset/HAM10000_images_part_2"):
#         os.rename(f"./combinedDataset/HAM10000_images_part_2/ISIC_{rand}.jpg", f"./combinedDataset/test/benign/HAM_{rand}.jpg")
#     else:
#         print("Could not find img in HAM metadata") #probably a malignant image that was already used
#         num -= 1
#     # except:
#     #     print(f"error occured")
#     #     num -= 1
#     num+=1

while(num < 2729): 
    #less than 34320, greater than 24306
    rand = int((random.random() * (34320 - 24306)) + 24306)
    rand = "00" + str(rand) #random 7 digit number
    #try:
    os.rename(f"./combinedDataset/test/benign/HAM_{rand}.jpg" ,f"./combinedDataset/HAM10000_images_part_1/HAM_{rand}.jpg")        
    # except:
    #     print(f"error occured")
    #     num -= 1
    num+=1