import os

#GOAL: Move 1000 imgs from training to create validation data
num = 0
zeros = "000"
for i in range(10):
    print(num)
    if num >= 10 and num < 100:
        zeros = "00"
    elif num >= 100:
        zeros = "0"
    try:
        os.rename(f"./initialDataset/train/lesion_Present/mn500/mn500/mn500/seed{zeros}{num}.png", f"./initialDataset/validation/lesion_Present/seed{zeros}{num}.png")
    except:
        print(f"error occured on img seed{zeros}{num}")
    num+=1

num = 0
counter = 0
while num < 500:
    try:
        os.rename(f"./initialDataset/train/lesion_NotPresent/seed{counter}.png", f"./initialDataset/validation/lesion_NotPresent/seed{counter}.png")
    except:
        num-=1
        print(f"error occured on img seed{num}")
    num+=1
    counter+=1