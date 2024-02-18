from PIL import Image

from os import listdir
path = "combinedDataset/train/benign/"
benign = [f for f in listdir(path)]


for img in benign:
    image_file = Image.open(f"{path}{img}") 

    image_file.save(f"newBenign/{img}", quality = 30)

path = "combinedDataset/train/benign/HAM10000_images_part_1/"
benign = [f for f in listdir(path)]


for img in benign:
    image_file = Image.open(f"{path}{img}") 

    image_file.save(f"newBenign/HAM{img}", quality = 30)

path = "combinedDataset/train/benign/HAM10000_images_part_2/"
benign = [f for f in listdir(path)]


for img in benign:
    image_file = Image.open(f"{path}{img}") 

    image_file.save(f"newBenign/HAM{img}", quality = 30)

path = "combinedDataset/train/malignant/"
malignant = [f for f in listdir(path)]


for img in malignant:
    image_file = Image.open(f"{path}{img}") 

    image_file.save(f"newMal/{img}", quality = 30)