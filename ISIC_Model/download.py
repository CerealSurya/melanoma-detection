import pandas as pd
import urllib.request 
from PIL import Image 

#https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip


opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

counter1 = 0
counter2 = 0
# url = "https://www.dermaamin.com/site/images/clinical-pic/m/minocycline-pigmentation/minocycline-pigmentation1.jpg"
# urllib.request.urlretrieve(url, f"initialDataset/lesion_Present/seed{counter}.jpg") 

urllib.request.urlretrieve("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip", "newTest.zip") 

# try:
#     urllib.request.urlretrieve("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip", f"combinedDataset/train.zip") 
# except:
#     print("Error occured")

print("FInished")
