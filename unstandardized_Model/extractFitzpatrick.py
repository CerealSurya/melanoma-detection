import pandas as pd
import urllib.request 
from PIL import Image 

df = pd.read_csv("initialDataset/lesion_NotPresent/fitzpatrick17k.csv")
df = df.reset_index()

opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

counter1 = 0
counter2 = 0
# url = "https://www.dermaamin.com/site/images/clinical-pic/m/minocycline-pigmentation/minocycline-pigmentation1.jpg"
# urllib.request.urlretrieve(url, f"initialDataset/lesion_Present/seed{counter}.jpg") 

for index, row in df.iterrows():
    try:
        if "melanoma" in row["label"] and row["url"] != "":
            counter2 += 1
            urllib.request.urlretrieve(row["url"], f"initialDataset/lesion_Present/seed{counter2}.png") 
        elif row["url"] != "":
            counter1 += 1
            urllib.request.urlretrieve(row["url"], f"initialDataset/lesion_NotPresent/seed{counter1}.png")
    except:
        print("Error occured")

print("FInished")
