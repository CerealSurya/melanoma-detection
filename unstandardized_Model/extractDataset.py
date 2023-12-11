import zipfile
with zipfile.ZipFile('initialDataset.zip', 'r') as zip_ref:
    zip_ref.extractall('initialDataset')
