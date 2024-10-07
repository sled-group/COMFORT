import os
import gdown
import zipfile

url = 'https://drive.google.com/uc?id=1kmB-tN2FmhaypJ0Y2dq5MtQSdpxpGWKq'
output = os.path.join('data_generation', 'assets.zip')
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data_generation')