import wget
import os

print('Downloading dataset...')

# 数据集的下载链接
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

# 如本地没有，则下载数据集 
if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')
