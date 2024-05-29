import requests
import os
from urllib.parse import urlparse
import zipfile

model_zip_dir = '/data/riki_lf/riki_models/zips/'
model_models_dir = '/data/riki_lf/riki_models/models/'

def extract_filename_from_url(url):
    # 解析URL
    parsed_url = urlparse(url)
    # 提取路径部分
    path = parsed_url.path
    # 分割路径以获取文件名
    filename = path.split('/')[-1]
    return filename


def download_file(url, filename):
    # 发送GET请求
    r = requests.get(url)
    # 确保请求成功
    r.raise_for_status()

    # 以二进制写模式打开一个文件
    with open(model_zip_dir + filename, 'wb') as f:
        f.write(r.content)
    print(f"模型下载完毕，地址：{url}")


def download_and_unzip(modelUrl, modelFile):
    download_file(modelUrl, modelFile)
    with zipfile.ZipFile(model_zip_dir + modelFile, 'r') as zip_ref:
        zip_ref.extractall(model_models_dir + modelFile.split('.')[0] + '/')
    print(f"模型解压完毕")
    pass


def get_path(modelUrl):
    modelFile = extract_filename_from_url(modelUrl)
    path = model_models_dir + modelFile.split('.')[0]
    if os.path.isfile(model_zip_dir + modelFile) is not True:
        download_and_unzip(modelUrl, modelFile)
    return path