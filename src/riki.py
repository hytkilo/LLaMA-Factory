import requests
import os
from urllib.parse import urlparse, unquote
import zipfile
from typing import Any, Dict
import json
import configparser

# 创建一个配置解析器对象
config = configparser.ConfigParser()

config.read('riki_config.ini')


def riki_config(key):
    keys = key.split(".")
    return config.get(keys[0], keys[1])


model_zip_dir = riki_config('train.zip_dir')
model_models_dir = riki_config('train.models_dir')


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


def get_filename_from_url(url):
    # 解析URL
    parsed_url = urlparse(url)
    # 提取路径部分，并解码（考虑到URL编码的情况）
    path = unquote(parsed_url.path)
    # 从路径中获取文件名
    return path.split('/')[-1]


def download(dataset_dir, datasetPath):
    response = requests.get(datasetPath)
    # 检查是否请求成功
    if response.status_code == 200:
        # 打开一个文件，准备写入
        # 注意，'wb'表示以二进制写模式打开文件
        filename_from_url = get_filename_from_url(datasetPath)
        with open(os.path.join(dataset_dir, filename_from_url), 'wb') as f:
            # 将请求得到的内容写入文件
            f.write(response.content)
        print('文件下载成功。')
        return filename_from_url
    else:
        print('文件下载失败。')
        return ''


def gen_dict(data) -> Dict[str, Any]:
    dataset_dir = riki_config('train.dataset_dir')
    dataset = download(dataset_dir, data['datasetPath'])
    output_dir = riki_config('train.lora_dir') + data['baseModel'] + '/' + data['modelPath']
    adapters = []
    if data.get("parentModelPath") is not None:
        for url in data.get("parentModelPath"):
            adapters.append(get_path(url))
    advance_config = json.loads(data['advanceConfig'])
    args = dict(
        stage='sft',
        model_id=data['modelId'],
        do_train=True,
        model_name_or_path=data['baseModel'],
        dataset=dataset,
        dataset_dir=dataset_dir,
        template='qwen',
        finetuning_type='lora',
        lora_target=advance_config.get('loraTarget'),
        output_dir=output_dir,
        overwrite_cache=True,
        overwrite_output_dir=True,
        preprocessing_num_workers=16,
        per_device_train_batch_size=advance_config['batchSize'],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=advance_config['gradientAccumulationSteps'],
        lr_scheduler_type='cosine',
        logging_steps=10,
        save_steps=500000,
        learning_rate=float(advance_config['learningRate']),
        num_train_epochs=data['epoch'],
        max_samples=999999,
        val_size=0.1,
        plot_loss=False,
        lora_rank=advance_config['loraRank'],
        lora_alpha=advance_config['loraAlpha'],
        lora_dropout=advance_config['loraDropout'],
        use_rslora=advance_config['useRslora'],
        use_dora=advance_config['useDora'],
        neftune_noise_alpha=advance_config['neftuneNoiseAlpha'],
        max_grad_norm=advance_config['maxGradNorm'],
        warmup_steps=advance_config['warmupSteps'],
        upcast_layernorm=advance_config['upcastLayernorm'],
        use_llama_pro=advance_config['useLlamaPro'],
        flash_attn='auto',
        use_unsloth=advance_config.get('useUnsloth', True),
        # num_layer_trainable=advance_config['numLayerTrainable'],
        fp16=True
    )
    if 'qwen' in data['baseModel']:
        args.setdefault("template", 'qwen')
    elif 'glm-4' in data['baseModel']:
        args.setdefault("template", 'glm4')
    if advance_config.get('cutoff_len') is not None:
        args.setdefault("cutoff_len", int(advance_config.get('cutoff_len')))
    else:
        args.setdefault("cutoff_len", 16384)
    if len(adapters) > 0:
        args.setdefault("adapter_name_or_path", ",".join(adapters))
    if advance_config.get('quantizationBit') is not None and advance_config.get('quantizationBit') != '':
        args.setdefault("quantization_bit", int(advance_config.get('quantizationBit')))
    if "32B" in data['baseModel']:
        args.setdefault("quantization_bit", 4)
    if advance_config.get('additional_target') is not None:
        args.setdefault("additional_target", advance_config.get('additional_target'))
    return args
