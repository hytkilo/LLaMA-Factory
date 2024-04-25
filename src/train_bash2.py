import socketio
from typing import Any, Dict
import json
import os.path
import requests
from urllib.parse import urlparse, unquote
import redis
import subprocess

redis_host = '10.12.0.16'
redis_port = 6379
redis_password = '2sSwwJ7@f8UT'
r = redis.Redis(host=redis_host, port=redis_port, db=5, password=redis_password, decode_responses=True)
sub = r.pubsub()


def main():
    sio = socketio.Client()
    sio.connect('https://admin.yunhelp.com/socket.io/?socketType=train_client&clientId=111', transports=['websocket'],
                retry=True, wait_timeout=2)

    def message_handler(message):
        print(f"Received: {message['data'].decode()}")

    sub.subscribe(**{'train_model': message_handler})

    @sio.event
    def do_train(data):
        print(data)
        args = gen_dict(data)
        cmd_text = gen_cmd(args)
        subprocess.run(cmd_text)

    sio.wait()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


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
    dataset_dir = '../riki_data'
    dataset = download(dataset_dir, data['datasetPath'])
    output_dir = '../saves_riki/lora/' + data['baseModel'] + '/' + data['modelPath']
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
        cutoff_len=16384,
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
        num_layer_trainable=advance_config['numLayerTrainable'],
        fp16=True
    )
    if advance_config.get('additional_target') is not None:
        args.setdefault("additional_target", advance_config.get('additional_target'))
    return args


def gen_cmd(args: Dict[str, Any]) -> str:
    cmd_lines = ["CUDA_VISIBLE_DEVICES=4,5,6 accelerate launch --config_file examples/accelerate/single_config.yaml src/train_bash.py"]
    for k, v in args.items():
        if v is not None and v is not False and v != "":
            cmd_lines.append("    --{} {} ".format(k, str(v)))
    # cmd_text = "\\\n".join(cmd_lines)
    # cmd_text = "```bash\n{}\n```".format(cmd_text)
    return cmd_lines


if __name__ == "__main__":
    main()
