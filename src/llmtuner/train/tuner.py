from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse, unquote
import os.path
import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras.callbacks import LogCallback, RikiLogCallback
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
from .dpo import run_dpo
from .orpo import run_orpo
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
import json
import requests

if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "orpo":
        run_orpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task.")


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


def run_riki_exp(sio, data):
    dataset_dir = '../riki_data'
    dataset = download(dataset_dir, data['datasetPath'])
    output_dir = '../saves_riki/lora/' + data['baseModel'] + '/' + data['modelPath']
    advance_config = json.loads(data['advanceConfig'])
    args = dict(
        stage='sft',
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
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback(), RikiLogCallback(socket=sio, data=data, output_dir=output_dir)]
    try:
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    except Exception as e:
        print(e)
        sio.emit("train_error", {'modelId': data['modelId'], 'errorMsg': str(e)})
        print("训练失败")

    print('riki')


def export_model(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer = load_tokenizer(model_args)
    get_template_and_fix_tokenizer(tokenizer, data_args.template)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is None:  # cannot convert dtype of a quantized model
        output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)
    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
