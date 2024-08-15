from callbacks import RikiLogCallback, RikiLogCallbackRedis
from .riki import gen_dict
from llamafactory.hparams import get_train_args
from llamafactory.train.tuner import run_sft
from llamafactory.train.callbacks import LogCallback
from typing import TYPE_CHECKING, Any, Dict, List, Optional


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks.append(LogCallback())
    callbacks.append(RikiLogCallbackRedis(data=finetuning_args, output_dir=training_args.output_dir))

    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)


def run_riki_exp(sio, data):
    args = gen_dict(data)
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback(),
                 RikiLogCallback(socket=sio, data=data, output_dir=args.get('output_dir'))]
    try:
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    except Exception as e:
        print(e)
        sio.emit("train_error", {'modelId': data['modelId'], 'errorMsg': str(e)})
        print("训练失败")

    print('训练完毕')
