import logging
import signal
import sys
import shutil
import json
import os
import time
import requests
from riki import riki_config
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional

import transformers
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length

from .constants import TRAINER_LOG
from .logging import LoggerHandler, get_logger
from .misc import fix_valuehead_checkpoint

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = get_logger(__name__)


def zip_dir(folder_path, output_path):
    # 压缩文件夹，这将创建一个.zip文件
    archive_path = shutil.make_archive(base_name=output_path, format='zip', root_dir=folder_path)

    return archive_path


class FixValueHeadModelCallback(TrainerCallback):
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        if args.should_save:
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"),
                output_dir=os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step)),
                safe_serialization=args.save_safetensors,
            )


class LogCallback(TrainerCallback):
    def __init__(self, output_dir: str) -> None:
        r"""
        Initializes a callback for logging training and evaluation status.
        """
        """ Progress """
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional["ThreadPoolExecutor"] = None
        """ Status """
        self.aborted = False
        self.do_train = False
        """ Web UI """
        self.webui_mode = os.environ.get("LLAMABOARD_ENABLED", "0").lower() in ["true", "1"]
        if self.webui_mode:
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = LoggerHandler(output_dir)
            logging.root.addHandler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: Dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of the initialization of the `Trainer`.
        """
        if (
                args.should_save
                and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
                and args.overwrite_output_dir
        ):
            logger.warning("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        self._close_thread_pool()

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if not self.do_train:
            self._close_thread_pool()

    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a successful prediction.
        """
        if not self.do_train:
            self._close_thread_pool()

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after logging the last logs.
        """
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            accuracy=state.log_history[-1].get("rewards/accuracies", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ["loss", "learning_rate", "epoch"]):
            logger.info(
                "{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                    logs["loss"], logs["learning_rate"], logs["epoch"]
                )
            )

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    def on_prediction_step(
            self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        r"""
        Event called after a prediction step.
        """
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class RikiLogCallback(TrainerCallback):
    def __init__(self, runner=None, socket=None, data=None, output_dir=None):
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.sio = socket
        self.data = data
        self.output_dir = output_dir

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps

        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        self.sio.emit("train_begin", {'modelId': self.data['modelId']})

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.in_training is not True:
            model_str = zip_dir(self.output_dir, self.output_dir)
            with open(model_str, 'rb') as f:
                # 定义files参数，用于上传文件
                files = {'file': (model_str, f)}

                # 发送POST请求上传文件
                response = requests.post(riki_config('riki.server_url') + riki_config('riki.upload_uri'), files=files)
                # 打印响应内容
                print(response.text)
                self.sio.emit("train_end", {'modelId': self.data['modelId'], 'modelUrl': response.json()['data']})

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(
            self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs
    ):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            accuracy=state.log_history[-1].get("rewards/accuracies", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if self.runner is not None:
            logger.info(
                "{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                    logs["loss"] or 0, logs["learning_rate"] or 0, logs["epoch"] or 0
                )
            )
        self.sio.emit("train_info", {'modelId': self.data['modelId'], 'info': logs})
        # os.makedirs(args.output_dir, exist_ok=True)
        # with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
        #     f.write(json.dumps(logs) + "\n")

    def on_prediction_step(
            self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()


class RikiLogCallbackRedis(TrainerCallback):
    def __init__(self, runner=None, data=None, output_dir=None):
        import redis
        redis_host = riki_config('redis.host')
        redis_port = int(riki_config('redis.port'))
        redis_password = riki_config('redis.password')
        redis_db = int(riki_config('redis.db'))
        self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password,
                             decode_responses=True)
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.data = data
        self.output_dir = output_dir

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps

        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        self.r.publish("train_model",
                       json.dumps({'modelId': self.data.model_id, 'type': 'train_begin'}, ensure_ascii=False))

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.in_training is not True:
            model_str = zip_dir(self.output_dir, self.output_dir)
            with open(model_str, 'rb') as f:
                # 定义files参数，用于上传文件
                files = {'file': (model_str, f)}

                # 发送POST请求上传文件
                response = requests.post(riki_config('riki.server_url') + riki_config('riki.upload_uri'), files=files)
                # 打印响应内容
                print(response.text)
                self.r.publish("train_model", json.dumps(
                    {'modelId': self.data.model_id, 'modelUrl': response.json()['data'], 'type': 'train_end'},
                    ensure_ascii=False))

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(
            self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs
    ):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            accuracy=state.log_history[-1].get("rewards/accuracies", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if self.runner is not None:
            logger.info(
                "{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                    logs["loss"] or 0, logs["learning_rate"] or 0, logs["epoch"] or 0
                )
            )
        self.r.publish("train_model", json.dumps(
            {'modelId': self.data.model_id, 'info': logs, 'type': 'train_info'},
            ensure_ascii=False))
        # os.makedirs(args.output_dir, exist_ok=True)
        # with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
        #     f.write(json.dumps(logs) + "\n")

    def on_prediction_step(
            self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()
