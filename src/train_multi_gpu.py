import socketio
from typing import Any, Dict
import json
import os.path
from riki import riki_config, gen_dict
import redis
import subprocess

redis_host = riki_config('redis.host')
redis_port = int(riki_config('redis.port'))
redis_password = riki_config('redis.password')
redis_db = int(riki_config('redis.db'))
r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password, decode_responses=True)
sub = r.pubsub()

training_model = ''


def main():
    sio = socketio.Client()

    @sio.event
    def do_train(data):
        print(data)
        args = gen_dict(data)
        cmd_text = gen_cmd(args)
        env_vars = {"CUDA_VISIBLE_DEVICES": riki_config('train.devices')}
        env = os.environ.copy()
        env.update(env_vars)
        global training_model
        training_model = data['modelId']
        cmd = "".join(cmd_text)
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    sio.connect(riki_config('riki.admin_url') + riki_config('riki.socket_uri')
                + '&clientId=' + riki_config('train.device_id')
                + '&vram=' + riki_config('train.vram')
                , transports=['websocket'],
                retry=True, wait_timeout=2)

    def message_handler(message):
        data = json.loads(message['data'])
        print(training_model)
        if str(training_model) == data['modelId']:
            sio.emit(data['type'], data)
        print(f"Received: {message}")

    sub.subscribe(**{'train_model': message_handler})
    sub.run_in_thread(sleep_time=0.001)

    sio.wait()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def gen_cmd(args: Dict[str, Any]) -> str:
    cmd_lines = [
        "conda run -n riki_lf accelerate launch --config_file ../examples/accelerate/single_config.yaml train.py"]
    for k, v in args.items():
        if v is not None and v is not False and v != "":
            cmd_lines.append(" --{} {} ".format(k, str(v)))
    # cmd_text = "\\\n".join(cmd_lines)
    # cmd_text = "```bash\n{}\n```".format(cmd_text)
    return cmd_lines


if __name__ == "__main__":
    main()
