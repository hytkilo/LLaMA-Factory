import socketio
from llamafactory.train.tuner import run_riki_exp
from riki import riki_config


def main():
    sio = socketio.Client()
    sio.connect(riki_config('riki.admin_url') + riki_config('riki.socket_uri')
                + '&clientId=' + riki_config('train.device_id')
                + '&vram=' + riki_config('train.vram'), transports=['websocket'], retry=True, wait_timeout=2)
    print("连接riki服务成功...")
    @sio.event
    def do_train(data):
        print(data)
        run_riki_exp(sio=sio, data=data)

    sio.wait()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
