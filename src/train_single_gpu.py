import socketio
from llmtuner import run_riki_exp


def main():
    sio = socketio.Client()
    sio.connect('https://admin.yunhelp.com/socket.io/?socketType=train_client&clientId=111&vram=24', transports=['websocket'], retry=True, wait_timeout=2)

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
