from llmtuner import run_riki_exp
import socketio


def main():
    sio = socketio.Client()
    sio.connect('wss://admin.yunhelp.com/socket.io/?socketType=train_client&clientId=111', transports=['websocket'],
                retry=True, wait_timeout=2)

    @sio.event
    def do_train(data):
        print(data)
        run_riki_exp(sio, data)

    sio.wait()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
