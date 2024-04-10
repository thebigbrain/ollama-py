import socket
import signal
import sys
import threading


PORT = 2080
TARGET_HOST = "127.0.0.1"
TARGET_PORT = "7890"


def signal_handler(signum, frame):
    print("Received Ctrl+C")
    sys.exit()


def handle_client(client_socket):
    # 接收来自客户端的请求
    request = client_socket.recv(1024)

    # 解析请求
    # ...

    # 将请求转发给目标服务器
    target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    target_socket.connect((TARGET_HOST, TARGET_PORT))
    target_socket.sendall(request)

    # 接收目标服务器的响应
    response = b""
    while True:
        data = target_socket.recv(1024)
        if not data:
            break
        response += data

    # 将响应发送给客户端
    client_socket.sendall(response)

    # 关闭连接
    client_socket.close()
    target_socket.close()


def start_proxy(server_socket, event):
    while True:
        client_socket, client_address = server_socket.accept()
        try:
            handle_client(client_socket)
        except Exception:
            print("An exception occurred in the proxy thread")
            client_socket.close()
            server_socket.close()
            event.set()
            sys.exit()


def main():
    signal.signal(signal.SIGINT, signal_handler)

    event = threading.Event()

    # 创建一个 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 设置 socket 为可重用
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定端口号
    server_socket.bind(("", PORT))

    # 监听连接
    server_socket.listen()

    print(f"proxy server listening :{PORT}")

    proxy_thread = threading.Thread(target=start_proxy, args=(server_socket, event))
    proxy_thread.start()

    print(f"proxy thread started.")

    event.wait()

    server_socket.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An exception occurred in the main thread")
        sys.exit()
