import queue
import signal
import sys
import time

from xlab.core.thread_group import ThreadGroup

# 定义队列
event_queue = queue.Queue()


# 定义数据收集线程
def collect_data():
    while True:
        # 收集交互数据
        data = ...

        # 将数据放入队列
        event_queue.put(data)

        time.sleep(1)


# 定义模型训练线程
def train_model():
    while True:
        # 从队列中获取数据
        data = event_queue.get()

        # 使用模型训练框架来训练模型
        # ...

        # 保存模型
        # ...


def signal_handler(sig, frame):
    # 退出应用
    print("退出主线程", sig, frame)
    ThreadGroup.stop()
    sys.exit()


if __name__ == "__main__":
    # 创建数据收集线程和模型训练线程
    signal.signal(signal.SIGINT, signal_handler)

    ThreadGroup.add_daemon(collect_data)
    ThreadGroup.add_daemon(train_model)

    ThreadGroup.start()

    print("程序已启动!")

    while True:
        time.sleep(1)
