from cefpython3 import cefpython as cef
import sys


def main():
    # 初始化CEF
    cef.Initialize()

    # 创建新的浏览器窗口
    cef.CreateBrowserSync(url="https://www.bing.com", window_title="Hello CEF Python!")

    # 进入消息循环
    cef.MessageLoop()

    # 清理CEF资源
    cef.Shutdown()


if __name__ == "__main__":
    main()
