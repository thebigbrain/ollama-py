import os
import sys
import unittest


def resource_path(relative_path):
    """获取资源的绝对路径。用于访问程序需要的资源文件，使其既可以在开发环境中运行，
    也可以在打包成单文件后运行。"""

    # 如果应用程序被打包，则使用 `sys._MEIPASS` 中的路径
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        # 否则使用当前文件的路径
        base_path = os.path.dirname(os.path.abspath(os.path.join(__file__, "../../")))

    return os.path.join(base_path, relative_path)


def get_resource(name: str) -> str:
    return resource_path(os.path.join("assets", name))


def get_model_path(name: str) -> str:
    return resource_path(os.path.join("data", "models", f"{name}.pth"))


# unittest 用例


class TestResourcePath(unittest.TestCase):
    def test_resource_path(self):
        self.assertEqual(
            resource_path("assets/logo.png"),
            os.path.join(os.getcwd(), "assets/logo.png").lower(),
        )

    def test_get_resource(self):
        self.assertEqual(
            get_resource("logo.png"),
            os.path.join(os.getcwd(), "assets", "logo.png").lower(),
        )


# 运行单元测试
if __name__ == "__main__":
    unittest.main()
