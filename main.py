from xlab.views.app import get_application_context
from xlab.views.floating_icon import create_floating_icon


if __name__ == "__main__":
    # 使用方式：
    context = get_application_context()
    # 在此配置应用程序窗口和其他组件...
    _ = create_floating_icon(context)
    context.start()
