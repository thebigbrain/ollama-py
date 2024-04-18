import string


class ComprehensiveActionSpace:
    def __init__(self):
        self.keyboard_actions = self.generate_keyboard_actions()
        self.mouse_actions = [
            "mouse_left_click",
            "mouse_right_click",
            "mouse_middle_click",
            "mouse_scroll_up",
            "mouse_scroll_down",
            "mouse_move_up",
            "mouse_move_down",
            "mouse_move_left",
            "mouse_move_right",
        ]
        # 合并所有操作作为行动空间
        self.action_space = self.keyboard_actions + self.mouse_actions

    def generate_keyboard_actions(self):
        """
        生成键盘行动列表，包括所有英文字符、数字、特殊符号及控制键。
        """
        # 基本字母和数字
        basic_actions = [char for char in string.ascii_letters + string.digits]

        # 特殊字符，可以根据需要增减
        special_actions = [char for char in string.punctuation]

        # 控制键
        control_actions = [
            "Key.down",
            "Key.up",
            "Key.left",
            "Key.right",
            "Key.enter",
            "Key.space",
            "Key.esc",
            "Key.shift",
            "Key.ctrl",
            "Key.alt",
            "Key.tab",
            "Key.caps_lock",
            "Key.delete",
            "Key.home",
            "Key.end",
            "Key.page_up",
            "Key.page_down",
        ]

        # 功能键 F1-F12
        function_actions = ["F" + str(i) for i in range(1, 13)]

        # 组合所有键盘操作
        keyboard_actions = (
            basic_actions + special_actions + control_actions + function_actions
        )

        return keyboard_actions

    def print_action_space(self):
        """打印所有可行的行动空间"""
        for action in self.action_space:
            print(action)

    def get_action_space(self):
        return self.action_space

    def get_num_actions(self):
        return len(self.action_space)


if __name__ == "__main__":
    # 使用这个类
    action_space = ComprehensiveActionSpace()
    # 用户的键盘输入
    user_input = "A"

    # 找到该行为在行动空间中的位置，此处 index 就是把行为转化成的动作
    index = action_space.get_action_space().index(user_input)

    # 打印动作
    print(f"Action for input {user_input} is: {index}")
