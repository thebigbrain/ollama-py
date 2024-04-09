keys = [
    "",
    "Key.down",
    "Key.up",
    "Key.left",
    "Key.right",
    "Key.enter",
    "Key.space",
    "Key.esc",
    "Key.shift",
    "Key.shift_r",
    "Key.ctrl_l",
    "Key.ctrl_r",
    "Key.alt_l",
    "Key.alt_r",
    "Key.tab",
    "Key.caps_lock",
    "Key.f1",
    "Key.f2",
    "Key.f3",
    "Key.f4",
    "Key.f5",
    "Key.f6",
    "Key.f7",
    "Key.f8",
    "Key.f9",
    "Key.f10",
    "Key.f11",
    "Key.f12",
    "Key.print_screen",
    "Key.scroll_lock",
    "Key.pause",
    "Key.insert",
    "Key.home",
    "Key.page_up",
    "Key.delete",
    "Key.end",
    "Key.page_down",
    "Key.num_lock",
    "Key.backspace",
    "Key.menu",
    # 字符按键
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "`",
    "~",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "-",
    "_",
    "=",
    "+",
    "[",
    "{",
    "]",
    "}",
    "\\",
    "|",
    ";",
    ":",
    "'",
    '"',
    ",",
    "<",
    ".",
    ">",
    "/",
    "?",
]

# 使用enumerate生成一个字典，其中字符串keys中的项作为键，生成的唯一整数作为值
keys_mapping = {key: index for index, key in enumerate(keys)}
