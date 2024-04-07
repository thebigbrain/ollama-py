
from xlab.win.active import get_active_win
from xlab.win.win import get_window_info
from xlab.win.info import print_win_info

if __name__ == '__main__':
    win = get_active_win()
    print_win_info(get_window_info(win))

