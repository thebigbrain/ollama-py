def calculate_grid_size(screen_width, screen_height, grid_rows, grid_cols):
    grid_width = screen_width // grid_cols
    grid_height = screen_height // grid_rows
    return grid_width, grid_height


def create_grid(screen_width, screen_height, grid_rows=100, grid_cols=100):
    grid_width, grid_height = calculate_grid_size(
        screen_width, screen_height, grid_rows, grid_cols
    )
    grid_list = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * grid_width
            y = row * grid_height
            grid_list.append((x, y, grid_width, grid_height))

    return grid_list


if __name__ == "__main__":
    # 假设屏幕分辨率为1920x1080
    screen_width = 1920
    screen_height = 1080
    grid_list = create_grid(screen_width, screen_height)

    # 输出网格信息
    for i, grid in enumerate(grid_list):
        print(
            f"Grid {i}: Position (x={grid[0]}, y={grid[1]}), Size (width={grid[2]}, height={grid[3]})"
        )
