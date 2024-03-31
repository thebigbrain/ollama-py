class DataUnderstanding:
    def __init__(self, data):
        self.data = data

    def basic_info(self):
        """显示数据的基本信息"""
        print(self.data.info())

    def event_counts(self):
        """显示各种类型事件的计数"""
        print(self.data['type'].value_counts())

    def mouse_coordinates(self):
        """显示鼠标坐标的统计情况"""
        if 'x' in self.data.columns and 'y' in self.data.columns:
            print(self.data[['x', 'y']].describe())

    def most_common_keys(self):
        """显示最常按的前10个键"""
        if 'key' in self.data.columns:
            print(self.data['key'].value_counts().head(10))
