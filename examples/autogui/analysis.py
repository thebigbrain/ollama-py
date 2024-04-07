from pymongo import MongoClient
import pandas as pd

# 已经连接到mongodb数据库
client = MongoClient('mongodb://localhost:27017/')
db = client['user_input']
collection = db['events']

# 从MongoDB读取数据
data = pd.DataFrame(list(collection.find()))

# 转换时间戳为日期时间类型
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 设定时间戳为数据框的索引
data.set_index('timestamp', inplace=True)

# 显示数据总览
print(data.info())

# 显示各种类型事件的计数
print(data['type'].value_counts())

# 如果有鼠标位置信息，显示其统计描述
if 'x' in data.columns and 'y' in data.columns:
    print(data[['x', 'y']].describe())

# 如果有按键信息，显示最常按的前10个键
if 'key' in data.columns:
    print(data['key'].value_counts().head(10))
