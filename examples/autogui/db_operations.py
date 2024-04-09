from datetime import datetime, timedelta

from pymongo import MongoClient
import pandas as pd


class DBOperations:
    def __init__(self, uri, db_name, collection_name):
        # 连接到mongodb数据库
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    # 插入数据
    def insert_data(self, data):
        self.collection.insert_one(data)

    # 读取所有数据
    def read_data(self):
        # 计算当前时间和12小时前的时间
        # current_time = datetime.now()
        # twelve_hours_ago = current_time - timedelta(hours=24)
        #
        # # 构建查询，只返回最近12小时的数据
        # query = {"timestamp": {"$gte": twelve_hours_ago}}
        d = list(self.collection.find())
        data = pd.DataFrame(d)
        return data
