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
        data = pd.DataFrame(list(self.collection.find()))
        return data
