import pymongo
from examples.autogui.db_operations import DBOperations

db = DBOperations("mongodb://localhost:27017/", "user_input", "events")

if __name__ == "__main__":
    results = (
        db.collection.find({}, {"_id": 0})
        .limit(100)
        .sort("timestamp", pymongo.DESCENDING)
    )

    for result in results:
        print(result)
