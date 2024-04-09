from examples.autogui.db_operations import DBOperations

db = DBOperations("mongodb://localhost:27017/", "user_input", "events")

if __name__ == "__main__":
    results = db.collection.aggregate(
        [
            {
                "$group": {
                    "_id": "$key",
                }
            }
        ]
    )

    for result in results:
        print(result)
