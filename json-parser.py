import ijson

filename = "/home/fatih/phd/mot_dataset/SOMPT22/images/annotations/updated_training_batch.json"
with open(filename, 'r') as file:
    objects = ijson.items(file, 'annotations')  # 'item' should be replaced with the appropriate key
    for obj in objects:
        # Analyze each object
        print(obj)