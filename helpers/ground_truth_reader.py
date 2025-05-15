import json

def read_ground_truth(image_path: str):
    image_name = image_path.split(".")[0]
    label_name = image_name.replace("images", "labels")
    
    label_path = f".{label_name}.json"
    try:
        with open(label_path, "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("File not found!")

    return ground_truth