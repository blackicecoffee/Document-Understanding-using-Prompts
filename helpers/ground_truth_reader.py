import json

def read_ground_truth(image_path: str) -> dict:
    image_name = image_path.split(".")[0]
    label_name = image_name.replace("images", "labels")
    
    label_path = f"{label_name}.json"
    ground_truth = ""

    try:
        with open(label_path, "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("File not found!")

    return ground_truth

def read_fields_from_ground_truth(image_path: str) -> dict:
    gt = read_ground_truth(image_path=image_path)
    
    if len(gt) == 0: return {}

    # Currently skip "Table"
    fields = {}
    for k in gt.keys():
        if k != "Table":
            fields[k] = ""
    
    return fields