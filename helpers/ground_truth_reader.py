from typing import List
import json

def read_ground_truth(image_path: str) -> dict:
    """Read ground truth from a image path"""
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

def read_formal_and_table_from_ground_truth(image_path: str) -> dict:
    """Split formal information and table information from ground truth"""
    ground_truth = read_ground_truth(image_path=image_path)

    formal = {k: v for k, v in ground_truth.items() if k != "Table"}
    if "Table" in ground_truth:
        table = ground_truth["Table"]
    else: table = []

    return formal, table

def read_fields_from_ground_truth(image_path: str) -> dict:
    """Read fields name from ground truth"""
    gt = read_ground_truth(image_path=image_path)
    
    if len(gt) == 0: return {}

    fields = {}
    for k in gt.keys():
        if k != "Table":
            fields[k] = ""
    
    return fields

def read_table_column_from_ground_truth(image_path: str) -> List[dict] | dict | None:
    """Read table's columns name from ground truth"""
    gt = read_ground_truth(image_path=image_path)

    if len(gt) == 0: return {}

    if "Table" not in gt: return None

    table_columns = {}
    row = gt["Table"][0]

    for col in row.keys():
        table_columns[col] = ""

    table_columns = [table_columns]

    return table_columns
