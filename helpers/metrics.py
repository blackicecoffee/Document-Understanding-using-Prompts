import numpy as np

"""
Metrics to evaluate model performance
"""

def f1_score(ground_truth: dict, pred: dict):
    """Calculate F1 score"""
    pass

def exact_match(ground_truth: dict, pred: dict):
    """Calculate Exact Match score"""
    em_score = 0
    num_field = 0

    for k in ground_truth.keys():
        if k not in pred:
            num_field += 1
            continue
        if k != "Table":
            num_field += 1
            gt_value = ground_truth[k].strip().replace(" ", "").lower()
            pred_value = pred[k].strip().replace(" ", "").lower()

            if gt_value == pred_value:
                em_score += 1

    return round(float(em_score) / float(num_field) * 100.0, 4)

def similarity_score(ground_truth: dict, pred: dict):
    """Calculate similarity score"""
    pass