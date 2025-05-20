import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    total_similarty = 0
    num_field = 0

    for k in ground_truth.keys():
        if k not in pred:
            num_field += 1
            continue
        if k != "Table":
            num_field += 1
            gt_value = ground_truth[k].strip().replace(" ", "").lower()
            pred_value = pred[k].strip().replace(" ", "").lower()

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([gt_value, pred_value])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            total_similarty += similarity
    
    return round(float(total_similarty) / float(num_field) * 100.0, 4)

def get_all_scores(ground_truth: dict, pred: dict):
    """Calculate all the scores: Exact Match and similarity score"""
    em_score = 0
    total_similarty = 0
    num_field = 0

    for k in ground_truth.keys():
        if k not in pred:
            num_field += 1
            continue
        if k != "Table":
            num_field += 1
            gt_value = ground_truth[k].strip().replace(" ", "").lower()
            pred_value = pred[k].strip().replace(" ", "").lower()

            # Get exact match scores
            if gt_value == pred_value:
                em_score += 1

            # Get similarity scores:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([gt_value, pred_value])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            total_similarty += similarity

    return {"EM": round(float(em_score) / float(num_field) * 100.0, 4), "similarity_score": round(float(total_similarty) / float(num_field) * 100.0, 4)}