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
            gt_value = " ".join(ground_truth[k].lower().split())
            pred_value = " ".join(pred[k].lower().split())

            if gt_value == pred_value:
                em_score += 1

        elif k == "Table":
            gt_table = ground_truth[k]
            pred_table = pred[k]
            
            num_field += len(gt_table)

            for idx in range(len(pred_table)):
                if idx > len(gt_table): break

                gt_row = gt_table[idx]
                pred_row = pred_table[idx]

                for col_name in gt_row.keys():
                    if col_name not in pred_row: continue
                    gt_col_value = " ".join(gt_row[col_name].lower().split())
                    pred_col_value = " ".join(pred_row[col_name].lower().split())

                    if gt_col_value == pred_col_value:
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
            gt_value = " ".join(ground_truth[k].lower().split())
            pred_value = " ".join(pred[k].lower().split())

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([gt_value, pred_value])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            total_similarty += similarity

        elif k == "Table":
            gt_table = ground_truth[k]
            pred_table = pred[k]
            
            num_field += len(gt_table)

            for idx in range(len(pred_table)):
                if idx > len(gt_table): break

                gt_row = gt_table[idx]
                pred_row = pred_table[idx]

                for col_name in gt_row.keys():
                    if col_name not in pred_row: continue
                    gt_col_value = " ".join(gt_row[col_name].lower().split())
                    pred_col_value = " ".join(pred_row[col_name].lower().split())

                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([gt_col_value, pred_col_value])
                    
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
            gt_value = " ".join(ground_truth[k].lower().split())
            pred_value = " ".join(pred[k].lower().split())

            # Get exact match scores
            if gt_value == pred_value:
                em_score += 1

            # Get similarity scores:
            if len(pred_value) <= 1 or len(gt_value) <= 1:
                if gt_value == pred_value:
                    total_similarty += 1
            else:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([gt_value, pred_value])
                
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

                total_similarty += similarity

        elif k == "Table":
            if k not in pred: continue

            gt_table = ground_truth[k]
            pred_table = pred[k]

            num_field += len(gt_table) * len(gt_table[0].keys())
            if len(pred_table) == 0: continue

            for idx in range(len(pred_table)):
                if idx + 1 > len(gt_table): break

                gt_row = gt_table[idx]
                pred_row = pred_table[idx]

                for col_name in gt_row.keys():
                    if col_name not in pred_row: continue
                    gt_col_value = " ".join(gt_row[col_name].lower().split())
                    pred_col_value = " ".join(str(pred_row[col_name]).lower().split())

                    # Get exact match scores
                    if gt_col_value == pred_col_value:
                        em_score += 1

                    # Get similarity scores
                    if len(gt_col_value) <= 1 or len(pred_col_value) <= 1:
                        if gt_col_value == pred_col_value:
                            total_similarty += 1
                    else:
                        vectorizer = TfidfVectorizer()
                        
                        tfidf_matrix = vectorizer.fit_transform([gt_col_value, pred_col_value])
                        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

                        total_similarty += similarity
                    
    return {"EM": round(float(em_score) / float(num_field) * 100.0, 4), "similarity_score": round(float(total_similarty) / float(num_field) * 100.0, 4)}