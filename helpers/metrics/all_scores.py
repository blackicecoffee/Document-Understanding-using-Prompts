import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpers.metrics.exact_match import exact_match
from helpers.metrics.similarity import tf_idf

"""
Metrics to evaluate model performance
"""

def get_all_scores(ground_truth: dict, pred: dict):
    """Calculate all the scores: Exact Match and similarity score"""
    em_score = 0                            # Total exact match scores
    total_similarity_tfidf = 0              # Total similarity scores using TF-IDF
    total_similarity_sbert = 0              # Total similarity scores using Sentence Transformers
    total_nls = 0                           # Total similarity scores using Normalized Levenshtein Similarity (NLS)
    total_precision = 0                     # Total Character-level Precision
    total_recall = 0                        # Total Character-level Recall
    total_f1 = 0                            # Total Character-level F1
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
            em_score += exact_match(gt_value, pred_value)

            # Get similarity scores:
            if len(pred_value) <= 1 or len(gt_value) <= 1:
                if gt_value == pred_value:
                    total_similarity_tfidf += 1
            else:
                similarity_tfidf = tf_idf(gt_value, pred_value)

                total_similarity_tfidf += similarity_tfidf

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
                    em_score += exact_match(gt_col_value, pred_col_value)

                    # Get similarity scores
                    if len(gt_col_value) <= 1 or len(pred_col_value) <= 1:
                        if gt_col_value == pred_col_value:
                            total_similarity_tfidf += 1
                    else:
                        similarity_tfidf = tf_idf(gt_col_value, pred_col_value)

                        total_similarity_tfidf += similarity_tfidf
                    
    return {"EM": round(float(em_score) / float(num_field) * 100.0, 4), "similarity_score": round(float(total_similarity_tfidf) / float(num_field) * 100.0, 4)}

def get_chacter_level_score(ground_truth: dict, pred: dict):
    """Calculate F1 score"""
    pass

def get_exact_match(ground_truth: dict, pred: dict):
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

            em_score += exact_match(gt_value, pred_value)

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

                    em_score += exact_match(gt_col_value, pred_col_value)

    return round(float(em_score) / float(num_field) * 100.0, 4)

def get_similarity_score(ground_truth: dict, pred: dict):
    """Calculate similarity score"""
    total_similarity_tfidf = 0
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

            total_similarity_tfidf += similarity

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

                    total_similarity_tfidf += similarity
    
    return round(float(total_similarity_tfidf) / float(num_field) * 100.0, 4)
