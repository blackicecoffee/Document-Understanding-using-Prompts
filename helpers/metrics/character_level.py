def character_level_score(pred: str, gt: str):
    len_p, len_g = len(pred), len(gt)

    # Create matrix
    dp = [[0] * (len_g + 1) for _ in range(len_p + 1)]
    for i in range(len_p + 1):
        dp[i][0] = i
    for j in range(len_g + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, len_p + 1):
        for j in range(1, len_g + 1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )

    # Backtrack to count matches
    i, j = len_p, len_g
    match = 0
    while i > 0 and j > 0:
        if pred[i-1] == gt[j-1]:
            match += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1

    tp = match
    fp = len(pred) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}