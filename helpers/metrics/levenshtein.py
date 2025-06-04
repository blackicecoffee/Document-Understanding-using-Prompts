def levenshtein_distance(str1: str, str2: str) -> int | float:
    len1, len2 = len(str1), len(str2)
    
    # Create a 2D distance matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize the base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill in the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    
    return dp[len1][len2]

def normalized_levenshtein_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    distance = levenshtein_distance(s1, s2)
    return 1 - distance / max(len(s1), len(s2))