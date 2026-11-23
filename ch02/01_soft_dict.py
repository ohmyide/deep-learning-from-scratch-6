import numpy as np

# 中文注释：请结合相邻代码理解此步骤
d = {
    'apple': 100,
    'banana': 200,
    'cherry': 300,
    'durian': 400
}

# 中文注释：ー
query = 'banana'
print(d[query])  # 200


# 中文注释：ー: (, , ) 0〜10
# 中文注释：ー: ーー (0〜100)
movie_preferences = {
    (8, 2, 3): 85,  # 中文注释：请结合相邻代码理解此步骤
    (3, 9, 1): 70,  # 中文注释：请结合相邻代码理解此步骤
    (1, 2, 9): 60,  # 中文注释：请结合相邻代码理解此步骤
    (5, 5, 5): 75,  # 中文注释：请结合相邻代码理解此步骤
    (7, 6, 2): 80,  # 中文注释：请结合相邻代码理解此步骤
    (2, 7, 6): 65,  # 中文注释：请结合相邻代码理解此步骤
    (9, 1, 1): 90,  # 中文注释：请结合相邻代码理解此步骤
}

# 中文注释：请结合相邻代码理解此步骤
new_movie = (6, 4, 5)

def soft_dictionary(query, dictionary):
    # 中文注释：请结合相邻代码理解此步骤
    similarity = []
    for key in dictionary:
        s = np.dot(query, key)
        similarity.append(s)

    # 中文注释：请结合相邻代码理解此步骤
    exp_similarity = np.exp(similarity)
    weights = exp_similarity / np.sum(exp_similarity)

    # 中文注释：请结合相邻代码理解此步骤
    result = 0
    for weight, value in zip(weights, dictionary.values()):
        result += weight * value

    return result, weights


predicted_rating, weights = soft_dictionary(new_movie, movie_preferences)

print(f"新しい映画 {new_movie} の予測評価: {predicted_rating:.2f} 点")
print("\n各映画の重み:")
for key, weight in zip(movie_preferences.keys(), weights):
    print(f"映画 {key}: {weight*100:.2f}%")
