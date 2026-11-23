import torch
import torch.nn.functional as F

# 中文注释：ー（）
K = torch.tensor([
    [8, 2, 3],  # 中文注释：请结合相邻代码理解此步骤
    [3, 9, 1],  # 中文注释：请结合相邻代码理解此步骤
    [1, 2, 9],  # 中文注释：请结合相邻代码理解此步骤
    [5, 5, 5],  # 中文注释：请结合相邻代码理解此步骤
    [7, 6, 2],  # 中文注释：请结合相邻代码理解此步骤
    [2, 7, 6],  # 中文注释：请结合相邻代码理解此步骤
    [9, 1, 1],  # 中文注释：请结合相邻代码理解此步骤
], dtype=torch.float32)

# 中文注释：ー（ーー）
V = torch.tensor([
    [85],
    [70],
    [60],
    [75],
    [80],
    [65],
    [90]
], dtype=torch.float32)

# 中文注释：（）
Q = torch.tensor([
    [6, 4, 5],  # 中文注释：请结合相邻代码理解此步骤
    [2, 8, 3],  # 中文注释：请结合相邻代码理解此步骤
    [4, 3, 7],  # 中文注释：请结合相邻代码理解此步骤
], dtype=torch.float32)

def attention(Q, K, V):
    similarity = torch.matmul(Q, K.t())     # 中文注释：QK^T
    weights = F.softmax(similarity, dim=1)  # 中文注释：请结合相邻代码理解此步骤
    output = torch.matmul(weights, V)       # 中文注释：请结合相邻代码理解此步骤
    return output, weights

predicted_ratings, weights = attention(Q, K, V)

# 中文注释：请结合相邻代码理解此步骤
for movie, rating in zip(Q, predicted_ratings):
    print(f"映画 {movie.numpy()} の予測評価: {rating.item():.2f}")
