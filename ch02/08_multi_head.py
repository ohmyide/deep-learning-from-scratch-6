import torch
import torch.nn as nn
import torch.nn.functional as F

B = 2   # 中文注释：（batch_size）
C = 4   # 中文注释：（context_len）
E = 16  # 中文注释：（embed_dim）
H = 3   # 中文注释：（n_head）
D = 8   # 中文注释：（head_dim）

# 中文注释：请结合相邻代码理解此步骤
x = torch.randn(B, C, E)

# 中文注释：：
W_q = nn.Linear(E, H*D, bias=False)
W_k = nn.Linear(E, H*D, bias=False)
W_v = nn.Linear(E, H*D, bias=False)

Q = W_q(x)  # (B, C, H*D)
K = W_k(x)  # (B, C, H*D)
V = W_v(x)  # (B, C, H*D)

# 中文注释：请结合相邻代码理解此步骤
Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
scores = scores / (D ** 0.5)

# 中文注释：请结合相邻代码理解此步骤
mask = torch.tril(torch.ones(C, C, device=scores.device))
scores = scores.masked_fill(mask == 0, float('-inf'))

# 中文注释：Attention
weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
hidden = torch.matmul(weights, V)   # (B, H, C, D)

# 中文注释：: (B, H, C, D) → (B, C, H*D)
hidden = hidden.transpose(1, 2)  # (B, C, H, D)
hidden = hidden.contiguous().view(B, C, H*D)  # (B, C, H*D)

# 中文注释：: (B, C, H*D) → (B, C, E)
W_o = nn.Linear(H*D, E, bias=False)
output = W_o(hidden)  # (B, C, E)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        # 中文注释：Dropout
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, E = x.shape  # 中文注释：、、
        H, D = self.n_head, self.head_dim  # 中文注释：、

        # 中文注释：Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 中文注释：请结合相邻代码理解此步骤
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
        scores = scores / (D ** 0.5)

        # 中文注释：请结合相邻代码理解此步骤
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 中文注释：Attention
        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        weights = self.attention_dropout(weights)  # 中文注释：weightsDropout
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        # 中文注释：请结合相邻代码理解此步骤
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H*D)
        output = self.W_o(hidden)  # (B, C, E)
        output = self.output_dropout(output)  # 中文注释：Dropout

        return output

# 中文注释：请结合相邻代码理解此步骤
embed_dim = 512
n_head = 8
head_dim = 64

mha = MultiHeadAttention(embed_dim, n_head, head_dim)

# 中文注释：请结合相邻代码理解此步骤
batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

# 中文注释：请结合相邻代码理解此步骤
output = mha(x)
print(f"入力形状: {x.shape}")       #  (2, 10, 512)
print(f"出力形状: {output.shape}") #  (2, 10, 512)
