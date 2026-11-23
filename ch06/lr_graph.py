import matplotlib.pyplot as plt
import numpy as np

# 中文注释：请结合相邻代码理解此步骤
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS
# plt.rcParams['font.family'] = 'Yu Gothic'  # Windows

# 中文注释：请结合相邻代码理解此步骤
warmup_ratio = 0.05  # 中文注释：ー（5%）
eta_min_ratio = 0.1  # 中文注释：ー（10%）

# 中文注释：请结合相邻代码理解此步骤
t = np.linspace(0, 1, 1000)

# 中文注释：ー（ー）
def cosine_annealing(t, warmup_ratio, eta_min_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # 中文注释：ー: 0 -> 1
            lr[i] = ti / warmup_ratio
        else:
            # 中文注释：ー: 1 -> eta_min
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + np.cos(np.pi * progress))
    return lr

# 中文注释：D2Z（ー）
def d2z(t, warmup_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # 中文注释：ー: 0 -> 1
            lr[i] = ti / warmup_ratio
        else:
            # 中文注释：: 1 -> 0
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = 1 - progress
    return lr

cosine_lr = cosine_annealing(t, warmup_ratio, eta_min_ratio)
d2z_lr = d2z(t, warmup_ratio)

# 中文注释：请结合相邻代码理解此步骤
fig, ax = plt.subplots(figsize=(10, 6))

# 中文注释：ーー
ax.axvspan(0, warmup_ratio, color='lightgray', alpha=0.5)
ax.text(0.01, 1.02, 'ウォームアップ', fontsize=10, va='bottom')

# 中文注释：请结合相邻代码理解此步骤
ax.plot(t, cosine_lr, 'b-', linewidth=2, label='コサインアニーリング')
ax.plot(t, d2z_lr, color='orange', linestyle='--', linewidth=2, label='D2Z')

# 中文注释：请结合相邻代码理解此步骤
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_xlabel('学習の進行度', fontsize=12)
ax.set_ylabel('学習率', fontsize=12)

# 中文注释：请结合相邻代码理解此步骤
ax.grid(True, linestyle='--', alpha=0.7)

# 中文注释：请结合相邻代码理解此步骤
ax.legend(loc='upper right', fontsize=12)

# 中文注释：请结合相邻代码理解此步骤
plt.tight_layout()

# 中文注释：PNG
plt.savefig('lr_schedule.png', format='png', bbox_inches='tight')
plt.close()
