import matplotlib.pyplot as plt
import numpy as np

# 中文注释：请结合相邻代码理解此步骤
x = np.linspace(-3, 3, 500)

# 中文注释：ReLU
relu = np.maximum(0, x)

# 中文注释：GELU（）
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 中文注释：请结合相邻代码理解此步骤
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, relu, 'b-', linewidth=2, label='ReLU')
ax.plot(x, gelu, 'r-', linewidth=2, label='GELU')

# 中文注释：请结合相邻代码理解此步骤
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)

# 中文注释：请结合相邻代码理解此步骤
ax.grid(True, linestyle='--', alpha=0.7)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

# 中文注释：请结合相邻代码理解此步骤
ax.legend(loc='upper left', fontsize=12)

# 中文注释：请结合相邻代码理解此步骤
plt.tight_layout()

# 中文注释：PNG
plt.savefig('relu_gelu.png', format='png', bbox_inches='tight')
plt.close()
