import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# 中文注释：请结合相邻代码理解此步骤
device = get_device()
model_path = 'storybot/model_pretrain.pt'
tokenizer_path = 'storybot/merge_rules.pkl'

# 中文注释：请结合相邻代码理解此步骤
# 中文注释：prompt = "Once upon a time"  #
prompt = "<|endoftext|>"
max_new_tokens = 300  # 中文注释：请结合相邻代码理解此步骤
temperature = 1.0  # 中文注释：（）
num_samples = 3  # 中文注释：请结合相邻代码理解此步骤

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# 中文注释：请结合相邻代码理解此步骤
for i in range(num_samples):
    print(f"--- サンプル {i+1} ---")
    story = generate(
        model, tokenizer, prompt, max_new_tokens, temperature
    )
    print(story)
