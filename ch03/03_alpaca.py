import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
from codebot.tokenizer import BPETokenizer

# 中文注释：ー
tokenizer = BPETokenizer.load_from('codebot/merge_rules.pkl')

# 中文注释：JSON
with open('codebot/tiny_codes_sft.json') as f:
    data = json.load(f)

# 中文注释：1
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# 中文注释：Alpaca
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>

# 中文注释：请结合相邻代码理解此步骤
ids = tokenizer.encode(text)
print(ids)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]
