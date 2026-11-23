import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import pickle
from codebot.tokenizer import train_bpe

vocab_size = 1000  # 中文注释：请结合相邻代码理解此步骤
text = open("codebot/tiny_codes.txt").read()
merge_rules = train_bpe(text, vocab_size)

# 中文注释：ー
with open("codebot/merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)
