import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
import torch
import torch.nn.functional as F
from openai import OpenAI
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# 中文注释：请结合相邻代码理解此步骤
# ==========================================
client = OpenAI(api_key="your_api_key_here")
# ==========================================
device = get_device()
tokenizer_path = 'storybot/merge_rules.pkl'
tokenizer = BPETokenizer.load_from(tokenizer_path)

# 中文注释：请结合相邻代码理解此步骤
model_paths = {
    'pretrain': 'storybot/model_pretrain.pt',
    'dpo': 'storybot/model_dpo.pt',
}

# 中文注释：请结合相邻代码理解此步骤
prompt = "Once upon a time"
num_comparisons = 100  # 中文注释：请结合相邻代码理解此步骤
max_new_tokens = 150
temperature = 1.0


def compare_stories(client, story_a, story_b):
    """比较两个故事，判断哪一个拥有更圆满的结局。"""

    evaluation_prompt = f"""以下の2つの子供向けストーリーを比較し、どちらがよりハッピーエンドかを判定してください。

【Story A】
{story_a}

【Story B】
{story_b}

どちらがより明るく幸せな結末か、または希望に満ちた内容かを判断してください。
JSON形式で回答: {{"winner": "A" or "B" or "tie", "reason": "簡潔な理由"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=150,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    return json.loads(text)

# 中文注释：ー
model_pretrain = GPT.load_from(model_paths['pretrain'], device=device)
model_dpo = GPT.load_from(model_paths['dpo'], device=device)

# 中文注释：请结合相邻代码理解此步骤
results = []
wins = {"pretrain": 0, "dpo": 0, "tie": 0}

for i in range(num_comparisons):
    print(f"\n{'='*60}")
    print(f"Comparison {i+1}/{num_comparisons}")
    print('='*60)

    # 中文注释：请结合相邻代码理解此步骤
    story_pretrain = generate(model_pretrain, tokenizer, prompt, max_new_tokens, temperature)
    story_dpo = generate(model_dpo, tokenizer, prompt, max_new_tokens, temperature)

    print(f"\n[Pretrain]: {story_pretrain[:100]}...")
    print(f"\n[DPO]: {story_dpo[:100]}...")

    # 中文注释：、
    import random
    if random.random() < 0.5:
        story_a, story_b = story_pretrain, story_dpo
        mapping = {"A": "pretrain", "B": "dpo"}
    else:
        story_a, story_b = story_dpo, story_pretrain
        mapping = {"A": "dpo", "B": "pretrain"}

    # 中文注释：LLM-as-a-Judge
    judgment = compare_stories(client, story_a, story_b)

    winner_label = judgment["winner"]
    if winner_label == "tie":
        winner = "tie"
    else:
        winner = mapping[winner_label]

    wins[winner] += 1

    print(f"\n🏆 Winner: {winner}")
    print(f"   Reason: {judgment['reason']}")

    results.append({
        "story_pretrain": story_pretrain,
        "story_dpo": story_dpo,
        "winner": winner,
        "reason": judgment["reason"]
    })

# 中文注释：ー
print("\n" + "="*60)
print("📊 PAIRWISE COMPARISON RESULTS")
print("="*60)

total = num_comparisons
print(f"\n  Pretrain wins: {wins['pretrain']:3d} ({wins['pretrain']/total*100:5.1f}%)")
print(f"  DPO wins:      {wins['dpo']:3d} ({wins['dpo']/total*100:5.1f}%)")
print(f"  Ties:          {wins['tie']:3d} ({wins['tie']/total*100:5.1f}%)")

# 中文注释：（tie）
if wins['pretrain'] + wins['dpo'] > 0:
    dpo_winrate = wins['dpo'] / (wins['pretrain'] + wins['dpo']) * 100
    print(f"\n  DPO win rate (excluding ties): {dpo_winrate:.1f}%")
