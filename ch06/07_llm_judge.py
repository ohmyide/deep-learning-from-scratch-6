import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
import statistics
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

# 中文注释：（ー）
model_paths = {
    500: 'storybot/model_iter_500.pt',
    5000: 'storybot/model_iter_5000.pt',
    40000: 'storybot/model_pretrain.pt',
}

# 中文注释：请结合相邻代码理解此步骤
prompt = "<|endoftext|>"
max_new_tokens = 200
temperature = 1.0
num_samples = 10  # 中文注释：请结合相邻代码理解此步骤

def evaluate_story(client, story):
    """使用 LLM-as-a-Judge 评估故事。"""

    evaluation_prompt = f"""以下の子供向けストーリーを2つの観点で1-5点で評価してください。

ストーリー:
{story}

評価観点:
1. Coherence（一貫性）: 論理的につながっているか、物語として筋が通っているか
2. Grammar（文法）: 文法的に正しい英語か

以下のJSON形式で回答してください:
{{
    "coherence": <1-5の整数>,
    "grammar": <1-5の整数>,
    "comment": "<評価の簡単な理由>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=300,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    print("=====出力====")
    print(text)

    # 中文注释：response_format、ー
    return json.loads(text)

results = {}
for iteration, model_path in model_paths.items():
    print(f"\n{'='*50}")
    print(f"Iteration {iteration}")
    print('='*50)

    model = GPT.load_from(model_path, device=device)
    iteration_results = []

    for i in range(num_samples):
        print(f"\n--- サンプル {i+1} ---")

        # 中文注释：请结合相邻代码理解此步骤
        story = generate(model, tokenizer, prompt, max_new_tokens, temperature)
        print(f"Story: {story[:200]}...")

        # 中文注释：LLM-as-a-Judge
        scores = evaluate_story(client, story)
        print(f"Scores: {scores}")

        iteration_results.append({
            "story": story,
            "scores": scores
        })

    results[iteration] = iteration_results

# 中文注释：ー
print("\n" + "="*50)
print("Summary")
print("="*50)

for iteration in model_paths.keys():
    scores_list = [r["scores"] for r in results[iteration]]

    print(f"\nIteration {iteration}:")
    for key in ["coherence", "grammar"]:
        values = [s[key] for s in scores_list]
        avg = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {key}: {avg:.2f} ± {std:.2f}")
