import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device

# 中文注释：请结合相邻代码理解此步骤
device = get_device()
model_path = 'codebot/model_sft.pt'
# model_path = 'codebot/model_grpo.pt'
tokenizer_path = 'codebot/merge_rules.pkl'
max_new_tokens = 200
temperature = 1.0

def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n\n### Response:\n"

# 中文注释：ー
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    # 中文注释：ー
    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    # 中文注释：请结合相邻代码理解此步骤
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # 中文注释：请结合相邻代码理解此步骤
    if "\n" in response:
        print(f"Bot:\n{response}")
    else:
        print(f"Bot: {response}")
