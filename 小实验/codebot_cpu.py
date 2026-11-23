"""用官方 CodeBot 分词器和 GPT，在 CPU 上走完训练和生成。"""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate


def next_token_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


def train_demo(tokenizer, steps):
    sample = "def add(a, b):\n    return a + b\n"
    ids = torch.tensor(tokenizer.encode(sample) + [tokenizer.end_token_id])
    x, y = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
    model = GPT(tokenizer.vocab_size, 64, 48, 4, 2, 96, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    before = next_token_loss(model, x, y).item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = next_token_loss(model, x, y)
        loss.backward()
        optimizer.step()
    after = next_token_loss(model, x, y).item()
    print(f"训练序列损失: {before:.4f} -> {after:.4f}")

    with TemporaryDirectory() as folder:
        path = Path(folder) / "demo.pt"
        model.save(path)
        restored = GPT.load_from(path).eval()
        with torch.inference_mode():
            torch.testing.assert_close(model.eval()(x), restored(x))
        print("保存后重新加载，输出一致")
        print(generate(restored, tokenizer, "def add", max_new_tokens=32, temperature=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80, help="小实验的训练步数")
    parser.add_argument("--checkpoint", type=Path, help="用已有权重进行本地生成")
    parser.add_argument("--prompt", default="def add(a, b):\n", help="已有权重的生成前缀")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = BPETokenizer.load_from(ROOT / "codebot/merge_rules.pkl")
    if args.checkpoint is None:
        train_demo(tokenizer, args.steps)
    else:
        model = GPT.load_from(args.checkpoint).eval()
        if model.vocab_size != tokenizer.vocab_size:
            parser.error("checkpoint vocab_size does not match the CodeBot tokenizer")
        print(generate(model, tokenizer, args.prompt, max_new_tokens=64, temperature=0))


if __name__ == "__main__":
    main()

