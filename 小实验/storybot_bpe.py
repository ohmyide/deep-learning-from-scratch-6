"""用几篇短文体验 StoryBot 的并行 BPE 和磁盘编码。"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from storybot.tokenizer import BPETokenizer, train_bpe


def main():
    eos = "<|endoftext|>"
    text = eos.join(["a cat met a dog. 猫", "a dog met a cat.", "a cat slept."] * 20)
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        source = folder / "stories.txt"
        source.write_text(text, encoding="utf-8")
        target = folder / "stories.bin"

        rules = train_bpe(str(source), vocab_size=265, num_processes=2, num_chunks=4)
        tokenizer = BPETokenizer(rules)
        tokenizer.encode_file(
            str(source), str(target), num_processes=2, num_chunks=4,
            cache_dir=str(folder / "chunks"),
        )
        ids = np.fromfile(target, dtype=np.uint16)
        np.testing.assert_array_equal(ids, tokenizer.encode(text))
        assert tokenizer.decode(ids.tolist()) == text
        print(f"合并规则: {len(rules)}，字节: {len(text.encode())}，tokens: {len(ids)}")
        print("分片文件和整段编码逐 ID 一致")


if __name__ == "__main__":
    main()

