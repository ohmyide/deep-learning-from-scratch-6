from collections import defaultdict

def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1

    return merged_ids


class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # 中文注释：ID（0~255）
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        # 中文注释：请结合相邻代码理解此步骤
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        # 中文注释：请结合相邻代码理解此步骤
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        # 中文注释：ー
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids

    def decode(self, ids):
        # 中文注释：ID
        byte_list = [self.id_to_bytes[i] for i in ids]

        # 中文注释：请结合相邻代码理解此步骤
        combined_bytes = b"".join(byte_list)

        # 中文注释：UTF-8
        text = combined_bytes.decode("utf-8", errors="replace")
        return text

# 中文注释：ー
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

# 中文注释：ーー
tokenizer = BPETokenizer(merge_rules)

# 中文注释：请结合相邻代码理解此步骤
text = "Hello世界😁"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)  # [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]
print(decoded)  # 中文注释：Hello😁
