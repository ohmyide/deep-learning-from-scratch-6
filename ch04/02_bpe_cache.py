import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from collections import defaultdict
import regex as re
from tqdm import tqdm


def pretokenize(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

def count_pairs(ids, weight=1, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
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

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # 中文注释：请结合相邻代码理解此步骤
    texts = input_text.split(end_token)

    # 中文注释：请结合相邻代码理解此步骤
    pretoken_counts = defaultdict(int)
    for text in tqdm(texts, desc="Pretokenizing"):  # 中文注释：tqdm
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1

    # 中文注释：ID
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)  # 中文注释：请结合相邻代码理解此步骤

    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):  # 中文注释：请结合相邻代码理解此步骤
            pair_to_ids[pair].add(ids)

    for step in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:  # 中文注释：请结合相邻代码理解此步骤
            break

        # 中文注释：请结合相邻代码理解此步骤
        # best_pair = max(pair_counts, key=pair_counts.get)
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # 中文注释：best_pairids
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]  # 中文注释：请结合相邻代码理解此步骤

        # 中文注释：ID
        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)]  # 中文注释：ID
            ids_counts[tuple(new_ids)] = ids_count  # 中文注释：ID

            # 中文注释：请结合相邻代码理解此步骤
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            # 中文注释：请结合相邻代码理解此步骤
            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))

    return merge_rules


vocab_size = 1000  # 中文注释：请结合相邻代码理解此步骤
file_path = "codebot/tiny_codes.txt"
text = open(file_path).read()
merge_rules = train_bpe(text, vocab_size)

# vocab_size = 10000
# file_path = "storybot/tiny_stories_train.txt"
# text = open(file_path).read()
# merge_rules = train_bpe(text, vocab_size)
