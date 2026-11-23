text = "hello世界😁"
print(list(text))  # 中文注释：：['h', 'e', 'l', 'l', 'o', '', '', '😁']

print(ord('h'))  # 104
print(ord('😁'))  # 128513

print(chr(104))    # 'h'
print(chr(128513)) # '😁'

ids = [ord(char) for char in list(text)]
print(ids)  # [104, 101, 108, 108, 111, 19990, 30028, 128513]

class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ''.join([chr(i) for i in ids])

tokenizer = CharTokenizer()
text = "hello世界😁"

# 中文注释：：ー
ids = tokenizer.encode(text)
print(ids)  # [104, 101, 108, 108, 111, 19990, 30028, 128513]

# 中文注释：：ー
decoded = tokenizer.decode(ids)
print(decoded)  # 中文注释：：hello😁
