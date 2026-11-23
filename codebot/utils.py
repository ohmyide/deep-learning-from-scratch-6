import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 中文注释：ー

    # 中文注释：❶
    device = next(model.parameters()).device  # 中文注释：请结合相邻代码理解此步骤
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 中文注释：请结合相邻代码理解此步骤
    generated_ids = ids.clone()

    # 中文注释：❷ ー
    for _ in range(max_new_tokens):
        # 中文注释：请结合相邻代码理解此步骤
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 中文注释：请结合相邻代码理解此步骤
        logits = model(ids)[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 中文注释：请结合相邻代码理解此步骤
        if next_id.item() == tokenizer.end_token_id:
            break

        # 中文注释：请结合相邻代码理解此步骤
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # 中文注释：❸
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
