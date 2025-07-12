# Advanced Text Generation Techniques

This guide explains three core methods to enhance neural network-based text generation:

1. **Temperature Sampling**
2. **Top-k / Top-p Sampling**
3. **Context Window (Feeding More History)**

---

## 1. Temperature Sampling 

Temperature controls the **randomness** of predictions by scaling the logits before applying softmax.

### Why It Matters

* **Low temperature (< 1):** More focused and deterministic outputs
* **High temperature (> 1):** More diverse and creative, but potentially nonsensical

### Formula

Given logits:

```
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
```

* When **temperature → 0**, the output becomes greedy (like `argmax`)
* When **temperature > 1**, the output distribution flattens and becomes more random

### Code Example

```python
def sample_with_temperature(probs, temperature=1.0):
    if temperature <= 0:
        return torch.argmax(probs).item()
    logp = torch.log(probs + 1e-9) / temperature
    p = torch.exp(logp)
    p /= p.sum()
    return torch.multinomial(p, 1).item()
```

---

## 2. Top-k / Top-p Sampling 

These are **sampling techniques** to refine the randomness introduced by temperature.

### Top-k Sampling

* Keep only the top `k` tokens with highest probability
* Normalize and sample from them

```python
def top_k_sampling(probs, k=50):
    top_p, top_idx = torch.topk(probs, k)
    top_p /= top_p.sum()
    choice = torch.multinomial(top_p, 1).item()
    return top_idx[choice].item()
```

### Top-p (Nucleus) Sampling

* Select the smallest group of tokens where cumulative probability ≥ `p`
* Normalize and sample from those tokens

```python
def top_p_sampling(probs, p=0.9):
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cum_p = torch.cumsum(sorted_p, dim=0)
    mask = cum_p <= p
    mask[0] = True
    filtered_p = sorted_p[mask] / sorted_p[mask].sum()
    filtered_idx = sorted_idx[mask]
    choice = torch.multinomial(filtered_p, 1).item()
    return filtered_idx[choice].item()
```

### Strategy

* Use **temperature + top-k** to tightly control randomness
* Use **temperature + top-p** to dynamically select likely outputs

---

## 3. Context Window 

Controls how much **history (previous tokens)** the model sees when generating text.

### Default Behavior

```python
input_seq = [[next_char_idx]]
```

* Only uses the last character — can break coherence

### Improved: Feed History

```python
input_seq = torch.cat([input_seq, next_idx], dim=1)
if input_seq.size(1) > max_context:
    input_seq = input_seq[:, -max_context:]
```

* Keeps track of the last N tokens
* Improves semantic consistency

---

##  Complete `generate_text()` Function

```python
def generate_text(start_seq, length=300, temperature=0.8, top_k=None, top_p=None, max_context=100):
    result = start_seq
    input_seq = torch.tensor([[char_to_idx[c] for c in start_seq]], dtype=torch.long)
    hidden = None
    model.eval()

    with torch.no_grad():
        for _ in range(length):
            if input_seq.size(1) > max_context:
                input_seq = input_seq[:, -max_context:]

            x = torch.nn.functional.one_hot(input_seq, num_classes=len(vocab)).float()
            out, hidden = model(x, hidden)

            probs = torch.softmax(out[0], dim=-1)
            idx = sample_with_temperature(probs, temperature)

            if top_k:
                idx = top_k_sampling(probs, top_k)
            if top_p:
                idx = top_p_sampling(probs, top_p)

            next_char = idx_to_char[idx]
            result += next_char
            input_seq = torch.cat([input_seq, torch.tensor([[idx]])], dim=1)
    return result
```

---

##  References

* [Foundations of Top‑k Decoding (arXiv)](https://arxiv.org/abs/2505.19371?utm_source=chatgpt.com)
* [Medium: Top-k and Top-p Sampling](https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832?utm_source=chatgpt.com)
* [Wikipedia: Top-p Sampling](https://en.wikipedia.org/wiki/Top-p_sampling?utm_source=chatgpt.com)
* [Codefinity: Top-k/Top-p Explained](https://codefinity.com/blog/Understanding-Temperature%2C-Top-k%2C-and-Top-p-Sampling-in-Generative-Models?utm_source=chatgpt.com)
* [IBM: What is LLM Temperature?](https://www.ibm.com/think/topics/llm-temperature?utm_source=chatgpt.com)
* [Medium: A Guide to Controlling Output](https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910?utm_source=chatgpt.com)

---

##  Summary Table

| Technique      | Purpose                          | Model Change? | Code Location                          |
| -------------- | -------------------------------- | ------------- | -------------------------------------- |
| Temperature    | Control randomness               |  No          | After softmax, before sampling         |
| Top‑k          | Limit choices to top‑k           |  No          | After temperature step                 |
| Top‑p          | Limit to cumulative probability  |  No          | After temperature/top‑k                |
| Context Window | Feed last N tokens for coherence |  No          | In generation loop (append & truncate) |

---
