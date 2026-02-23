import timeit
import pandas as pd
import numpy as np
import torch
import logging
import torch.nn as nn
import itertools
from cs336_basics.model import scaled_dot_product_attention

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V):
        seq_len = Q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0)
        return scaled_dot_product_attention(Q, K, V, mask)

def measure(d_model, seq_len, warmup_steps):
    B = 8
    S, D = seq_len, d_model
    Q, K, V = [torch.randn(B, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3)]
    f_time = []
    b_time = []
    f_mem = []
    b_mem = []
    attn = ScaledDotProductAttention()
    attn = torch.compile(attn)
    for i in range(warmup_steps + 100):
        start_time = timeit.default_timer()
        out = attn(Q, K, V)
        torch.cuda.synchronize()
        mid_time = timeit.default_timer()
        mid_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        end_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        if i >= warmup_steps:
            f_time.append(mid_time - start_time)
            b_time.append(end_time - mid_time)
            f_mem.append(mid_mem)
            b_mem.append(end_mem)
        torch.cuda.reset_peak_memory_stats(device)
    return np.mean(f_time) * 1000, np.mean(b_time) * 1000, np.mean(f_mem), np.mean(b_mem)

def benchmark():
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    warmup_steps = 2
    rows = []
    for d_model, seq_len in itertools.product(d_models, seq_lens):
        print(f"Running benchmark: d_model {d_model} seq_len {seq_len}")
        try:
            f_time, b_time, f_mem, b_mem = measure(d_model, seq_len, warmup_steps)
            rows.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'f_time': f_time,
                'b_time': b_time,
                'f_memory': f_mem,
                'b_memory': b_mem
            })
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM for d_model={d_model}, seq_len={seq_len}")
                torch.cuda.empty_cache()
                rows.append({
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'f_time': None,
                    'b_time': None,
                    'f_memory': None,
                    'b_memory': None
                })
            else:
                raise
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False, floatfmt=[".0f", ".0f", ".2f", ".2f", ".2f", ".2f"]))

if __name__ == "__main__":
    benchmark()