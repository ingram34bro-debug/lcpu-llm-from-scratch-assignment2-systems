import pandas as pd
import itertools
from flashattn2_torch import FlashAttnWithTorch
from flashattn2_triton import FlashAttnWithTriton
import torch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 128
import triton
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def make_attn_inputs(batch_size, Nq, Nk, D, dtype=torch.bfloat16, device="cuda"):
    torch.random.manual_seed(0)
    q = torch.randn(batch_size, Nq, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, Nk, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, Nk, D, dtype=dtype, device=device, requires_grad=True)
    do = torch.randn(batch_size, Nq, D, dtype=dtype, device=device)

    return q, k, v, do
torch_flash_compiled = torch.compile(FlashAttnWithTorch.apply, dynamic=True)

def test_timing_flash_forward_backward(Nq, Nk, D, dtype=torch.bfloat16, is_triton=False, device="cuda"):
    q, k, v, do = make_attn_inputs(1, Nq, Nk, D, dtype=dtype, device=device)
    if is_triton:
        flash = FlashAttnWithTriton.apply
    else:
        flash = torch_flash_compiled
    def run_fwd():
        flash(q, k, v, True)
    def run_fwd_bwd():
        o = flash(q, k, v, True)
        o.backward(do)
    run_fwd()
    f_time = triton.testing.do_bench(lambda: run_fwd(), rep=10, warmup=5)
    fb_time = triton.testing.do_bench(lambda: run_fwd_bwd(), grad_to_none=[q, k, v], rep=10, warmup=5)
    return f_time, fb_time

def benchmark():
    d_models = [16, 32, 64, 128]
    seq_lens = [128, 256, 512, 1024, 2048]
    dtypes = [torch.bfloat16, torch.float32]
    rows = []
    configs = list(itertools.product(d_models, seq_lens, dtypes))
    total = len(configs)
    for i, (d_model, seq_len, dtype) in enumerate(configs):
        logger.info(f"[{i + 1}/{total}] Running: d_model={d_model}, seq_len={seq_len}, dtype={dtype}")
        try:
            torch_f_time, torch_fb_time = test_timing_flash_forward_backward(seq_len, seq_len, d_model, dtype)
            triton_f_time, triton_fb_time = test_timing_flash_forward_backward(seq_len, seq_len, d_model, dtype, is_triton=True)
            rows.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'dtype': str(dtype).split('.')[-1],
                'torch_f_time': torch_f_time,
                'torch_fb_time': torch_fb_time,
                'triton_f_time': triton_f_time,
                'triton_fb_time': triton_fb_time,
            })
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM for d_model={d_model}, seq_len={seq_len}, dtype={dtype}")
                torch.cuda.empty_cache()
                rows.append({
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'dtype': str(dtype).split('.')[-1],
                    'torch_f_time': None,
                    'torch_fb_time': None,
                    'triton_f_time': None,
                    'triton_fb_time': None,
                })
            else:
                raise
    df = pd.DataFrame(rows)
    float_fmt = [".0f", ".0f", "", ".2f", ".2f", ".2f", ".2f"]
    md_output = df.to_markdown(index=False, floatfmt=float_fmt)
    print("\nBenchmark Results:")
    print(md_output)
    output_file = "reports/torch_vs_triton.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Flash Attention Benchmark Results\n\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n\n")
        f.write(md_output)

    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    benchmark()