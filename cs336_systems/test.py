import os
import itertools
import logging
import pandas as pd

# 让 torch._logging.set_logs 能生效（如果你 shell 里设置过 TORCH_LOGS，会导致 set_logs 失效）
os.environ.pop("TORCH_LOGS", None)
# 兜底：防止 torch.compile / Dynamo 在某些路径里意外介入
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch._dynamo
import torch._logging
import triton

from flashattn2_torch import FlashAttnWithTorch
from flashattn2_triton import FlashAttnWithTriton

# -------------------------
# Benchmark knobs (absolute time oriented)
# -------------------------
DEVICE = "cuda"
BENCH_WARMUP = 5
BENCH_REP = 20

# -------------------------
# Torch configs (keep your original intent)
# -------------------------
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 128

# -------------------------
# Logging (keep your style, but quieter)
# -------------------------
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# 避免在 notebook / 多次运行时重复 addHandler
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 关键：压掉你看到的 torch/utils/_sympy/interp.py 那类 warning
# 以及 Dynamo/Inductor 的噪声日志
try:
    torch._logging.set_logs(all=logging.ERROR)
except Exception:
    pass

for name in [
    "torch.utils._sympy.interp",
    "torch._dynamo",
    "torch._inductor",
]:
    logging.getLogger(name).setLevel(logging.ERROR)


def cuda_sync():
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_attn_inputs(batch_size, Nq, Nk, D, dtype=torch.bfloat16, device=DEVICE):
    q = torch.randn(batch_size, Nq, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, Nk, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, Nk, D, dtype=dtype, device=device, requires_grad=True)
    do = torch.randn(batch_size, Nq, D, dtype=dtype, device=device)
    return q, k, v, do


# 明确禁用 Dynamo，避免走符号形状推导那条链路
@torch._dynamo.disable
def run_fwd(flash, q, k, v):
    with torch.no_grad():
        flash(q, k, v, True)


@torch._dynamo.disable
def run_fwd_bwd(flash, q, k, v, do):
    o = flash(q, k, v, True)
    o.backward(do)


def compile_once(flash, q, k, v, do):
    """
    编译/预热阶段：只负责触发 JIT/Autotune/初始化等开销，不计入 timing。
    同时分别触发：
      - inference/no_grad 路径
      - training + backward 路径
    """
    run_fwd(flash, q, k, v)
    q.grad = k.grad = v.grad = None

    run_fwd_bwd(flash, q, k, v, do)
    q.grad = k.grad = v.grad = None


def bench_one_backend(flash, q, k, v, do):
    """
    纯测量阶段：假设 compile_once 已经跑过，do_bench 不再吃到编译时间。
    """
    f_time = triton.testing.do_bench(
        lambda: run_fwd(flash, q, k, v),
        warmup=BENCH_WARMUP,
        rep=BENCH_REP,
    )
    fb_time = triton.testing.do_bench(
        lambda: run_fwd_bwd(flash, q, k, v, do),
        grad_to_none=[q, k, v],
        warmup=BENCH_WARMUP,
        rep=BENCH_REP,
    )
    q.grad = k.grad = v.grad = None
    return f_time, fb_time


def benchmark():
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' but torch.cuda.is_available() is False")

    # 一次性初始化 CUDA context，避免第一个 config 被“冷启动”拖爆
    if DEVICE.startswith("cuda"):
        torch.empty(1, device=DEVICE)
        cuda_sync()

    # 和你原来保持一致
    d_models = [16, 32, 64, 128]
    seq_lens = [128, 256, 512, 1024, 2048,4096,8192]#,16384,32768,65536]
    dtypes = [torch.bfloat16,torch.float32]

    configs = list(itertools.product(d_models, seq_lens, dtypes))
    total = len(configs)

    # 固定随机种子（只设一次，不要每次 make_attn_inputs 都 reset）
    torch.manual_seed(0)

    # 预先创建输入：避免测量阶段夹杂大量 allocation / randn 的干扰
    # 也确保 torch / triton 后端用的是同一份输入
    inputs = {}
    for d_model, seq_len, dtype in configs:
        q, k, v, do = make_attn_inputs(1, seq_len, seq_len, d_model, dtype=dtype, device=DEVICE)
        inputs[(d_model, seq_len, dtype)] = (q, k, v, do)

    # -------------------------
    # Phase 1: compile / warmup (excluded from timing)
    # -------------------------
    logger.info("Phase 1/2: compile & warm up kernels (excluded from timing)...")
    for i, (d_model, seq_len, dtype) in enumerate(configs):
        logger.info(f"[compile {i + 1}/{total}] d_model={d_model}, seq_len={seq_len}, dtype={dtype}")
        q, k, v, do = inputs[(d_model, seq_len, dtype)]
        try:
            compile_once(FlashAttnWithTorch.apply, q, k, v, do)
            compile_once(FlashAttnWithTriton.apply, q, k, v, do)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM during compile for d_model={d_model}, seq_len={seq_len}, dtype={dtype}")
                torch.cuda.empty_cache()
            else:
                raise

    cuda_sync()

    # -------------------------
    # Phase 2: benchmark (timing only)
    # -------------------------
    logger.info("Phase 2/2: benchmarking (timing only)...")
    rows = []
    for i, (d_model, seq_len, dtype) in enumerate(configs):
        logger.info(f"[bench {i + 1}/{total}] d_model={d_model}, seq_len={seq_len}, dtype={dtype}")
        q, k, v, do = inputs[(d_model, seq_len, dtype)]
        try:
            torch_f_time, torch_fb_time = bench_one_backend(FlashAttnWithTorch.apply, q, k, v, do)
            triton_f_time, triton_fb_time = bench_one_backend(FlashAttnWithTriton.apply, q, k, v, do)

            rows.append(
                {
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "dtype": str(dtype).split(".")[-1],
                    "torch_f_time": torch_f_time,
                    "torch_fb_time": torch_fb_time,
                    "triton_f_time": triton_f_time,
                    "triton_fb_time": triton_fb_time,
                }
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM during bench for d_model={d_model}, seq_len={seq_len}, dtype={dtype}")
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "dtype": str(dtype).split(".")[-1],
                        "torch_f_time": None,
                        "torch_fb_time": None,
                        "triton_f_time": None,
                        "triton_fb_time": None,
                    }
                )
            else:
                raise

    # -------------------------
    # Output (keep md unchanged)
    # -------------------------
    df = pd.DataFrame(rows)
    float_fmt = [".0f", ".0f", "", ".2f", ".2f", ".2f", ".2f"]
    md_output = df.to_markdown(index=False, floatfmt=float_fmt)

    print("\nBenchmark Results:")
    print(md_output)

    os.makedirs("reports", exist_ok=True)
    output_file = "reports/torch_vs_triton_16.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Flash Attention Benchmark Results\n\n")
        if DEVICE.startswith("cuda"):
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n\n")
        else:
            f.write("Device: cpu\n\n")
        f.write(md_output)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    benchmark()
