import argparse
import timeit
from contextlib import nullcontext
import math
import gc

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model as model_module
from cs336_basics.model import *
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import pandas as pd


# =========================
# Table 1: model size specs
# =========================
MODEL_SPECS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output


model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def get_data(batch_size: int, context_length: int, device: str):
    data = torch.randint(10000, (batch_size, context_length + 1), dtype=torch.int64, device=device)
    return data[:, :-1], data[:, 1:]


def benchmark(args):
    device = args.device

    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000,
    ).to(device)

    optimizer = AdamW(model.parameters())

    inputs, targets = get_data(args.batch_size, args.context_length, device=device)

    # Warmup
    for _ in range(args.warmup_steps):
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark phase
    timings = []
    if args.mode == "forward":
        with nvtx.range("forward"):
            for i in range(args.steps):
                start_time = timeit.default_timer()
                nvtx.range_push("forward step")
                with torch.autocast(device_type=device, dtype=torch.bfloat16) if args.use_mix else nullcontext():
                    model(inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                nvtx.range_pop()
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)
    else:
        with nvtx.range("forward+backward"):
            for i in range(args.steps):
                start_time = timeit.default_timer()
                with torch.autocast(device_type=device, dtype=torch.bfloat16) if args.use_mix else nullcontext():
                    with nvtx.range("forward step"):
                        logits = model(inputs)
                    with nvtx.range("loss step"):
                        loss = cross_entropy(logits, targets)
                    with nvtx.range("backward step"):
                        loss.backward()
                    with nvtx.range("optimizer.step"):
                        optimizer.step()
                    with nvtx.range("optimizer.zero_grad"):
                        optimizer.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)

    # Report results
    mean_t = float(np.mean(timings))
    std_t = float(np.std(timings))
    total_t = float(sum(timings))

    print(f"\nBenchmark results ({args.mode})")
    print(f"Config: d_model={args.d_model}, d_ff={args.d_ff}, layers={args.num_layers}, heads={args.num_heads}, "
          f"context={args.context_length}, batch={args.batch_size}, mix={args.use_mix}, device={args.device}")
    print(f"time for all steps: {timings}")
    print(f"step time: {mean_t:.6f}, std: {std_t:.6f}")
    print(f"Total time for {args.steps} steps: {total_t:.4f} seconds")

    return mean_t, std_t, total_t


def run_all_sizes(args):
    results = []
    sizes = list(MODEL_SPECS.keys())

    for size in sizes:
        spec = MODEL_SPECS[size]
        # 不改算法，仅切换参数
        args.d_model = spec["d_model"]
        args.d_ff = spec["d_ff"]
        args.num_layers = spec["num_layers"]
        args.num_heads = spec["num_heads"]

        # 清理显存/内存（不影响算法，仅避免累积占用）
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\n==================== Running size: {size} ====================")
        try:
            mean_t, std_t, total_t = benchmark(args)
            results.append((size, mean_t, std_t, total_t))
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] size={size} 发生显存不足，已跳过。你可以尝试减小 --batch_size 或 --context_length。")
            if args.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # 汇总（用 DataFrame.to_markdown()）
    if results:
        rows = []
        for size, mean_t, std_t, total_t in results:
            spec = MODEL_SPECS[size]
            rows.append({
                "size": size,
                "d_model": spec["d_model"],
                "d_ff": spec["d_ff"],
                "num_layers": spec["num_layers"],
                "num_heads": spec["num_heads"],
                "step_mean_s": mean_t,
                "step_std_s": std_t,
                "total_s": total_t,
            })

        df = pd.DataFrame(rows)
        print("\n==================== Summary (Markdown) ====================")
        print(df.to_markdown(index=False, floatfmt=".6f"))
    else:
        print("\nNo successful runs (all OOM or failed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Benchmark")
    parser.add_argument("--d_model", type=int, default=128, help="Size of model")
    parser.add_argument("--d_ff", type=int, default=512, help="Size of feed-forward layer")
    parser.add_argument("--context_length", type=int, default=2048, help="Context length")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warm-up iterations")
    parser.add_argument("--steps", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (used in get_data)")

    parser.add_argument("--mode", choices=["forward", "backward"], default="forward", help="Benchmark mode")
    parser.add_argument("--device", choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    # 更可靠的 bool 参数解析（不影响算法，仅参数解析更标准）
    parser.add_argument("--use_mix", action="store_true", help="Enable autocast bfloat16")

    # 新增：选择表格规格或跑全部
    parser.add_argument("--size", choices=list(MODEL_SPECS.keys()) + ["all"], default=None,
                        help="Run a preset model size from Table 1, or 'all' to run all sizes")

    args = parser.parse_args()

    if args.size is None:
        benchmark(args)
    elif args.size == "all":
        run_all_sizes(args)
    else:
        spec = MODEL_SPECS[args.size]
        args.d_model = spec["d_model"]
        args.d_ff = spec["d_ff"]
        args.num_layers = spec["num_layers"]
        args.num_heads = spec["num_heads"]
        benchmark(args)
