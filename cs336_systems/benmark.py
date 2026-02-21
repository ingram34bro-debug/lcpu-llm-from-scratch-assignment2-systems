import argparse
import timeit
from contextlib import nullcontext
import math
import numpy as np
import torch

# NVTX: 在无 CUDA 时做兼容
if torch.cuda.is_available():
    import torch.cuda.nvtx as nvtx
else:
    class _DummyNVTX:
        def range(self, *_args, **_kwargs):
            return nullcontext()
        def range_push(self, *_args, **_kwargs):
            pass
        def range_pop(self, *_args, **_kwargs):
            pass
    nvtx = _DummyNVTX()

import cs336_basics.TransformerLM as model_module
from cs336_basics.TransformerLM import *
from cs336_basics.cross_entropy_loss import cross_entropy_loss
from cs336_basics.optimizer import AdamW


# ====== 你的 attention 标注保持不变 ======
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
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output

model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention


# ====== 表格配置：按你图里的五行 ======
PRESETS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def get_data(batch_size: int, context_length: int, device: str):
    data = torch.randint(10000, (batch_size, context_length + 1), dtype=torch.int64, device=device)
    return data[:, :-1], data[:, 1:]


def run_benchmark_once(
    *,
    device: str,
    context_length: int,
    batch_size: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    warmup_steps: int,
    steps: int,
    mode: str,
    use_mix: bool,
):
    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000,
    ).to(device)

    optimizer = AdamW(model.parameters())
    inputs, targets = get_data(batch_size, context_length, device=device)

    # warmup
    model.train()
    for _ in range(warmup_steps):
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()

    timings = []

    if mode == "forward":
        model.eval()
        with nvtx.range("forward"):
            for _ in range(steps):
                start_time = timeit.default_timer()
                nvtx.range_push("forward step")

                autocast_ctx = (
                    torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if use_mix and device in ("cuda", "cpu")
                    else nullcontext()
                )
                with autocast_ctx:
                    with torch.inference_mode():
                        model(inputs)

                if device == "cuda":
                    torch.cuda.synchronize()
                nvtx.range_pop()
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)

    else:
        model.train()
        with nvtx.range("forward+backward"):
            for _ in range(steps):
                start_time = timeit.default_timer()

                autocast_ctx = (
                    torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if use_mix and device in ("cuda", "cpu")
                    else nullcontext()
                )

                with autocast_ctx:
                    with nvtx.range("forward step"):
                        logits = model(inputs)
                    with nvtx.range("loss step"):
                        loss = cross_entropy_loss(logits, targets)
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

    mean_t = float(np.mean(timings))
    std_t = float(np.std(timings))
    total_t = float(np.sum(timings))
    return mean_t, std_t, total_t


def main():
    parser = argparse.ArgumentParser(description="PyTorch Model Benchmark (with table presets)")

    # 仍支持手动传参
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)

    # 表格相关
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                        help="Run a single preset from the table")
    parser.add_argument("--all_presets", action="store_true",
                        help="Run all presets from the table")

    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "backward"], default="forward")

    parser.add_argument("--device", choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_mix", action="store_true",
                        help="Enable autocast(bfloat16)")

    args = parser.parse_args()

    # 要跑哪些配置
    configs = []
    if args.all_presets:
        for name, cfg in PRESETS.items():
            configs.append((name, cfg))
    elif args.preset is not None:
        configs.append((args.preset, PRESETS[args.preset]))
    else:
        # 没选 preset 就用命令行手动那组
        configs.append(("custom", {
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }))

    print(f"device={args.device}, mode={args.mode}, ctx={args.context_length}, bs={args.batch_size}, mix={args.use_mix}")
    print("-" * 90)
    print(f"{'name':<8} {'d_model':>7} {'d_ff':>7} {'layers':>7} {'heads':>7} | {'mean(s)':>10} {'std(s)':>10} {'total(s)':>10}")
    print("-" * 90)

    for name, cfg in configs:
        mean_t, std_t, total_t = run_benchmark_once(
            device=args.device,
            context_length=args.context_length,
            batch_size=args.batch_size,
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            mode=args.mode,
            use_mix=args.use_mix,
        )
        print(f"{name:<8} {cfg['d_model']:>7} {cfg['d_ff']:>7} {cfg['num_layers']:>7} {cfg['num_heads']:>7} | {mean_t:>10.6f} {std_t:>10.6f} {total_t:>10.4f}")

    print("-" * 90)


if __name__ == "__main__":
    main()
