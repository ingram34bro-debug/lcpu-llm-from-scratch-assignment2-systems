from __future__ import annotations

import timeit
import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml
from pathlib import Path
from torch import Tensor
import numpy as np
import typing
from typing import Optional
from jaxtyping import Bool, Float, Int
import argparse
from cs336_basics.cross_entropy_loss import *
from cs336_basics.optimizer import *
from cs336_basics.get_batch import *
from cs336_basics.TransformerLM import TransformerLM
import torch.cuda.nvtx as nvtx
import os
from contextlib import nullcontext
def get_data(batch_size: int, context_length: int, device: str):
    data = torch.randint(10000, (batch_size, context_length+1), dtype=torch.int64, device=device)
    return data[:, :-1], data[:, 1:]
def benchmark_model(args) -> Tuple[float, float]:
    device = args.device
    model=TransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000,
    ).to(device)
    optimizer=AdamW(model.parameters(), lr=1e-3)
    batch_size = 4
    # Create random input and target tensors
    inputs,targets=get_data(batch_size, args.context_length, device)
    for _ in range(args.warmup_steps):
        with torch.autocast(device_type=device, dtype=torch.bfloat16) if args.use_mix else nullcontext():
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    # Benchmark phase
    timings = []
   
    if args.mode == "forward":
        with nvtx.range("forward"):
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.memory._record_memory_history(max_entries=1000000)
            for i in range(args.steps):
                start_time = timeit.default_timer()
                nvtx.range_push("forward step")
                with torch.autocast(device_type=device, dtype=torch.bfloat16) if args.use_mix else nullcontext():
                    model(inputs)
                torch.cuda.synchronize()
                nvtx.range_pop()
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)
            # torch.cuda.memory._dump_snapshot("memory_snapshot_forward.pickle")
            # torch.cuda.memory._record_memory_history(enabled=None)
    else:
        with nvtx.range("forward+backward"):
            time_cnt = 0
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.memory._record_memory_history(enabled=None)
            for i in range(args.steps):
                start_time = timeit.default_timer()
                with torch.autocast(device_type=device, dtype=torch.bfloat16) if args.use_mix else nullcontext():
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
                torch.cuda.synchronize()
                end_time = timeit.default_timer()
                # if time_cnt == 0:
                #     torch.cuda.memory._dump_snapshot("memory_snapshot_train_step.pickle")
                #     torch.cuda.memory._record_memory_history(enabled=None)
                timings.append(end_time - start_time)
                time_cnt += 1
    # Report results
    print(f"\nBenchmark results ({args.mode}):")
    print(f"time for all steps: {timings}")
    print(f"step time: {np.mean(timings):.6f}, std: {np.std(timings):.6f}")
    print(f"Total time for {args.steps} steps: {sum(timings):.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Benchmark")
    parser.add_argument("--d_model", type=int, default=128, help="Size of model")
    parser.add_argument("--d_ff", type=int, default=512, help="Size of feed-forward layer")
    parser.add_argument("--context_length", type=int, default=2048, help="Context length")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warm-up iterations")
    parser.add_argument("--steps", type=int, default=10, help="Benchmark iterations")

    parser.add_argument("--mode", choices=["forward", "backward"], default="forward",
                        help="Benchmark mode")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--use_mix", type=bool, default=False, help="Whether to use mix-precision")
    args = parser.parse_args()
    benchmark_model(args)