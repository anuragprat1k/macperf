#!/usr/bin/env python3
"""
Apple Silicon ML Benchmark

Comprehensive performance profiling for ML workloads on Apple Silicon Macs.
Benchmarks CPU and GPU (MPS) with practical ML-focused metrics including
transformer throughput, model feasibility, and training time estimates.

Usage:
    python bench.py              # Full benchmark suite
    python bench.py --quick      # Skip slower benchmarks
    python bench.py --submit     # Auto-create a PR with your results
    python bench.py --share      # Print compact shareable summary
    python bench.py --no-color   # Disable colored output

Results are printed to terminal and saved to results/ as JSON.
"""

import argparse
import gc
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn as nn

REPO_URL = "https://github.com/anuragprat1k/macperf"


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

class _C:
    """ANSI color helpers — auto-disabled when stdout is not a tty."""

    def __init__(self):
        self.on = sys.stdout.isatty()

    def _w(self, code, t):
        return f"\033[{code}m{t}\033[0m" if self.on else str(t)

    def bold(self, t):   return self._w("1", t)
    def dim(self, t):    return self._w("2", t)
    def red(self, t):    return self._w("31", t)
    def green(self, t):  return self._w("32", t)
    def yellow(self, t): return self._w("33", t)
    def blue(self, t):   return self._w("34", t)
    def cyan(self, t):   return self._w("36", t)

c = _C()


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _system_profiler_gpu() -> dict:
    info: dict = {"gpu_name": "", "gpu_cores": "", "metal_family": ""}
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, stderr=subprocess.DEVNULL
        )
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("Chipset Model:"):
                info["gpu_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Total Number of Cores:"):
                info["gpu_cores"] = line.split(":", 1)[1].strip()
            elif line.startswith("Metal Family:") or line.startswith("Metal Support:"):
                info["metal_family"] = line.split(":", 1)[1].strip()
    except subprocess.CalledProcessError:
        pass
    return info


def collect_system_info() -> dict:
    gpu = _system_profiler_gpu()
    chip = _sysctl("machdep.cpu.brand_string")
    perf_cores = _sysctl("hw.perflevel0.logicalcpu")
    eff_cores = _sysctl("hw.perflevel1.logicalcpu")
    total_cores = _sysctl("hw.logicalcpu")
    ram_bytes = _sysctl("hw.memsize")
    ram_gb = int(ram_bytes) / (1024 ** 3) if ram_bytes else 0
    mps_available = torch.backends.mps.is_available()

    return {
        "hostname": socket.gethostname(),
        "chip": chip,
        "cpu_cores_total": total_cores,
        "cpu_cores_performance": perf_cores,
        "cpu_cores_efficiency": eff_cores,
        "gpu_name": gpu["gpu_name"],
        "gpu_cores": gpu["gpu_cores"],
        "metal_family": gpu["metal_family"],
        "ram_gb": round(ram_gb, 1),
        "os_version": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "mps_available": mps_available,
    }


def print_system_info(si):
    """Print compact system summary."""
    mps_tag = c.green("MPS \u2713") if si["mps_available"] else c.red("MPS \u2717")
    os_ver = si["os_version"]
    # "macOS-14.2.1-arm64-arm-64bit" -> "macOS 14.2.1"
    parts = os_ver.split("-")
    os_short = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else os_ver
    print(f"  {c.bold(si['chip'])} | {int(si['ram_gb'])} GB | {os_short}")
    print(f"  {si['cpu_cores_total']} CPU cores "
          f"({si['cpu_cores_performance']}P + {si['cpu_cores_efficiency']}E) | "
          f"{si['gpu_cores']} GPU cores | {si['metal_family']}")
    print(f"  Python {si['python_version']} | PyTorch {si['torch_version']} | {mps_tag}")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _mps_mem_mb() -> float | None:
    try:
        return round(torch.mps.current_allocated_memory() / 1024 ** 2, 1)
    except Exception:
        return None


def _mps_reset():
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def bench(fn, *, warmup=3, iters=7, device="cpu") -> float:
    """Run *fn* with warmup, return median wall-clock seconds."""
    mps = device == "mps"
    for _ in range(warmup):
        fn()
        if mps:
            _sync()
    times = []
    for _ in range(iters):
        if mps:
            _sync()
        t0 = time.perf_counter()
        fn()
        if mps:
            _sync()
        times.append(time.perf_counter() - t0)
    return median(times)


def _gflops(n, t):
    return 2 * n ** 3 / t / 1e9


def _section(title):
    print(f"\n{c.bold(c.blue(title))}")


def _sub(title):
    print(f"  {c.bold(title)}")


# ---------------------------------------------------------------------------
# Low-level CPU benchmarks
# ---------------------------------------------------------------------------

def bench_cpu_gemm(sizes):
    results = []
    for n in sizes:
        a, b = torch.randn(n, n), torch.randn(n, n)
        t = bench(lambda: torch.mm(a, b))
        gf = _gflops(n, t)
        results.append({"size": n, "seconds": round(t, 4), "gflops": round(gf, 2)})
        print(f"    {n:>5}x{n:<5}  {t:8.4f}s  {c.cyan(f'{gf:8.2f}')} GFLOPS")
    return results


def bench_cpu_batch_matmul(batch=64, seq=512, dim=64):
    a, b = torch.randn(batch, seq, dim), torch.randn(batch, dim, seq)
    t = bench(lambda: torch.bmm(a, b))
    gf = 2 * batch * seq * dim * seq / t / 1e9
    print(f"    Batch MM (B={batch} S={seq} D={dim})  {t:.4f}s  {c.cyan(f'{gf:.2f}')} GFLOPS")
    return {"batch": batch, "seq": seq, "dim": dim, "seconds": round(t, 4), "gflops": round(gf, 2)}


def bench_cpu_membw(size_mb=256):
    n = size_mb * 1024 * 1024 // 4
    src = np.random.randn(n).astype(np.float32)
    t = bench(lambda: np.copy(src), warmup=2, iters=5)
    bw = size_mb / t / 1024
    print(f"    Mem BW ({size_mb} MB copy)  {t:.4f}s  {c.cyan(f'{bw:.2f}')} GB/s")
    return {"size_mb": size_mb, "seconds": round(t, 4), "gb_per_s": round(bw, 2)}


# ---------------------------------------------------------------------------
# Low-level GPU benchmarks
# ---------------------------------------------------------------------------

def bench_gpu_gemm(sizes):
    dev = torch.device("mps")
    results = []
    for n in sizes:
        _mps_reset()
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        t = bench(lambda: torch.mm(a, b), device="mps")
        gf = _gflops(n, t)
        mem = _mps_mem_mb()
        results.append({"size": n, "seconds": round(t, 4), "gflops": round(gf, 2), "mem_mb": mem})
        print(f"    {n:>5}x{n:<5}  {t:8.4f}s  {c.cyan(f'{gf:8.2f}')} GFLOPS  {c.dim(f'mem:{mem}MB')}")
        del a, b
    return results


def bench_gpu_batch_matmul(batch=64, seq=512, dim=64):
    dev = torch.device("mps")
    _mps_reset()
    a = torch.randn(batch, seq, dim, device=dev)
    b = torch.randn(batch, dim, seq, device=dev)
    t = bench(lambda: torch.bmm(a, b), device="mps")
    gf = 2 * batch * seq * dim * seq / t / 1e9
    mem = _mps_mem_mb()
    print(f"    Batch MM (B={batch} S={seq} D={dim})  {t:.4f}s  {c.cyan(f'{gf:.2f}')} GFLOPS  {c.dim(f'mem:{mem}MB')}")
    del a, b
    return {"batch": batch, "seq": seq, "dim": dim, "seconds": round(t, 4), "gflops": round(gf, 2), "mem_mb": mem}


def bench_gpu_conv2d():
    dev = torch.device("mps")
    _mps_reset()
    conv = nn.Conv2d(64, 128, 3, padding=1).to(dev)
    x = torch.randn(32, 64, 56, 56, device=dev)
    t = bench(lambda: conv(x), device="mps")
    mem = _mps_mem_mb()
    print(f"    Conv2D (32x64x56x56 -> 128ch k=3)  {c.cyan(f'{t:.4f}s')}  {c.dim(f'mem:{mem}MB')}")
    del conv, x
    return {"batch": 32, "in_ch": 64, "out_ch": 128, "spatial": 56, "kernel": 3,
            "seconds": round(t, 4), "mem_mb": mem}


def bench_gpu_transfer():
    size_mb = 128
    n = size_mb * 1024 * 1024 // 4
    dev = torch.device("mps")
    cpu_t = torch.randn(n)

    t_h2d = bench(lambda: cpu_t.to(dev), device="mps")
    bw_h2d = size_mb / t_h2d / 1024
    print(f"    CPU\u2192GPU ({size_mb}MB)  {t_h2d:.4f}s  {c.cyan(f'{bw_h2d:.2f}')} GB/s")

    gpu_t = cpu_t.to(dev)
    _sync()
    t_d2h = bench(lambda: gpu_t.cpu(), device="mps")
    bw_d2h = size_mb / t_d2h / 1024
    print(f"    GPU\u2192CPU ({size_mb}MB)  {t_d2h:.4f}s  {c.cyan(f'{bw_d2h:.2f}')} GB/s")

    del cpu_t, gpu_t
    return {
        "size_mb": size_mb,
        "h2d_seconds": round(t_h2d, 4), "h2d_gb_per_s": round(bw_h2d, 2),
        "d2h_seconds": round(t_d2h, 4), "d2h_gb_per_s": round(bw_d2h, 2),
    }


def bench_gpu_fp16_fp32():
    dev = torch.device("mps")
    n = 2048
    _mps_reset()
    a32, b32 = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
    t32 = bench(lambda: torch.mm(a32, b32), device="mps")
    gf32 = _gflops(n, t32)

    a16, b16 = a32.half(), b32.half()
    t16 = bench(lambda: torch.mm(a16, b16), device="mps")
    gf16 = _gflops(n, t16)

    speedup = t32 / t16 if t16 > 0 else 0
    print(f"    FP32  {n}x{n}  {t32:.4f}s  {c.cyan(f'{gf32:.2f}')} GFLOPS")
    print(f"    FP16  {n}x{n}  {t16:.4f}s  {c.cyan(f'{gf16:.2f}')} GFLOPS")
    print(f"    FP16 speedup: {c.green(f'{speedup:.2f}x')}")
    del a32, b32, a16, b16
    return {
        "size": n,
        "fp32_seconds": round(t32, 4), "fp32_gflops": round(gf32, 2),
        "fp16_seconds": round(t16, 4), "fp16_gflops": round(gf16, 2),
        "fp16_speedup": round(speedup, 2),
    }


# ---------------------------------------------------------------------------
# Transformer benchmarks
# ---------------------------------------------------------------------------

def _make_block(d_model, nhead, device):
    return nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
        dropout=0.0, batch_first=True,
    ).to(device)


def bench_transformer_throughput(configs, device_name):
    """Benchmark transformer block fwd+bwd at various dims."""
    dev = torch.device(device_name)
    results = []
    for d_model, nhead, batch, seq in configs:
        _mps_reset()
        try:
            block = _make_block(d_model, nhead, dev)
            x = torch.randn(batch, seq, d_model, device=dev, requires_grad=True)

            def train_step():
                block.zero_grad()
                out = block(x)
                out.sum().backward()

            t_train = bench(train_step, warmup=2, iters=5, device=device_name)
            tok_train = batch * seq / t_train
            mem_train = _mps_mem_mb()

            block.eval()
            with torch.no_grad():
                t_infer = bench(lambda: block(x), warmup=2, iters=5, device=device_name)
            tok_infer = batch * seq / t_infer
            block.train()

            r = {
                "d_model": d_model, "nhead": nhead, "batch": batch, "seq": seq,
                "train_sec": round(t_train, 4), "train_tok_s": round(tok_train),
                "infer_sec": round(t_infer, 4), "infer_tok_s": round(tok_infer),
                "mem_mb": mem_train,
            }
            results.append(r)
            print(f"    dim={d_model:>4} heads={nhead:>2} batch={batch:>2} seq={seq:>4}  "
                  f"train:{c.cyan(f'{tok_train:>8.0f}')} tok/s  "
                  f"infer:{c.cyan(f'{tok_infer:>8.0f}')} tok/s  "
                  f"{c.dim(f'mem:{mem_train}MB')}")
            del block, x
            _mps_reset()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"    dim={d_model:>4} heads={nhead:>2} batch={batch:>2} seq={seq:>4}  "
                      f"{c.red('SKIPPED (OOM)')}")
                results.append({"d_model": d_model, "nhead": nhead, "batch": batch, "seq": seq, "error": "OOM"})
                _mps_reset()
            else:
                raise
    return results


def bench_seq_scaling(d_model=1024, nhead=8, batch=8, device_name="mps"):
    """Show how throughput degrades with sequence length (O(n^2) attention)."""
    dev = torch.device(device_name)
    results = []
    for seq in [128, 256, 512, 1024, 2048]:
        _mps_reset()
        try:
            block = _make_block(d_model, nhead, dev)
            x = torch.randn(batch, seq, d_model, device=dev, requires_grad=True)

            def step():
                block.zero_grad()
                block(x).sum().backward()

            t = bench(step, warmup=2, iters=5, device=device_name)
            tok = batch * seq / t
            mem = _mps_mem_mb()
            results.append({"seq": seq, "seconds": round(t, 4), "tok_s": round(tok), "mem_mb": mem})
            print(f"    seq={seq:>5}  {t:.4f}s  {c.cyan(f'{tok:>8.0f}')} tok/s  {c.dim(f'mem:{mem}MB')}")
            del block, x
            _mps_reset()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"    seq={seq:>5}  {c.red('SKIPPED (OOM)')}")
                results.append({"seq": seq, "error": "OOM"})
                _mps_reset()
            else:
                raise
    return results


def bench_lora(d_model=1024, nhead=8, batch=8, seq=512, rank=16, device_name="mps"):
    """Compare LoRA fine-tuning vs full fine-tuning throughput."""
    dev = torch.device(device_name)
    _mps_reset()
    results: dict = {}

    try:
        # --- full fine-tune baseline ---
        block = _make_block(d_model, nhead, dev)
        x = torch.randn(batch, seq, d_model, device=dev, requires_grad=True)

        def full_step():
            block.zero_grad()
            block(x).sum().backward()

        t_full = bench(full_step, warmup=2, iters=5, device=device_name)
        tok_full = batch * seq / t_full
        mem_full = _mps_mem_mb()
        full_params = sum(p.numel() for p in block.parameters())
        del block, x
        _mps_reset()

        # --- LoRA fine-tune ---
        block = _make_block(d_model, nhead, dev)
        for p in block.parameters():
            p.requires_grad_(False)

        lora_a = nn.ParameterDict({
            k: nn.Parameter(torch.randn(d_model, rank, device=dev) * 0.01)
            for k in ["q", "k", "v", "out"]
        })
        lora_b = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(rank, d_model, device=dev))
            for k in ["q", "k", "v", "out"]
        })
        lora_params = sum(p.numel() for p in lora_a.values()) + sum(p.numel() for p in lora_b.values())

        x = torch.randn(batch, seq, d_model, device=dev)

        def lora_step():
            for p in lora_a.values():
                if p.grad is not None:
                    p.grad.zero_()
            for p in lora_b.values():
                if p.grad is not None:
                    p.grad.zero_()
            out = block(x)
            lora_out = x @ lora_a["out"] @ lora_b["out"]
            (out + lora_out).sum().backward()

        t_lora = bench(lora_step, warmup=2, iters=5, device=device_name)
        tok_lora = batch * seq / t_lora
        mem_lora = _mps_mem_mb()

        speedup = t_full / t_lora if t_lora > 0 else 0
        mem_save = (1 - mem_lora / mem_full) * 100 if mem_full and mem_lora and mem_full > 0 else 0

        print(f"    Full FT:       {c.cyan(f'{tok_full:>8.0f}')} tok/s  "
              f"{c.dim(f'mem:{mem_full}MB')}  params:{full_params:,}")
        print(f"    LoRA (r={rank}):   {c.cyan(f'{tok_lora:>8.0f}')} tok/s  "
              f"{c.dim(f'mem:{mem_lora}MB')}  params:{lora_params:,}")
        print(f"    LoRA speedup: {c.green(f'{speedup:.2f}x')}   "
              f"memory saving: {c.green(f'{mem_save:.0f}%')}")

        results = {
            "d_model": d_model, "nhead": nhead, "batch": batch, "seq": seq, "rank": rank,
            "full_sec": round(t_full, 4), "full_tok_s": round(tok_full),
            "full_mem_mb": mem_full, "full_params": full_params,
            "lora_sec": round(t_lora, 4), "lora_tok_s": round(tok_lora),
            "lora_mem_mb": mem_lora, "lora_params": lora_params,
            "speedup": round(speedup, 2), "mem_save_pct": round(mem_save, 1),
        }
        del block, x, lora_a, lora_b
        _mps_reset()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    {c.red('SKIPPED (OOM)')}")
            results = {"error": "OOM"}
            _mps_reset()
        else:
            raise
    return results


# ---------------------------------------------------------------------------
# Practical estimates
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    ("GPT-2 Small",   0.124,  12,   768,  12),
    ("GPT-2 Medium",  0.355,  24,  1024,  16),
    ("GPT-2 Large",   0.774,  36,  1280,  20),
    ("Phi-2",         2.7,    32,  2560,  32),
    ("Mistral 7B",    7.2,    32,  4096,  32),
    ("Llama-3 8B",    8.0,    32,  4096,  32),
    ("Llama-2 13B",  13.0,    40,  5120,  40),
]


def model_feasibility(ram_gb):
    """Show which models fit in memory for inference, training, and LoRA."""
    avail = ram_gb * 0.75
    results = []

    print(f"\n    Available memory (est.): {c.bold(f'{avail:.1f} GB')} of {ram_gb:.1f} GB total\n")
    print(f"    {'Model':<15} {'Params':>7}  {'FP16 Inf':>10}  {'FP32 Train':>12}  {'LoRA FT':>10}")
    print(f"    {'':->15} {'':->7}  {'':->10}  {'':->12}  {'':->10}")

    for name, params_b, layers, d, nh in MODEL_CONFIGS:
        params = params_b * 1e9
        fp16_gb = params * 2 / 1e9
        train_gb = params * 20 / 1e9
        lora_gb = (params * 2 + params * 0.01 * 20) / 1e9

        def _tag(gb):
            if gb < avail:
                return c.green(f"\u2713 {gb:>5.1f}GB")
            return c.red(f"\u2717 {gb:>5.1f}GB")

        print(f"    {name:<15} {params_b:>5.1f}B   {_tag(fp16_gb)}   {_tag(train_gb):>12}   {_tag(lora_gb)}")
        results.append({
            "model": name, "params_B": params_b,
            "fp16_inf_gb": round(fp16_gb, 2), "fits_inf": fp16_gb < avail,
            "train_gb": round(train_gb, 2), "fits_train": train_gb < avail,
            "lora_gb": round(lora_gb, 2), "fits_lora": lora_gb < avail,
        })
    return results


def find_max_batch(d_model=1024, nhead=8, seq=512, device_name="mps"):
    """Find the largest training batch that fits in memory."""
    dev = torch.device(device_name)
    max_b = 0
    for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        _mps_reset()
        try:
            block = _make_block(d_model, nhead, dev)
            x = torch.randn(b, seq, d_model, device=dev, requires_grad=True)
            block.zero_grad()
            block(x).sum().backward()
            _sync()
            max_b = b
            del block, x
            _mps_reset()
        except RuntimeError:
            _mps_reset()
            break
    print(f"    Max batch (dim={d_model}, seq={seq}): {c.cyan(str(max_b))}")
    return {"d_model": d_model, "seq": seq, "max_batch": max_b}


def estimate_training_times(transformer_results, device_name):
    """Project per-step training time for popular models."""
    block_times: dict[int, float] = {}
    for r in transformer_results:
        if "error" not in r:
            block_times[r["d_model"]] = r["train_sec"]

    if not block_times:
        print(f"    {c.dim('(no transformer data available)')}")
        return []

    results = []
    steps = 1000
    print(f"\n    Projected for {c.bold(f'{steps}')} training steps "
          f"{c.dim('(single block scaled by layers)')}:\n")
    print(f"    {'Model':<15} {'Step':>8}  {'Steps/hr':>9}  {'1K steps':>10}")
    print(f"    {'':->15} {'':->8}  {'':->9}  {'':->10}")

    for name, params_b, layers, d, nh in MODEL_CONFIGS:
        closest = min(block_times, key=lambda k: abs(k - d))
        step_t = block_times[closest] * layers * (d / closest) ** 2
        sph = 3600 / step_t if step_t > 0 else 0
        total_h = steps * step_t / 3600

        if total_h < 1:
            fmt = f"{total_h * 60:.0f} min"
        elif total_h < 48:
            fmt = f"{total_h:.1f} hr"
        else:
            fmt = f"{total_h / 24:.1f} days"

        print(f"    {name:<15} {step_t:>6.3f}s  {sph:>9.0f}  {c.cyan(f'{fmt:>10}')}")
        results.append({
            "model": name, "layers": layers, "d_model": d,
            "step_sec": round(step_t, 3), "steps_per_hr": round(sph),
            "hours_1k": round(total_h, 2),
        })
    return results


def bench_inference_tok(configs, device_name):
    """Measure single-batch inference tokens/sec (prefill + decode)."""
    dev = torch.device(device_name)
    results = []
    for d_model, nhead, batch, seq in configs:
        _mps_reset()
        try:
            block = _make_block(d_model, nhead, dev)
            block.eval()

            x = torch.randn(batch, seq, d_model, device=dev)
            with torch.no_grad():
                t_pf = bench(lambda: block(x), warmup=2, iters=5, device=device_name)
            pf_tok = batch * seq / t_pf

            x1 = torch.randn(batch, 1, d_model, device=dev)
            with torch.no_grad():
                t_dec = bench(lambda: block(x1), warmup=2, iters=5, device=device_name)
            dec_tok = batch / t_dec

            mem = _mps_mem_mb()
            results.append({
                "d_model": d_model, "nhead": nhead, "batch": batch, "seq": seq,
                "prefill_tok_s": round(pf_tok), "decode_tok_s": round(dec_tok), "mem_mb": mem,
            })
            print(f"    dim={d_model:>4}  prefill:{c.cyan(f'{pf_tok:>8.0f}')} tok/s  "
                  f"decode:{c.cyan(f'{dec_tok:>6.0f}')} tok/s  {c.dim(f'mem:{mem}MB')}")
            del block, x, x1
            _mps_reset()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"    dim={d_model:>4}  {c.red('SKIPPED (OOM)')}")
                results.append({"d_model": d_model, "error": "OOM"})
                _mps_reset()
            else:
                raise
    return results


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def bench_disk_io(size_mb=512):
    data = os.urandom(size_mb * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp = f.name
    try:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            times.append(time.perf_counter() - t0)
        t_w = median(times)
        bw_w = size_mb / t_w / 1024
        print(f"    Seq Write ({size_mb}MB)  {t_w:.3f}s  {c.cyan(f'{bw_w:.2f}')} GB/s")

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with open(tmp, "rb") as f:
                f.read()
            times.append(time.perf_counter() - t0)
        t_r = median(times)
        bw_r = size_mb / t_r / 1024
        print(f"    Seq Read  ({size_mb}MB)  {t_r:.3f}s  {c.cyan(f'{bw_r:.2f}')} GB/s")

        return {
            "size_mb": size_mb,
            "write_sec": round(t_w, 3), "write_gb_s": round(bw_w, 2),
            "read_sec": round(t_r, 3), "read_gb_s": round(bw_r, 2),
        }
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Scoring & comparison
# ---------------------------------------------------------------------------

REFERENCE_CHIPS = [
    ("M1",        1.0,  "7-8",   2020),
    ("M1 Pro",    2.0,  "14-16", 2021),
    ("M1 Max",    3.5,  "24-32", 2021),
    ("M1 Ultra",  6.5,  "48-64", 2022),
    ("M2",        1.4,  "8-10",  2022),
    ("M2 Pro",    2.4,  "16-19", 2023),
    ("M2 Max",    4.5,  "30-38", 2023),
    ("M2 Ultra",  8.5,  "60-76", 2023),
    ("M3",        1.6,  "8-10",  2023),
    ("M3 Pro",    2.5,  "14-18", 2023),
    ("M3 Max",    5.5,  "30-40", 2023),
    ("M4",        1.8,  "10",    2024),
    ("M4 Pro",    3.0,  "16-20", 2024),
    ("M4 Max",    6.0,  "40",    2024),
]

CLOUD_GPUS = [
    ("NVIDIA RTX 4090",    40,  24),
    ("NVIDIA A100 (80GB)", 60,  80),
    ("NVIDIA H100",       100,  80),
]


def compute_score(transformer_results):
    for r in transformer_results:
        if "error" not in r and r.get("d_model") == 1024:
            return r["train_tok_s"]
    for r in transformer_results:
        if "error" not in r:
            return r["train_tok_s"]
    return 0


def print_comparison(score, chip_name):
    detected = None
    chip_lower = chip_name.lower().replace(" ", "")
    for ref_name, mult, cores, year in sorted(REFERENCE_CHIPS, key=lambda r: -len(r[0])):
        if ref_name.lower().replace(" ", "") in chip_lower:
            detected = (ref_name, mult)
            break

    base_score = score / detected[1] if detected else score
    max_mult = max(r[1] for r in REFERENCE_CHIPS)
    bar_w = 30

    print(f"\n    Your score: {c.bold(c.green(f'{score:,} tok/s'))}  "
          f"{c.dim('(transformer training, d=1024)')}\n")
    print(f"    {'Chip':<12} {'GPU Cores':>10} {'Est. tok/s':>11} {'vs You':>7}")
    print(f"    {'':->12} {'':->10} {'':->11} {'':->7}")

    for ref_name, mult, cores, year in REFERENCE_CHIPS:
        est = round(base_score * mult)
        rel = mult / detected[1] if detected else mult
        bar_len = int(mult / max_mult * bar_w)

        if detected and ref_name == detected[0]:
            bar = c.green("\u2588" * bar_len)
            tag = c.bold(c.yellow(" \u25c0 you"))
        elif detected and mult <= detected[1]:
            bar = c.green("\u2588" * bar_len)
            tag = ""
        else:
            bar = c.cyan("\u2588" * bar_len)
            tag = ""

        print(f"    {ref_name:<12} {cores:>10} {est:>11,}  {rel:>5.1f}x  {bar}{tag}")

    print(f"\n    {c.bold('Cloud GPU Equivalence')} {c.dim('(approximate)')}:")
    for name, mult, mem in CLOUD_GPUS:
        pct = (detected[1] if detected else 1.0) / mult * 100
        print(f"      ~{c.cyan(f'{pct:.1f}%')} of {name}")

    return {
        "score_tok_s": score,
        "detected_chip": detected[0] if detected else None,
        "reference": {
            name: round(base_score * mult)
            for name, mult, _, _ in REFERENCE_CHIPS
        },
        "cloud_pct": {
            name: round((detected[1] if detected else 1.0) / mult * 100, 1)
            for name, mult, _ in CLOUD_GPUS
        },
    }


# ---------------------------------------------------------------------------
# Submit (auto-PR via gh)
# ---------------------------------------------------------------------------

def _has_gh():
    """Check if gh CLI is installed and authenticated."""
    try:
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _try_auto_pr(submit_path, chip, ram_gb):
    """Attempt to auto-create a PR with gh. Returns PR URL or None."""
    repo_root = Path(__file__).parent
    if not (repo_root / ".git").exists():
        return None
    if not _has_gh():
        return None

    chip_slug = re.sub(r"[^a-zA-Z0-9]+", "-", chip).strip("-").lower()
    branch = f"results/{chip_slug}-{int(ram_gb)}gb"
    rel_path = submit_path.relative_to(repo_root)

    try:
        # Create branch from current HEAD
        subprocess.run(["git", "checkout", "-b", branch],
                       cwd=repo_root, capture_output=True, check=True)
        subprocess.run(["git", "add", str(rel_path)],
                       cwd=repo_root, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Add benchmark: {chip} {int(ram_gb)}GB"],
            cwd=repo_root, capture_output=True, check=True)

        # gh pr create handles forking automatically if needed
        r = subprocess.run(
            ["gh", "pr", "create",
             "--title", f"Add benchmark: {chip} {int(ram_gb)}GB",
             "--body", f"Automated submission from `python bench.py --submit`\n\n"
                       f"**Chip:** {chip}\n**RAM:** {int(ram_gb)}GB"],
            cwd=repo_root, capture_output=True, text=True, check=True)

        pr_url = r.stdout.strip()

        # Switch back to previous branch
        subprocess.run(["git", "checkout", "-"], cwd=repo_root, capture_output=True)
        return pr_url
    except subprocess.CalledProcessError:
        # Clean up: try to get back to previous branch
        subprocess.run(["git", "checkout", "-"], cwd=repo_root, capture_output=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Apple Silicon ML Benchmark")
    ap.add_argument("--quick", action="store_true", help="Skip slower benchmarks")
    ap.add_argument("--share", action="store_true", help="Print compact shareable summary")
    ap.add_argument("--submit", action="store_true", help="Save result and create PR")
    ap.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = ap.parse_args()

    if args.no_color:
        c.on = False

    gemm_sizes = [512, 1024, 2048, 4096]

    transformer_cfgs = [
        (768,  12, 8, 512),    # GPT-2 Small / BERT-base
        (1024, 16, 8, 512),    # GPT-2 Medium
        (2048, 16, 4, 512),    # larger
        (4096, 32, 2, 512),    # Llama-class
    ]

    inference_cfgs = [
        (768,  12, 1, 512),
        (1024, 16, 1, 512),
        (2048, 16, 1, 512),
        (4096, 32, 1, 512),
    ]

    W = 65
    print(c.bold("=" * W))
    print(c.bold("  Apple Silicon ML Benchmark"))
    print(c.bold("=" * W))

    # ---- system info ----
    _section("[System Info]")
    sys_info = collect_system_info()
    print_system_info(sys_info)

    res: dict = {
        "system": sys_info, "cpu": {}, "gpu": {},
        "transformer": {}, "practical": {}, "scoring": {},
    }
    has_mps = sys_info["mps_available"]
    device = "mps" if has_mps else "cpu"

    # ---- CPU low-level ----
    _section("[CPU Low-Level]")
    _sub("Matrix Multiply (GEMM)")
    res["cpu"]["gemm"] = bench_cpu_gemm(gemm_sizes)
    res["cpu"]["batch_matmul"] = bench_cpu_batch_matmul()
    res["cpu"]["memory_bw"] = bench_cpu_membw()

    # ---- GPU low-level ----
    if has_mps:
        _section("[GPU Low-Level (MPS)]")
        _sub("Matrix Multiply (GEMM)")
        res["gpu"]["gemm"] = bench_gpu_gemm(gemm_sizes)
        res["gpu"]["batch_matmul"] = bench_gpu_batch_matmul()
        res["gpu"]["conv2d"] = bench_gpu_conv2d()
        res["gpu"]["transfer"] = bench_gpu_transfer()
        res["gpu"]["fp16_fp32"] = bench_gpu_fp16_fp32()
    else:
        _section("[GPU SKIPPED - MPS not available]")

    # ---- transformer ----
    _section(f"[Transformer Benchmarks ({device.upper()})]")

    _sub("Training + inference throughput")
    res["transformer"]["throughput"] = bench_transformer_throughput(transformer_cfgs, device)

    if not args.quick:
        print()
        _sub("Sequence length scaling (dim=1024, batch=8)")
        res["transformer"]["seq_scaling"] = bench_seq_scaling(device_name=device)

        print()
        _sub("LoRA vs full fine-tuning (dim=1024)")
        res["transformer"]["lora"] = bench_lora(device_name=device)

    print()
    _sub("Inference throughput (batch=1)")
    res["transformer"]["inference"] = bench_inference_tok(inference_cfgs, device)

    # ---- practical ----
    _section("[Practical Estimates]")

    _sub("Model feasibility")
    res["practical"]["feasibility"] = model_feasibility(sys_info["ram_gb"])

    if not args.quick:
        print()
        _sub("Max batch size")
        res["practical"]["max_batch"] = find_max_batch(device_name=device)

    print()
    _sub("Training time estimates")
    res["practical"]["training_times"] = estimate_training_times(
        res["transformer"]["throughput"], device
    )

    # ---- disk I/O ----
    _section("[Disk I/O]")
    res["practical"]["disk_io"] = bench_disk_io()

    # ---- scoring ----
    _section("[Performance Score & Comparison]")
    score = compute_score(res["transformer"]["throughput"])
    res["scoring"] = print_comparison(score, sys_info["chip"])

    # ---- save ----
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    chip_slug = re.sub(r"[^a-zA-Z0-9]+", "_", sys_info["chip"]).strip("_")
    hostname = re.sub(r"[^a-zA-Z0-9]+", "_", sys_info["hostname"]).strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{hostname}_{chip_slug}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n{c.dim(f'Results saved to {out_path}')}")

    # ---- submit ----
    if args.submit:
        community_dir = Path(__file__).parent / "results" / "community"
        community_dir.mkdir(parents=True, exist_ok=True)
        ram_int = int(sys_info["ram_gb"])
        submit_name = f"{chip_slug}_{ram_int}gb_{hostname}.json"
        submit_path = community_dir / submit_name
        shutil.copy2(out_path, submit_path)

        print(f"\n{c.bold('Submitting results...')}")

        pr_url = _try_auto_pr(submit_path, sys_info["chip"], sys_info["ram_gb"])
        if pr_url:
            print(f"  {c.green('PR created:')} {pr_url}")
        else:
            print(f"  Result saved to {c.cyan(str(submit_path))}")
            print(f"  To submit manually:")
            print(f"    1. Fork {REPO_URL}")
            print(f"    2. Add {c.cyan(f'results/community/{submit_name}')}")
            print(f"    3. Open a pull request")

    # ---- share ----
    if args.share:
        max_inf = max_lora = "\u2014"
        for m in reversed(res.get("practical", {}).get("feasibility", [])):
            if m.get("fits_inf") and max_inf == "\u2014":
                max_inf = f"{m['model']} {m['params_B']}B"
            if m.get("fits_lora") and max_lora == "\u2014":
                max_lora = f"{m['model']} {m['params_B']}B"

        gemm4k = "\u2014"
        for g in res.get("gpu", {}).get("gemm", []):
            if g.get("size") == 4096:
                gemm4k = f"{g['gflops']:,.0f}"
        fp16x = res.get("gpu", {}).get("fp16_fp32", {}).get("fp16_speedup", "\u2014")
        os_short = sys_info["os_version"].split("-")[0].replace("macOS", "macOS ")

        border = c.cyan("\u2550" * 45)
        print(f"\n{border}")
        print(f"  {c.bold(sys_info['chip'])} | {int(sys_info['ram_gb'])}GB "
              f"| {os_short} | PyTorch {sys_info['torch_version']}")
        print(f"  Score: {c.bold(c.green(f'{score:,}'))} tok/s (d=1024 training)")
        print(f"  GPU GEMM 4K: {gemm4k} GFLOPS | FP16 speedup: {fp16x}x")
        print(f"  Max model (inference): {max_inf} | Max model (LoRA): {max_lora}")
        print(f"  {c.dim(REPO_URL)}")
        print(border)

    print(c.bold("=" * W))


if __name__ == "__main__":
    main()
