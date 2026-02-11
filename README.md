# macperf — Apple Silicon ML Benchmark

Benchmark your Mac's ML performance in one command. Get practical numbers — transformer throughput in tokens/sec, which models fit in your RAM, estimated training times — not just raw GFLOPS.

Built for ML researchers who want to know: *"Can I fine-tune this model on my MacBook, and how long will it take?"*

## Quick Start

```bash
git clone https://github.com/anuragprat1k/macperf.git
cd macperf
pip install torch numpy
python bench.py
```

Or run it directly:

```bash
curl -sL https://raw.githubusercontent.com/aragun/macperf/main/bench.py | python
```

For a faster run (~2 min instead of ~5), skip sequence scaling, LoRA, and max batch tests:

```bash
python bench.py --quick
```

## Leaderboard

See **[LEADERBOARD.md](LEADERBOARD.md)** for the full community results table.

| Rank | Chip | RAM | Score (tok/s) | GEMM 4K | FP16x | Max LoRA Model |
|-----:|------|----:|--------------:|--------:|------:|----------------|
| 1 | Apple M1 | 8GB | 11,974 | 1,536 | 1.32x | Phi-2 |

**Your machine not here? [Submit your results!](#submit-your-results)**

## What It Measures

### Low-Level Compute
- **CPU & GPU GEMM** — Matrix multiply at 512 to 4096, reported as GFLOPS
- **FP16 vs FP32** — Mixed-precision speedup on MPS
- **Memory bandwidth** — CPU copy throughput, GPU transfer (CPU<>GPU)
- **Conv2D** — Vision model staple

### Transformer Benchmarks
- **Training throughput** — Forward + backward pass through a transformer block at dim=768/1024/2048/4096, reported as **tokens/sec**
- **Inference throughput** — Prefill and single-token decode speed
- **Sequence length scaling** — Shows the O(n^2) attention wall on your hardware
- **LoRA vs full fine-tuning** — Speedup and memory savings from parameter-efficient training

### Practical Estimates
- **Model feasibility table** — Which popular models (GPT-2, Phi-2, Llama, Mistral) fit in your RAM for FP16 inference, full training, or LoRA
- **Max batch size** — Largest training batch before OOM
- **Training time projections** — Estimated step time and hours-to-complete for each model
- **Disk I/O** — Sequential read/write (catches data-loading bottlenecks)

### Scoring & Comparison
- **Composite score** in tokens/sec for easy comparison
- **Reference table** with estimated performance for all Apple Silicon chips (M1 through M4 Max)
- **Cloud GPU equivalence** — How your Mac compares to RTX 4090, A100, H100

## Example Output (M1 MacBook Air, 8GB)

```
[Transformer Benchmarks (MPS)]
  Training + inference throughput
    dim= 768 heads=12 batch= 8 seq= 512  train:   17096 tok/s  infer:   51231 tok/s
    dim=1024 heads=16 batch= 8 seq= 512  train:   11974 tok/s  infer:   31680 tok/s
    dim=2048 heads=16 batch= 4 seq= 512  train:    3759 tok/s  infer:   10447 tok/s
    dim=4096 heads=32 batch= 2 seq= 512  train:     943 tok/s  infer:    2201 tok/s

[Practical Estimates]
  Model feasibility
    Model            Params    FP16 Inf    FP32 Train     LoRA FT
    GPT-2 Small       0.1B   ✓   0.2GB      ✓   2.5GB   ✓   0.3GB
    GPT-2 Medium      0.4B   ✓   0.7GB      ✗   7.1GB   ✓   0.8GB
    Phi-2             2.7B   ✓   5.4GB      ✗  54.0GB   ✓   5.9GB
    Mistral 7B        7.2B   ✗  14.4GB      ✗ 144.0GB   ✗  15.8GB

[Performance Score & Comparison]
    Your score: 11,974 tok/s  (transformer training, d=1024)

    M1                  7-8      11,974    1.0x  ████ <- you
    M2                 8-10      16,764    1.4x  █████
    M3 Pro            14-18      29,935    2.5x  ██████████
    M4 Max               40      71,844    6.0x  ████████████████████████
```

## Share Your Results

Print a compact summary to paste in discussions:

```bash
python bench.py --share
```

```
══════════════════════════════════════
Apple M1 | 8GB | macOS  14.2.1 | PyTorch 2.9.1
Score: 11,974 tok/s (d=1024 training)
GPU GEMM 4K: 1,536 GFLOPS | FP16 speedup: 1.32x
Max model (inference): Phi-2 2.7B | Max model (LoRA): Phi-2 2.7B
https://github.com/anuragprat1k/macperf
══════════════════════════════════════
```

## Submit Your Results

We want data from every Apple Silicon chip. Here's how to contribute:

1. **Run the benchmark with `--submit`:**
   ```bash
   python bench.py --submit
   ```
   This saves your result to `results/community/`.

2. **Fork this repo and add your file:**
   ```bash
   git add results/community/
   git commit -m "Add results: [your chip] [your RAM]"
   ```

3. **Open a pull request.** The leaderboard will be updated after merge.

We especially want results from: M1 Pro/Max/Ultra, M2 family, M3 family, M4 family.

## JSON Output

Results save to `results/{hostname}_{chip}_{timestamp}.json` with the full data:

```json
{
  "system": { "chip": "Apple M1", "ram_gb": 8.0, "..." : "..." },
  "cpu": { "gemm": ["..."], "memory_bw": {"..."  : "..."} },
  "gpu": { "gemm": ["..."], "fp16_fp32": {"..." : "..."} },
  "transformer": { "throughput": ["..."], "seq_scaling": ["..."], "lora": {"..." : "..."} },
  "practical": { "feasibility": ["..."], "training_times": ["..."] },
  "scoring": { "score_tok_s": 11974, "reference": {"..." : "..."} }
}
```

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.9+
- PyTorch 2.0+ (with MPS support)
- NumPy

```bash
pip install torch numpy
```

Falls back to CPU-only benchmarks if MPS is unavailable.

## License

MIT
