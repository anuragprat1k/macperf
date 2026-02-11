# macperf — Apple Silicon ML Benchmark

Benchmark your Mac's ML performance in one command. Get practical numbers — transformer throughput in tokens/sec, which models fit in your RAM, estimated training times — not just raw GFLOPS.

Built for ML researchers who want to know: *"Can I fine-tune this model on my MacBook, and how long will it take?"*

## Quick Start

```bash
pip install torch numpy
python bench.py
```

That's it. Results print to terminal and save as JSON in `results/`.

For a faster run (~2 min instead of ~5 min), skip sequence scaling, LoRA, and max batch tests:

```bash
python bench.py --quick
```

## What It Measures

### Low-Level Compute
- **CPU & GPU GEMM** — Matrix multiply at 512 to 4096, reported as GFLOPS
- **FP16 vs FP32** — Mixed-precision speedup on MPS
- **Memory bandwidth** — CPU copy throughput, GPU transfer (CPU↔GPU)
- **Conv2D** — Vision model staple

### Transformer Benchmarks
- **Training throughput** — Forward + backward pass through a transformer block at d=768/1024/2048/4096, reported as **tokens/sec**
- **Inference throughput** — Prefill and single-token decode speed
- **Sequence length scaling** — Shows the O(n²) attention wall on your hardware
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
    d= 768 h=12 b= 8 s= 512  train:   17096 tok/s  infer:   51231 tok/s
    d=1024 h=16 b= 8 s= 512  train:   11974 tok/s  infer:   31680 tok/s
    d=2048 h=16 b= 4 s= 512  train:    3759 tok/s  infer:   10447 tok/s
    d=4096 h=32 b= 2 s= 512  train:     943 tok/s  infer:    2201 tok/s

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

## JSON Output

Results save to `results/{hostname}_{chip}_{timestamp}.json` with the full data:

```json
{
  "system": { "chip": "Apple M1", "ram_gb": 8.0, ... },
  "cpu": { "gemm": [...], "memory_bw": {...} },
  "gpu": { "gemm": [...], "fp16_fp32": {...} },
  "transformer": { "throughput": [...], "seq_scaling": [...], "lora": {...} },
  "practical": { "feasibility": [...], "training_times": [...] },
  "scoring": { "score_tok_s": 11974, "reference": {...} }
}
```

Use this to compare across machines or track performance across PyTorch versions.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.9+
- PyTorch 2.0+ (with MPS support)
- NumPy

```bash
pip install torch numpy
```

Falls back to CPU-only benchmarks if MPS is unavailable.

## Contributing

Run the benchmark on your machine and open a PR adding your results to the comparison database. We especially want data from M2/M3/M4 Pro/Max/Ultra machines.

## License

MIT
