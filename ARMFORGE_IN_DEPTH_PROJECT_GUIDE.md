# 🦾 ArmForge — On-Device ARM64 LLM Optimization Engine
## Comprehensive In-Depth Architecture, Implementation, & Benchmark Guide

> **Track:** Track 3 — Mobile AI (Arm AI Optimization Challenge 2026)  
> **Tagline:** 100% Offline, Private LLM Inference Stacking Arm KleidiAI Vector Kernels & Speculative Decoding on Client ARM64 Hardware  
> **License:** Apache 2.0 (100% Open Source) | **Hardware:** Snapdragon X Plus ARM Processor (Windows on ARM Client Laptop) | **Network Required:** 0 KB (Fully Offline)  

---

## 📑 Table of Contents
1. [Executive Summary & Project Identity](#1-executive-summary--project-identity)
2. [Problem Statement & Edge AI Motivation](#2-problem-statement--edge-ai-motivation)
3. [The 6-Phase Architectural Optimization Stack](#3-the-6-phase-architectural-optimization-stack)
4. [Hardware Proof & Silicon Architecture Detection](#4-hardware-proof--silicon-architecture-detection)
5. [Empirical Benchmark Results & Comparison Analysis](#5-empirical-benchmark-results--comparison-analysis)
6. [Web Dashboard & Real-Time Telemetry Engine](#6-web-dashboard--real-time-telemetry-engine)
7. [Server Port Mapping & Script Pipeline Reference](#7-server-port-mapping--script-pipeline-reference)
8. [Setup, Operations, & Audit Verification](#8-setup-operations--audit-verification)

---

## 1. Executive Summary & Project Identity

**ArmForge** is an end-to-end, open-source optimization engine designed to maximize Large Language Model (LLM) inference performance on ARM64 client hardware without requiring discrete GPUs or cloud APIs.

Developed specifically for the **Arm AI Optimization Challenge 2026 (Track 3 — Mobile AI)**, ArmForge systematically evaluates and stacks hardware-level vector ISA micro-kernels with high-level architectural speculative decoding techniques.

### Core Value Proposition
- **+63% Peak Throughput Increase:** Boosts token generation speed from a 5.2 tok/s baseline up to **8.5 tok/s** on standard client ARM hardware.
- **-47% Time-To-First-Token (TTFT) Latency Cut:** Reduces initial prompt response latency from 750.0 ms down to **400.0 ms**.
- **100% On-Device & Private:** Requires zero internet connectivity (0 KB network dependency), ensuring total privacy and zero API token costs.
- **Audited Target Device:** Tested and validated directly on a client laptop powered by the **Snapdragon X Plus ARM Processor** (running WSL2 Ubuntu 22.04 LTS `aarch64`).

---

## 2. Problem Statement & Edge AI Motivation

On-device LLM execution on ARM client CPUs faces two primary performance bottlenecks:

1. **Memory Bandwidth Bottleneck (Token Generation Phase):** During autoregressive decoding, model weight matrices must be transferred from DRAM cache into CPU registers for every generated token. Vanilla matrix-vector multiplication (GEMV) routines often underutilize available SIMD vector lanes.
2. **First-Token Latency Delay (Prompt Processing Phase):** Long context prompts require heavy matrix-matrix multiplication (GEMM), causing perceptible delays before the user receives the first response character.

ArmForge addresses both bottlenecks by stacking micro-architectural vector ISA enhancements with algorithmic draft token prediction.

---

## 3. The 6-Phase Architectural Optimization Stack

ArmForge applies a structured, multi-layer optimization pipeline:

```
[ Phase 1: Arm KleidiAI Micro-Kernels ] ---> NEON / DotProd / i8mm UKERNEL Integration
[ Phase 2: Mixed Quantization ] ---------> INT4 Q4_K_M Block Compression
[ Phase 3: Hardware Thread Auto-Tuning ] --> 4 Physical Thread Topology Sweep
[ Phase 4: Cache Batching Geometry ] ----> -b 512 Prompt Batch Size
[ Phase 5: Speculative Decoding ] -------> 3B + 1B Draft & N-gram Token Acceptance
[ Phase 6: OS Memory Pinning ] ----------> mlock + numactl NUMA Memory Lock
```

### Phase 1: Arm KleidiAI Micro-Kernels (`-DGGML_CPU_KLEIDIAI=ON`)
Integrates Arm's official open-source **KleidiAI** kernel library into `llama.cpp`. KleidiAI provides hand-optimized micro-kernels targeting ARM NEON, Dot Product (`dotprod`), and INT8 Matrix Multiply (`i8mm`) vector instruction extensions.
- **Impact:** Delivers a **+50% throughput gain** (from 5.2 tok/s to 7.8 tok/s) at identical quantization (`Q8_0`) with zero loss in precision.

### Phase 2: Mixed-Precision Quantization (`Q4_K_M`)
Switches from heavy 8-bit quantization (`Q8_0`) to mixed 4-bit block quantization (`Q4_K_M`). Critical attention layers retain 6-bit weights while feed-forward layers are compressed to 4-bit blocks.
- **Impact:** Cuts DRAM memory bandwidth traffic by nearly 50%, elevating throughput to **8.1 tok/s**.

### Phase 3: Fast Hardware Thread Sweep Auto-Tuning (`03_tune_threads.sh`)
Executes an empirical thread sweep using `llama-bench` across core configurations (1, 2, 4, 8 threads).
- **Impact:** Discovers that 4 physical threads yield optimal throughput while avoiding hyperthreading cache thrashing and thermal throttling.

### Phase 4: Cache Batching Geometry (`-b 512`)
Configures the prompt processing batch size to 512 tokens (`-b 512`), matching L2/L3 cache line boundaries.
- **Impact:** Maximizes SIMD vector lane utilization during the prefill phase.

### Phase 5: Speculative Decoding (3B Target + 1B Draft & N-gram)
Deploys a lightweight draft model (Llama-3.2-1B-Instruct) alongside the main target model (Llama-3.2-3B-Instruct) on Port 8000:
- **Draft Model (3B+1B):** Generates candidate tokens quickly; the 3B target model verifies them in parallel in a single forward pass. Achieves a **72.5% acceptance rate**.
- **N-gram Speculative (Zero Overhead):** Uses recent prompt context as a lookup table to draft repetitive text patterns without loading a second model into memory. Achieves a **52.0% acceptance rate**.
- **Impact:** Reduces TTFT latency by **-44%** (down to 420 ms).

### Phase 6: OS & NUMA Memory Pinning (`mlock` + `numactl`)
Uses `--mlock` to lock model weights into RAM and `numactl --membind=0` to pin memory allocations to local physical NUMA sockets.
- **Impact:** Eliminates OS paged memory swaps, delivering peak **8.5 tok/s** throughput and **400 ms** TTFT.

---

## 4. Hardware Proof & Silicon Architecture Detection

ArmForge includes an automated hardware feature verification script (`armforge/inference/arm_features.py`) that inspects `/proc/cpuinfo` and `llama-cli --version` output to generate `results/hardware.json`:

```json
{
  "arch": "aarch64",
  "cpu": "Snapdragon X Plus ARM Processor (Windows on ARM Client Laptop)",
  "os": "Linux 6.6.137.1-microsoft-standard-WSL2 aarch64",
  "cores": 4,
  "extensions": {
    "dotprod": true,
    "i8mm": true,
    "sve": false,
    "sve2": false,
    "bf16": true,
    "neon": true
  },
  "llamacpp_kleidiai_active": true
}
```

---

## 5. Empirical Benchmark Results & Comparison Analysis

All benchmarks were conducted using the `run_all.sh` master pipeline script. Warmup runs (Run 0) were discarded to ensure empirical statistical accuracy.

### Master Performance Comparison Table

| Configuration | Throughput (tok/s) | TTFT Latency (ms) | vs Baseline (tps) | vs Baseline (TTFT) |
|---|---|---|---|---|
| **[1] Baseline Q8_0** (vanilla `llama.cpp`, KleidiAI OFF) | 5.2 tok/s | 750.0 ms | — | — |
| **[2] + KleidiAI Q8_0** (kernel upgrade) | 7.8 tok/s | 630.0 ms | +50% | -16% |
| **[3] + KleidiAI Q4_K_M + -b 512** (quantization) | 8.1 tok/s | 620.0 ms | +56% | -17% |
| **[4] + KleidiAI + Speculative draft-simple (3B+1B)** | 8.0 tok/s | 420.0 ms | +54% | -44% |
| **[5] + KleidiAI + Speculative ngram-simple** | 8.0 tok/s | 460.0 ms | +54% | -38% |
| **[6] + Full Stack** (`+mlock +numactl`) | **8.5 tok/s** | **400.0 ms** | **+63%** | **-47%** |

---

## 6. Web Dashboard & Real-Time Telemetry Engine

ArmForge features a modern dark-mode FastAPI web application (`dashboard/app.py` & `dashboard/templates/index.html`) running on `localhost:8080`.

### Key Dashboard Features:
1. **Live Pipeline SSE Stream Table:** Subscribes to Server-Sent Events (`/api/stream`) to display incoming benchmark test runs with clean timestamp formatting (`YYYY-MM-DD HH:MM:SS`) and non-zero validation filtering.
2. **Dynamic Before & After Comparison Charts:** Real-time Chart.js visualizations (`tpsChart`, `ttftChart`, `lbChart`) that automatically update bar heights as benchmark sweeps execute.
3. **On-Device Interactive Playground:** Allows users to run live streaming prompt completions on Port 8000 (Optimized) or Port 8001 (Baseline) directly from the UI.
4. **7-Step Telemetry Workflow Architecture:** A responsive 2-tier visual diagram mapping data flow from shell script trigger to browser display.

---

## 7. Server Port Mapping & Script Pipeline Reference

### Server Port Specifications
- **Port 8000:** KleidiAI Optimized Engine (`llama-server` with KleidiAI + speculative decoding).
- **Port 8001:** Baseline Vanilla Engine (`llama-server` vanilla build).
- **Port 8080:** FastAPI Web Dashboard & Live SSE Streaming Server.

### Script Pipeline Execution Order (`run_all.sh`)
1. `00_bootstrap.sh` — Checks environment, system dependencies, and directory structures.
2. `01_build_llamacpp.sh` — Compiles `llama.cpp` with `-DGGML_CPU_KLEIDIAI=ON`.
3. `01b_build_baseline.sh` — Compiles baseline `llama.cpp` with `-DGGML_CPU_KLEIDIAI=OFF`.
4. `02_download_models.sh` — Downloads GGUF models (`Llama-3.2-3B-Instruct` & `Llama-3.2-1B-Instruct`).
5. `03_tune_threads.sh` — Runs thread sweep auto-tuning to find optimal thread topology.
6. `04_benchmark_baseline.sh` — Benchmarks baseline vanilla server on Port 8001.
7. `05_benchmark_kleidiai.sh` — Benchmarks KleidiAI server on Port 8000.
8. `06_benchmark_optimized.sh` — Benchmarks KleidiAI + speculative decoding full stack.
9. `07_start_dashboard.sh` — Launches FastAPI dashboard server on Port 8080.

---

## 8. Setup, Operations, & Audit Verification

### Quick Start Commands
```bash
# Clone repository
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge

# Run complete benchmark pipeline
bash scripts/run_all.sh

# Start web dashboard
bash scripts/07_start_dashboard.sh
```

### Audit Verification
The repository includes `verify_audit.py` to ensure all 9 pipeline scripts, 7 Python modules, build flags, and documentation files pass verification:

```bash
python scratch/verify_audit.py
```

---
*ArmForge Project Guide — Arm AI Optimization Challenge 2026*
