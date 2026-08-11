# 🦾 ArmForge — On-Device ARM64 LLM Optimization Engine
## Comprehensive In-Depth Architecture, Implementation, & Benchmark Guide

> **Track:** Track 3 — Mobile AI (Arm AI Optimization Challenge 2026)  
> **Tagline:** 100% Offline, Private LLM Inference Stacking Arm KleidiAI Vector Kernels & Speculative Decoding on Client ARM64 Hardware  
> **License:** Apache 2.0 (100% Open Source) | **Hardware:** Snapdragon X Plus ARM Processor (Windows on ARM Client Laptop) | **Network Required:** 0 KB (Fully Offline)  

---

## 📑 Table of Contents
1. [Executive Summary & Project Identity](#1-executive-summary--project-identity)
2. [Problem Statement](#2-problem-statement)
3. [PROPOSED SOLUTION](#3-proposed-solution)
4. [4–6 Key Features of the Solution](#4-46-key-features-of-the-solution)
5. [Idea/Approach](#5-ideaapproach)
6. [System Architecture / Workflow](#6-system-architecture--workflow)
7. [Innovation & Existing Solutions](#7-innovation--existing-solutions)
8. [The 6-Phase Architectural Optimization Stack](#8-the-6-phase-architectural-optimization-stack)
9. [Hardware Proof & Silicon Architecture Detection](#9-hardware-proof--silicon-architecture-detection)
10. [Empirical Benchmark Results & Comparison Analysis](#10-empirical-benchmark-results--comparison-analysis)
11. [Web Dashboard & Real-Time Telemetry Engine](#11-web-dashboard--real-time-telemetry-engine)
12. [Impact & Future Scope](#12-impact--future-scope)
13. [Server Port Mapping & Script Pipeline Reference](#13-server-port-mapping--script-pipeline-reference)
14. [Setup, Operations, & Audit Verification](#14-setup-operations--audit-verification)

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

## 2. Problem Statement

On-device LLM execution on client ARM CPUs faces two fundamental performance bottlenecks:

1. **Memory Bandwidth Bottleneck (Autoregressive Generation Phase):**  
   During token generation, weight matrices must be transferred from DRAM memory into CPU registers for every single token produced. Standard matrix-vector multiplication routines (GEMV) fail to fully utilize available SIMD vector lanes, leading to severe DRAM memory bus saturation and low token generation throughput (often under 5–6 tokens/second).

2. **Prompt Processing Latency Delay (Prefill Phase):**  
   Processing long user prompts requires compute-heavy matrix-matrix multiplications (GEMM). On standard client CPUs, this prefill step creates a perceptible delay (750+ ms) before the user receives the first response character (Time-To-First-Token / TTFT).

3. **Cloud Dependency & Privacy Risks:**  
   Relying on cloud-hosted LLM APIs introduces subscription costs, API throttling, network latency overhead, and privacy risks when processing sensitive personal or enterprise data locally.

---

## 3. PROPOSED SOLUTION

**ArmForge** introduces a unified, 6-phase optimization engine that stacks hardware-level ARM vector ISA extensions (Arm KleidiAI) with high-level architectural speculative decoding (3B target + 1B draft & N-gram lookup).

By optimizing the entire execution stack—from compiler flags and vector micro-kernels up to cache geometry, thread topology, OS memory locking, and draft token prediction—ArmForge achieves sub-500ms TTFT latency and smooth token generation **100% on-device on standard ARM laptops**.

```
+-------------------------------------------------------------------+
|                        ArmForge Solution Stack                     |
+-------------------------------------------------------------------+
|  [Phase 6] OS Memory Lock & NUMA Pinning (mlock + numactl)         |
|  [Phase 5] Algorithmic Speculative Decoding (3B Target + 1B Draft) |
|  [Phase 4] Cache Geometry Prompt Batching (-b 512)                |
|  [Phase 3] Hardware Thread Topology Auto-Tuning (4 Physical Cores) |
|  [Phase 2] Mixed-Precision Quantization Compression (Q4_K_M)       |
|  [Phase 1] Arm KleidiAI Micro-Kernels (NEON / DotProd / i8mm)      |
+-------------------------------------------------------------------+
```

---

## 4. 4–6 Key Features of the Solution

1. **Arm KleidiAI Micro-Kernel Integration (`-DGGML_CPU_KLEIDIAI=ON`)**  
   Integrates Arm's official open-source KleidiAI library into `llama.cpp`, accelerating FP16/INT4 matrix multiplications via NEON, Dot Product (`dotprod`), and INT8 Matrix Multiply (`i8mm`) vector instruction sets for a **+50% pure kernel speedup**.

2. **Architectural Speculative Decoding (3B Target + 1B Draft & N-gram)**  
   Deploys a lightweight 1B draft model alongside the 3B target model on Port 8000. Generates multiple candidate tokens per step, verifying them in parallel in a single target forward pass (**72.5% draft acceptance rate**), cutting TTFT latency by **-44%**.

3. **Automated Thread Sweep Auto-Tuning (`03_tune_threads.sh`)**  
   Empirically benchmarks thread topologies (1, 2, 4, 8 threads) to lock execution to 4 physical cores, eliminating hyperthreading cache thrashing and thermal throttling.

4. **Real-Time Telemetry & Web Dashboard (Port 8080)**  
   Features a modern dark-mode FastAPI dashboard with Server-Sent Events (SSE) live streaming (`/api/stream`), real-time dynamic visual bar charts (Chart.js), and an interactive on-device AI Playground.

5. **OS Memory Locking & NUMA Topology Pinning (`mlock` + `numactl`)**  
   Prevents OS memory page swaps by locking weight matrices in RAM (`--mlock`) and binding allocations to physical NUMA sockets (`numactl --membind=0`).

6. **100% Offline & Zero-Dependency Execution**  
   Operates fully on-device with zero external API calls or network requirements (0 KB bandwidth), guaranteeing private execution.

---

## 5. Idea/Approach

ArmForge's core hypothesis is that **stacking optimizations across different abstraction layers yields compounding performance gains**:

```
Total Acceleration = (Hardware Micro-Kernel Boost) x (Quantization Bandwidth Reduction) x (Speculative Draft Prediction) x (OS NUMA Pinning)
```

1. **At the Hardware Layer:** Harness ARM vector ISA extensions (`dotprod`, `i8mm`) via Arm KleidiAI to maximize FLOPs per clock cycle during matrix multiplication.
2. **At the Data Layer:** Compress weight matrices using mixed 4-bit block quantization (`Q4_K_M`) to reduce DRAM memory bandwidth pressure during autoregressive generation.
3. **At the Algorithmic Layer:** Use speculative decoding to convert slow sequential token generation into fast parallel verification.
4. **At the System Layer:** Auto-tune thread topology and pin memory to physical NUMA nodes to ensure maximum CPU cache hits.

---

## 6. System Architecture / Workflow

ArmForge operates a 7-step telemetry workflow connecting the execution pipeline to the web dashboard:

```
 [1. run_all.sh] ----> Trigger benchmark suite
       |
 [2. llama-server] -> Ports 8000 (Optimized) & 8001 (Baseline) C++ instances
       |
 [3. run_bench.py] --> Python test harness (warmup run 0 discarded)
       |
 [4. HTTP REST API] -> Async POST /v1/completions prompt sweeps
       |
 [5. Sampling Engine] -> Empirical measurement of TTFT latency (ms) & tok/s
       |
 [6. JSON Storage] --> Persists metrics to results/bench_*.json
       |
 [7. Dashboard UI] --> Live SSE stream broadcast to http://localhost:8080
```

### Server Port Architecture
- **Port 8000:** KleidiAI Optimized Engine (`llama-server` built with `-DGGML_CPU_KLEIDIAI=ON` + speculative decoding).
- **Port 8001:** Baseline Vanilla Engine (`llama-server` built with `-DGGML_CPU_KLEIDIAI=OFF`).
- **Port 8080:** FastAPI Web Dashboard & Real-Time SSE Server.

---

## 7. Innovation & Existing Solutions

### Comparative Analysis Matrix

| Feature / Dimension | Standard llama.cpp (Baseline) | Cloud API Services (e.g. OpenAI) | ArmForge Optimization Engine |
|---|---|---|---|
| **Execution Environment** | Local CPU (Unoptimized) | Remote Cloud Data Center | **100% On-Device (Client ARM Laptop)** |
| **Network Dependency** | 0 KB | Requires high-speed internet | **0 KB (Fully Offline)** |
| **Privacy & Security** | Private | Data sent to third party | **100% On-Device Private** |
| **Hardware Kernel Acceleration** | Generic GEMV | Enterprise GPU Cluster | **Arm KleidiAI (NEON / i8mm UKERNELs)** |
| **TTFT Latency** | 750.0 ms | 300–800 ms (plus network RTT) | **400.0 ms (-47% cut)** |
| **Generation Speed** | 5.2 tok/s | Variable | **8.5 tok/s (+63% boost)** |
| **Token Cost** | $0.00 | $0.0015–$0.015 per 1K tokens | **$0.00 (Zero API Fees)** |

### Key Innovations in ArmForge
- **First Open-Source Full-Stack KleidiAI + Speculative Pipeline:** Combines Arm KleidiAI micro-kernels with dual-engine speculative decoding on client laptops.
- **Zero-Overhead N-Gram Lookup Integration:** Provides instant n-gram speculative prediction without requiring extra RAM for a secondary model.
- **Empirical SSE Live Dashboard:** Streamlines benchmark telemetry directly into a real-time web interface.

---

## 8. The 6-Phase Architectural Optimization Stack

### Phase 1: Arm KleidiAI Micro-Kernels (`-DGGML_CPU_KLEIDIAI=ON`)
Compiles `llama.cpp` against Arm's official KleidiAI kernel library. Replaces default FP16/INT4 matrix-vector multiplication loops with hand-optimized assembly micro-kernels leveraging ARM NEON, Dot Product (`dotprod`), and INT8 Matrix Multiply (`i8mm`).
- **Impact:** **+50% throughput gain** (from 5.2 tok/s to 7.8 tok/s) at identical precision (`Q8_0`).

### Phase 2: Mixed-Precision Quantization (`Q4_K_M`)
Replaces 8-bit quantization (`Q8_0`) with mixed 4-bit block quantization (`Q4_K_M`). Attention layers maintain 6-bit weights while feed-forward networks use 4-bit blocks.
- **Impact:** Reduces DRAM bandwidth demand by ~50%, raising generation throughput to **8.1 tok/s**.

### Phase 3: Fast Hardware Thread Sweep Auto-Tuning (`03_tune_threads.sh`)
Executes automated thread sweeps (1, 2, 4, 8 threads) using `llama-bench`.
- **Impact:** Discovers that 4 physical threads yield maximum performance without hyperthreading overhead.

### Phase 4: Cache Batching Geometry (`-b 512`)
Configures prompt processing batch size to 512 tokens (`-b 512`), aligning GEMM memory operations with CPU L2/L3 cache lines.

### Phase 5: Architectural Speculative Decoding (3B Target + 1B Draft & N-gram)
Runs a 1B draft model alongside the 3B target model on Port 8000. Drafts candidate tokens rapidly and verifies them in a single forward pass (**72.5% draft acceptance rate**).
- **Impact:** Reduces TTFT latency by **-44%** (down to 420 ms).

### Phase 6: OS & NUMA Memory Pinning (`mlock` + `numactl`)
Locks weight matrices into system RAM via `--mlock` and binds memory allocations to local physical NUMA nodes (`numactl --membind=0`).
- **Impact:** Reaches peak **8.5 tok/s** throughput and **400 ms** TTFT.

---

## 9. Hardware Proof & Silicon Architecture Detection

ArmForge includes an automated hardware feature detector (`armforge/inference/arm_features.py`) that inspects `/proc/cpuinfo` and `llama-cli --version` to produce `results/hardware.json`:

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

## 10. Empirical Benchmark Results & Comparison Analysis

All benchmarks were recorded using `run_all.sh`. Warmup run 0 was discarded to ensure statistical accuracy.

### Master Performance Breakdown Table

| Configuration | Throughput (tok/s) | TTFT Latency (ms) | vs Baseline (tps) | vs Baseline (TTFT) |
|---|---|---|---|---|
| **[1] Baseline Q8_0** (vanilla `llama.cpp`, KleidiAI OFF) | 5.2 tok/s | 750.0 ms | — | — |
| **[2] + KleidiAI Q8_0** (kernel upgrade) | 7.8 tok/s | 630.0 ms | +50% | -16% |
| **[3] + KleidiAI Q4_K_M + -b 512** (quantization) | 8.1 tok/s | 620.0 ms | +56% | -17% |
| **[4] + KleidiAI + Speculative draft-simple (3B+1B)** | 8.0 tok/s | 420.0 ms | +54% | -44% |
| **[5] + KleidiAI + Speculative ngram-simple** | 8.0 tok/s | 460.0 ms | +54% | -38% |
| **[6] + Full Stack** (`+mlock +numactl`) | **8.5 tok/s** | **400.0 ms** | **+63%** | **-47%** |

---

## 11. Web Dashboard & Real-Time Telemetry Engine

The dashboard (`dashboard/app.py` & `dashboard/templates/index.html`) runs on `http://localhost:8080`.

### Features
- **Live SSE Stream Table:** Displays live benchmark runs formatted cleanly (`YYYY-MM-DD HH:MM:SS`), filtering out invalid/zero runs.
- **Dynamic Visual Bar Charts:** Real-time Chart.js charts (`tpsChart`, `ttftChart`, `lbChart`) that dynamically update bar heights as telemetry streams in.
- **On-Device Interactive Playground:** Allows users to run prompt completions on Port 8000 (Optimized) or Port 8001 (Baseline) directly from the UI.
- **7-Step Workflow Architecture Diagram:** A responsive 2-tier visual diagram showing data movement from shell script to web interface.

---

## 12. Impact & Future Scope

### Immediate Impact
- **Democratizes High-Performance On-Device AI:** Enables smooth, responsive LLM assistant execution on standard ARM laptops without requiring expensive discrete GPUs.
- **Zero Privacy Leakage & Zero Token Costs:** Complete local execution protects user data privacy while eliminating API subscription fees.

### Future Scope & Roadmap
1. **Arm SVE / SVE2 Vector Extension Support:** Incorporate scalable vector extensions (SVE2) for next-generation ARM Neoverse & Snapdragon processors.
2. **NPU Offloading & Hybrid Execution:** Explore offloading prefill prompt processing to on-device Neural Processing Units (NPUs) while running speculative generation on ARM CPU cores.
3. **Multi-Model Speculative Tree Search:** Implement tree-based speculative decoding (Medusa / EAGLE head integration) to increase candidate token acceptance rates above 85%.

---

## 13. Server Port Mapping & Script Pipeline Reference

### Port Specifications
- **Port 8000:** KleidiAI Optimized Engine (`llama-server` with KleidiAI + speculative decoding).
- **Port 8001:** Baseline Vanilla Engine (`llama-server` vanilla build).
- **Port 8080:** FastAPI Web Dashboard & Live SSE Streaming Server.

### Script Pipeline Order (`run_all.sh`)
1. `00_bootstrap.sh` — Environment & dependency bootstrap.
2. `01_build_llamacpp.sh` — Compiles `llama.cpp` with `-DGGML_CPU_KLEIDIAI=ON`.
3. `01b_build_baseline.sh` — Compiles baseline `llama.cpp` with `-DGGML_CPU_KLEIDIAI=OFF`.
4. `02_download_models.sh` — Downloads GGUF models (`Llama-3.2-3B` & `Llama-3.2-1B`).
5. `03_tune_threads.sh` — Runs thread sweep auto-tuning.
6. `04_benchmark_baseline.sh` — Benchmarks baseline server on Port 8001.
7. `05_benchmark_kleidiai.sh` — Benchmarks KleidiAI server on Port 8000.
8. `06_benchmark_optimized.sh` — Benchmarks KleidiAI + speculative full stack.
9. `07_start_dashboard.sh` — Launches web dashboard on Port 8080.

---

## 14. Setup, Operations, & Audit Verification

### Master Command Suite
```bash
# Clone repository
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge

# Run complete benchmark pipeline
bash scripts/run_all.sh

# Start web dashboard
bash scripts/07_start_dashboard.sh
```

### Automated Audit Verification
```bash
python scratch/verify_audit.py
```

---
*ArmForge Project Guide — Arm AI Optimization Challenge 2026*
