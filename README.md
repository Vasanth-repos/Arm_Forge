# 🦾 ArmForge — On-Device ARM64 LLM Optimization Engine

> **Track 3 — Mobile AI (ARM AI Optimization Challenge 2026)**  
> **Tagline:** 100% Offline, Private LLM Inference Stacking Arm KleidiAI Vector Kernels & Speculative Decoding on Client ARM64 Hardware.  
> **License:** Apache 2.0 (100% Open Source) | **Hardware:** Snapdragon X Plus ARM Processor (Client Laptop) | **Network Required:** 0 KB (Fully Offline)  

---

## 📑 Table of Contents
1. [Project Overview & Quick Start](#-project-overview--quick-start)
2. [Complete Step-by-Step Operations Guide](#-complete-step-by-step-operations-guide)
   - [Option A: Windows on ARM via WSL2 (Recommended)](#option-a-windows-on-arm-via-wsl2-recommended)
   - [Option B: Local ARM Linux / Mobile Workstation](#option-b-local-arm-linux--mobile-workstation)
3. [About the Project (Hackathon Story)](#-about-the-project)
   - [Inspiration](#inspiration)
   - [How I built it](#how-i-built-it)
   - [What I learned](#what-i-learned)
   - [Challenges faced](#challenges-faced)
4. [Technical Build & Architecture Specification](#-technical-build--architecture-specification)
   - [Optimizations Matrix](#optimizations--fixes-matrix)
   - [Repository Structure](#repository-structure)
5. [Benchmark Suite & Performance Comparison](#-benchmark-suite--performance-comparison)
   - [Performance Comparison Table](#performance-comparison-table)
   - [Visual Throughput & TTFT Bar Charts](#visual-performance-bar-charts)
6. [Redesigned Web Platform & Developer Tools](#-dev-web-platform--developer-tools)
7. [Tags & Try-It-Out Links](#-tags--try-it-out-links)

---

## 🚀 Project Overview & Quick Start

**ArmForge** evaluates and stacks hardware-level and architectural optimizations for Large Language Model (LLM) inference on ARM Neoverse & Snapdragon processors. All empirical results in this benchmark suite were obtained directly on an **ARM client laptop powered by the Snapdragon X Plus ARM Processor** (running WSL2 Ubuntu ARM64).

### 1-Command Master Execution
```bash
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge
bash scripts/run_all.sh
```

---

## 📖 Complete Step-by-Step Operations Guide

### Option A: Windows on ARM via WSL2 (Recommended)

Windows on ARM devices (Snapdragon X Elite / X Plus, Surface Pro 11, Lenovo Yoga Slim 7x) run ArmForge natively at 100% full hardware speed using WSL2 Ubuntu ARM64.

1. **Install WSL2 Ubuntu ARM64:** Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. **Launch & Run Master Pipeline:**
   ```bash
   wsl
   git clone https://github.com/Vasanth-repos/Armforge.git
   cd Armforge
   bash scripts/run_all.sh
   ```
3. **Launch Web Platform & Playground:**
   ```bash
   bash scripts/07_start_dashboard.sh
   ```
   Open your Windows browser at: **`http://localhost:8080`**

---

### Option B: Local ARM Linux / Mobile Workstation

* **Hardware:** Any ARM64 laptop or mobile workstation running Ubuntu 22.04 LTS (`aarch64`).

Run in terminal:
```bash
bash scripts/run_all.sh
```

---

## 💡 About the Project

### Inspiration

As Large Language Models become integral to everyday productivity, relying exclusively on remote server APIs introduces major compromises: sensitive personal data leaves the user's hands, latency depends on network quality, and apps become non-functional when offline.

Arm-powered laptops and client devices possess tremendous compute potential via ARM64 SIMD and vector extensions (`dotprod`, `i8mm`). However, naive local CPU inference frequently suffers from unoptimized generic matrix routines and memory bandwidth saturation, resulting in sluggish token generation.

I was inspired to build **ArmForge** to solve a fundamental question:

> *How can I stack hardware-level ARM vector kernels with architectural speculative decoding to deliver sub-500ms TTFT latency and peak generation throughput 100% on-device on a standard ARM laptop?*

### How I Built It

I engineered ArmForge in 4 distinct phases on my local ARM client laptop:

#### Phase 1: Vector Kernel Integration (Arm KleidiAI)
I compiled `llama.cpp` from source with `-DGGML_CPU_KLEIDIAI=ON` targeting ARM `dotprod` (`armv8.2-a+dotprod`) and `i8mm` vector instructions. By setting a batch size of `-b 512`, quantized matrix operations are routed directly through KleidiAI vectorized kernel paths.

#### Phase 2: Speculative Decoding Pair
I paired `Llama-3.2-3B-Instruct` (verifier) with a tokenizer-matched `Llama-3.2-1B-Instruct` (draft model). Using the exact same tokenizer family guarantees 100% token alignment:

$$\text{Vocabulary Match: } \mathcal{V}_{\text{verifier}} \equiv \mathcal{V}_{\text{draft}}$$

#### Phase 3: Dynamic Memory Channel Tuning
To prevent memory channel congestion on client chips, I created `03_tune_threads.sh` using `llama-bench` to sweep threads $T \in \{1, 2, 3, 4\}$. Throughput follows the client RAM bandwidth saturation curve:

$$\text{Throughput}(T) = \min\left(T \cdot \mu_{\text{core\_bandwidth}}, \mathcal{B}_{\text{RAM\_max}}\right)$$

#### Phase 4: Local Developer Interface
I built a FastAPI monitoring platform (`localhost:8080`) featuring real-time CPU/RAM meters, Chart.js performance graphs, a recommendation engine score (98/100), and a live streaming prompt playground powered by Server-Sent Events (SSE).

### What I Learned

1. **Hardware Vector Extensions Are Essential for Client AI:** Integrating Arm KleidiAI unlocked an **+56% throughput boost** (jumping from 5.2 tok/s to 8.1 tok/s) without modifying model weights.
2. **Speculative Decoding on CPU Solves TTFT Latency:** On client CPUs, verification occurs sequentially:
   $$T_{\text{step}} = \sum_{i=1}^{N} T_{\text{draft\_step}, i} + T_{\text{verifier\_verify}}$$
   While generation throughput remains flat, prefill overlap cuts TTFT latency by **-44%** (from 750 ms down to 420 ms).
3. **OS Memory Locking Prevents Latency Spikes:** Enforcing `--load-mode mlock` prevents the client operating system from swapping model weights to disk, ensuring smooth, predictable generation.

### Challenges Faced

* **Windows UNC Path Build Conflicts:** When compiling `llama.cpp` inside WSL across Windows file shares (`\\wsl.localhost`), CMake attempted to invoke `npm run build` for the built-in UI, failing due to CMD path limits. I resolved this by passing `-DLLAMA_BUILD_SERVER_WEBUI=OFF` to CMake, accelerating builds by 3x.
* **Non-Interactive Thread Probing:** Default interactive CLI modes caused thread sweep scripts to wait for user input. I refactored the tuner to use `llama-bench` with `--simple-io`, enabling automated non-interactive thread selection in under 20 seconds.

---

## 🛠️ Technical Build & Architecture Specification

### Optimizations & Fixes Matrix

| # | Issue / Challenge | Old Approach | Corrected Approach |
|---|---|---|---|
| 1 | Speculative decoding framing | Claimed throughput win on CPU | Framed as TTFT/latency play; isolated from KleidiAI in results |
| 2 | Benchmark warmup | No warmup — first result skewed | Warmup call before recording any measurements |
| 3 | Thread tuning | Fixed `nproc-1` | Sweep 1,2,3,4 threads via `llama-bench` non-interactive |
| 4 | Model locking | Deprecated `--mlock` | Updated to `--load-mode mlock` |
| 5 | GPU layer ambiguity | Implicit | `-ngl 0` explicit on all server calls |
| 6 | KleidiAI batch size | Default batch | `-b 512` added to activate dotprod kernel paths |
| 7 | Windows UNC CMake npm error | Failed on `\\wsl.localhost` | Added `-DLLAMA_BUILD_SERVER_WEBUI=OFF` |
| 8 | CLI argument deprecation | `--draft-max 5`, `--log-format` | Updated to `--spec-draft-n-max 5`, removed `--log-format` |
| 9 | UI/UX Quality & Math | Generic metrics UI | Redesigned Mobile AI developer platform (`:8080`) |
| 10 | On-Demand Server Auto-Spawn | Server manual launch only | Auto-spawn background server on port 8000/8001 if inactive |

### Repository Structure

```
armforge/
├── LICENSE
├── README.md                  ← Master repository landing page (this file)
├── ARMFORGE_COMPLETE_DOCS.md  ← Master consolidated documentation
├── TUTORIAL.md                ← Step-by-step operations tutorial
├── ArmForge_Project_Submission_Details.docx
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── run_all.sh             ← 1-command master execution script
│   ├── start_server.sh        ← KleidiAI model server helper (:8000)
│   ├── start_baseline_server.sh ← Baseline model server helper (:8001)
│   ├── start_both_servers.sh  ← Dual model server background launcher
│   ├── 00_bootstrap.sh
│   ├── 01_build_llamacpp.sh   ← Builds llama.cpp with KleidiAI ON (-DLLAMA_BUILD_SERVER_WEBUI=OFF)
│   ├── 01b_build_baseline.sh  ← Builds llama.cpp baseline (KleidiAI OFF)
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh     ← Fast llama-bench thread sweep
│   ├── 04_benchmark_baseline.sh
│   ├── 05_benchmark_kleidiai.sh
│   ├── 06_benchmark_optimized.sh
│   └── 07_start_dashboard.sh
├── inference/
│   ├── arm_features.py
│   ├── speculative_server.py
│   ├── sglang_server.py
│   └── demo_chat.py           ← Interactive CLI chat client
├── benchmark/
│   ├── run_bench.py           ← Records system reproducibility metadata
│   └── compare.py             ← Generates SUMMARY.md and ASCII bar charts
├── dashboard/
│   ├── app.py                 ← FastAPI streaming endpoint & metrics
│   └── templates/
│       └── index.html         ← Linear/Vercel-inspired Mobile AI UI platform
└── results/                   ← Benchmark JSON outputs & SUMMARY.md
```

---

## 📊 Benchmark Suite & Performance Comparison

### Performance Comparison Table (Empirical Results Obtained)

| Configuration | Throughput | TTFT | vs Baseline (tps) | vs Baseline (ttft) | Primary Focus |
|---|---|---|---|---|---|
| **[1] Baseline (vanilla `llama.cpp`, KleidiAI OFF)** | 5.2 tok/s | 750.0 ms | — | — | Reference baseline |
| **[2] + KleidiAI dotprod kernels** | **8.1 tok/s** | 620.0 ms | **+56%** | −17% | **Throughput Acceleration** |
| **[3] + KleidiAI + Speculative Decoding** | **8.0 tok/s** | **420.0 ms** | **+54%** | **−44%** | **TTFT Latency Reduction** |

### Visual Performance Bar Charts

```text
--- Throughput Comparison (tokens/sec — higher is better) ---
Baseline:     ############         5.2 tok/s
+KleidiAI:    #################### 8.1 tok/s (+56%)
+Speculative: ###################  8.0 tok/s (+54%)

--- Latency Comparison (TTFT ms — lower is better) ---
Baseline:     #################### 750.0 ms
+KleidiAI:    ################     620.0 ms (-17%)
+Speculative: ###########          420.0 ms (-44%)
```

---

## 🖥️ Redesigned Web Platform & Developer Tools

* **Mobile AI Developer UI (`:8080`):** Dark theme (`#0B0F14`), glassmorphism, Lucide icons, live CPU/RAM gauges, 7-stage pipeline visualizer, Chart.js performance charts, recommendation engine (score 98/100), centerpiece comparison cards, and live `SUMMARY.md` report display.
* **Live Playground:** Interactive prompt tester with real-time SSE streaming, auto-server spawning, and live meters (**TTFT ms**, **tok/s**, **tokens**, **elapsed time**).
* **Interactive CLI Chat (`demo_chat.py`):** Terminal client streaming responses turn-by-turn.
* **1-Click Export:** Download benchmark reports as JSON, Markdown Summary, CSV, or PDF.

---

## 🏷️ Tags & Try-It-Out Links

### Technology Tags
`mobile-ai, on-device-ai, offline-ai, arm64, kleidiai, llama-cpp, speculative-decoding, snapdragon-x, python3, fastapi, ubuntu, docker, docker-compose, gguf, pytorch, c++, cmake, huggingface, uvicorn, numactl, openblas, jinja2, bash, rest-api, llm-inference`

### Links
* **Main GitHub Repository:** `https://github.com/Vasanth-repos/Armforge.git`
* **Root Specification Repo:** `https://github.com/Vasanth-repos/Arm_Forge.git`
