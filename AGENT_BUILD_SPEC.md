# ArmForge — Agent Build Specification v3 (Corrected)
> **AI Agent Instructions:** Execute every shell block in strict order. All files defined inline. Do not skip sections. Each phase must succeed before proceeding.

---

## Project Summary
**Name:** ArmForge  
**Track:** ARM AI Optimization Challenge 2026 — Cloud AI (Track 2)  
**Tagline:** Open-source ARM64 LLM inference benchmark: KleidiAI + speculative decoding on free Oracle A1  
**License:** Apache 2.0 | **Cost:** $0 | **Python:** 3.12 | **OS:** Ubuntu 22.04 aarch64

---

## Corrections Applied vs v2
| # | Issue | Old Approach | Corrected Approach |
|---|---|---|---|
| 1 | Speculative decoding framing | Claimed throughput win on CPU | Framed as TTFT/latency play; isolated from KleidiAI in results |
| 2 | Benchmark warmup | No warmup — first result skewed | Warmup call before recording any measurements |
| 3 | Thread tuning | Fixed `nproc-1` | Sweep 1,2,3,4 threads; auto-select best before final benchmark |
| 4 | Model locking | No mlock | `--mlock` added to prevent swap thrash on 12 GB instance |
| 5 | GPU layer ambiguity | Implicit | `-ngl 0` explicit on all server calls |
| 6 | KleidiAI batch size | Default batch | `-b 512` added to activate dotprod kernel paths |
| 7 | SGLang in comparison | In primary benchmark table | Demoted to optional bonus; not in main comparison |
| 8 | Result isolation | KleidiAI+speculative combined | Three-row table: baseline / +KleidiAI / +KleidiAI+speculative |
| 9 | Oracle region note | Missing | Added: use ap-singapore-1 or eu-frankfurt-1 if A1 unavailable |
| 10 | Batch size in benchmark | 128 tokens fixed | Configurable; throughput measured per-token correctly |

---

## Repository Structure
```
armforge/
├── LICENSE
├── README.md
├── AGENT_BUILD_SPEC.md       ← this file
├── requirements.txt
├── scripts/
│   ├── 00_bootstrap.sh
│   ├── 01_build_llamacpp.sh
│   ├── 01b_build_baseline.sh  ← NEW: separate baseline build
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh     ← NEW: thread sweep before benchmarking
│   ├── 04_benchmark_baseline.sh
│   ├── 05_benchmark_kleidiai.sh    ← NEW: KleidiAI only, no speculative
│   ├── 06_benchmark_optimized.sh   ← KleidiAI + speculative
│   └── 07_start_dashboard.sh
├── inference/
│   ├── arm_features.py
│   ├── speculative_server.py
│   └── sglang_server.py
├── benchmark/
│   ├── run_bench.py
│   └── compare.py
├── dashboard/
│   ├── app.py
│   └── templates/index.html
└── results/
```

---

## Infrastructure

### Oracle Cloud A1 (Primary — Always Free)
```
1. https://cloud.oracle.com/free → sign up
2. Create instance:
   Shape:   VM.Standard.A1.Flex
   OCPUs:   2  (set manually — default may give 1)
   Memory:  12 GB (set manually — default may give 6)
   OS:      Canonical Ubuntu 22.04 aarch64
   Storage: 100 GB boot volume
3. Security List: open ports 22, 8000, 8001, 8002, 8080, 30000
4. SSH: ssh -i key.pem ubuntu@YOUR_IP

REGION NOTE: A1 is often out of capacity in US regions.
Try: ap-singapore-1, eu-frankfurt-1, ap-tokyo-1 (in that order).
```

### AWS Graviton2 (Fallback — Free until Dec 31 2026)
```
Instance: t4g.small (2 vCPU, 2 GB RAM)
OS:       Ubuntu 22.04 ARM64 AMI
Limit:    Use Q2_K model. No SGLang. No speculative (OOM risk).
```

---

## Tech Stack
| Layer | Tool | Version | Notes |
|---|---|---|---|
| Primary inference | llama.cpp | latest main | KleidiAI build |
| ARM acceleration | KleidiAI | bundled in llama.cpp | GGML_CPU_KLEIDIAI=ON |
| Secondary (bonus) | SGLang | 0.5.16+ | optional only |
| Main model | Llama-3.2-3B-Instruct-Q4_K_M | ~2.0 GB | bartowski mirror |
| Draft model | Llama-3.2-1B-Instruct-Q4_K_M | ~0.7 GB | same tokenizer family |
| Web framework | FastAPI | 0.111+ | dashboard |
| Build tools | cmake 3.22+, gcc 12+ | system | |

---

## requirements.txt
```
fastapi==0.111.1
uvicorn[standard]==0.29.0
psutil==5.9.8
httpx==0.27.0
huggingface_hub==0.23.4
pandas==2.2.2
tabulate==0.9.0
jinja2==3.1.4
requests==2.32.3
```

---

## LICENSE — Apache 2.0
```
Copyright 2026 ArmForge Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
```

---

## PHASE 0 — Bootstrap

### scripts/00_bootstrap.sh
```bash
#!/bin/bash
set -e
echo "=== ArmForge Bootstrap ==="

[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: Must run on aarch64"; exit 1; }
echo "Platform: aarch64 OK"

sudo apt-get update -qq
sudo apt-get install -y \
  build-essential cmake git wget curl \
  python3.12 python3.12-venv python3-pip \
  libblas-dev liblapack-dev libopenblas-dev \
  pkg-config libnuma-dev numactl htop

# 4 GB swap — prevents OOM during 3B model load
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "Swap: 4 GB created"
fi

python3.12 -m venv ~/armforge_env
source ~/armforge_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# CPU PyTorch FIRST — prevents CUDA wheel resolution before sglang install
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

python3 -c "import torch; assert not torch.cuda.is_available(); print('PyTorch CPU OK:', torch.__version__)"

echo "=== CPU Features ==="
grep -m1 'Features' /proc/cpuinfo | tr ' ' '\n' | grep -E 'i8mm|sve|asimddp|asimd' || echo "baseline NEON only"
echo "Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "=== Bootstrap done ==="
```

---

## PHASE 1 — Build llama.cpp

### scripts/01_build_llamacpp.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Building llama.cpp (KleidiAI ON) ==="

cd ~
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git pull

FEATURES=$(grep -m1 'Features' /proc/cpuinfo 2>/dev/null || echo "")
if echo "$FEATURES" | grep -q "i8mm"; then
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+i8mm+dotprod"
  echo "Detected: i8mm → armv8.2-a+i8mm+dotprod"
elif echo "$FEATURES" | grep -q "asimddp"; then
  ARCH_FLAG="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"
  echo "Detected: dotprod → armv8.2-a+dotprod (Neoverse N1 path)"
else
  ARCH_FLAG="-DGGML_NATIVE=ON"
  echo "Baseline NEON — no dotprod detected"
fi

cmake -B build \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DCMAKE_BUILD_TYPE=Release \
  $ARCH_FLAG

cmake --build build -j$(nproc)
grep -i "KLEIDIAI" build/CMakeCache.txt | head -3
echo "KleidiAI build complete: $(ls ~/llama.cpp/build/bin/llama-server)"
```

### scripts/01b_build_baseline.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Building llama.cpp (KleidiAI OFF — baseline) ==="

cd ~/llama.cpp
cmake -B build_baseline \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_baseline -j$(nproc)
echo "Baseline build complete: $(ls ~/llama.cpp/build_baseline/bin/llama-server)"
```

---

## PHASE 2 — Download Models

### scripts/02_download_models.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
echo "=== Downloading Models ==="

mkdir -p ~/llama.cpp/models

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

MODELS_DIR = os.path.expanduser("~/llama.cpp/models")
downloads = [
    {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "dest": "main_model.gguf",
    },
    {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "dest": "draft_model.gguf",
    },
]
for m in downloads:
    dest = os.path.join(MODELS_DIR, m["dest"])
    if os.path.exists(dest):
        print(f"Already exists: {m['dest']} ({os.path.getsize(dest)/1e9:.2f} GB)")
        continue
    print(f"Downloading {m['file']} ...")
    tmp = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=MODELS_DIR)
    shutil.move(tmp, dest)
    print(f"  Saved: {m['dest']} ({os.path.getsize(dest)/1e9:.2f} GB)")
print("All models ready.")
EOF
```

---

## PHASE 3 — Thread Sweep (Run Before Any Benchmark)

### scripts/03_tune_threads.sh
```bash
#!/bin/bash
# CRITICAL: Run this before benchmarks. Neoverse N1 memory bandwidth saturates
# at low thread counts for large quantized models. Best thread count is often
# nproc/2, not nproc. This script finds the optimal count automatically.
set -e
source ~/armforge_env/bin/activate

LLAMA_CLI=~/llama.cpp/build/bin/llama-cli
MODEL=~/llama.cpp/models/main_model.gguf
PROMPT="Explain how ARM Neoverse processors accelerate AI inference in detail."
BEST_TPS=0
BEST_T=1
RESULTS_FILE=~/armforge/results/thread_sweep.txt
mkdir -p ~/armforge/results

echo "=== Thread Sweep ===" | tee $RESULTS_FILE
for T in 1 2 3 4; do
  [ $T -gt $(nproc) ] && continue
  echo -n "Threads=$T: "
  OUT=$($LLAMA_CLI \
    -m $MODEL \
    -t $T \
    -ngl 0 \
    --mlock \
    -n 64 \
    -p "$PROMPT" \
    --log-disable 2>&1 | grep "eval time")
  TPS=$(echo "$OUT" | grep -oP '\d+\.\d+ tokens per second' | head -1 | grep -oP '[\d.]+' || echo "0")
  echo "~$TPS tok/s" | tee -a $RESULTS_FILE
  if (( $(echo "$TPS > $BEST_TPS" | bc -l 2>/dev/null || echo 0) )); then
    BEST_TPS=$TPS
    BEST_T=$T
  fi
done

echo "OPTIMAL_THREADS=$BEST_T" | tee -a $RESULTS_FILE
echo "$BEST_T" > ~/armforge/results/optimal_threads.txt
echo "=== Optimal threads: $BEST_T (${BEST_TPS} tok/s) ==="
```

---

## PHASE 4 — Benchmark Scripts

### scripts/04_benchmark_baseline.sh
```bash
#!/bin/bash
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: Baseline (KleidiAI OFF, threads=$T) ==="

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build_baseline/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --mlock \
  -c 2048 \
  --host 0.0.0.0 --port 8001 &
SERVER_PID=$!
echo "Waiting 25s for model load..."
sleep 25

python benchmark/run_bench.py --mode baseline --port 8001

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Baseline done."
```

### scripts/05_benchmark_kleidiai.sh
```bash
#!/bin/bash
# KleidiAI only — NO speculative decoding
# Isolates KleidiAI contribution from speculative decoding contribution
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: KleidiAI only (threads=$T, batch=512) ==="

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build/bin/llama-server \
  -m ~/llama.cpp/models/main_model.gguf \
  -t $T \
  -ngl 0 \
  --mlock \
  -b 512 \
  -c 2048 \
  --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
echo "Waiting 25s for model load..."
sleep 25

python benchmark/run_bench.py --mode kleidiai --port 8000

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "KleidiAI benchmark done."
```

### scripts/06_benchmark_optimized.sh
```bash
#!/bin/bash
# KleidiAI + Speculative Decoding
# NOTE: On CPU, speculative decoding primarily reduces TTFT, not throughput.
# Throughput may be flat vs KleidiAI-only — this is expected and correct behavior.
set -e
source ~/armforge_env/bin/activate
cd ~/armforge

T=$(cat ~/armforge/results/optimal_threads.txt 2>/dev/null || echo $(nproc))
echo "=== Benchmark: KleidiAI + Speculative Decoding (threads=$T) ==="
echo "NOTE: Measuring TTFT reduction. Throughput gain vs KleidiAI-only is expected to be flat."

pkill -f llama-server 2>/dev/null || true; sleep 2

~/llama.cpp/build/bin/llama-server \
  -m  ~/llama.cpp/models/main_model.gguf \
  --model-draft ~/llama.cpp/models/draft_model.gguf \
  --spec-type draft-simple \
  --draft-max 5 \
  -t $T \
  -ngl 0 \
  --mlock \
  -b 512 \
  -c 2048 \
  --host 0.0.0.0 --port 8000 \
  --log-format json &
SERVER_PID=$!
echo "Waiting 30s for both models to load..."
sleep 30

python benchmark/run_bench.py --mode optimized --port 8000

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Optimized benchmark done."
```

### scripts/07_start_dashboard.sh
```bash
#!/bin/bash
source ~/armforge_env/bin/activate
cd ~/armforge
echo "Dashboard: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP'):8080"
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
```

---

## PHASE 5 — Python Source Files

### inference/arm_features.py
```python
"""Detect and report ARM64 CPU features."""
import os, platform

def detect():
    feats = {}
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('Features'):
                    parts = line.split(':')[1].strip().split()
                    feats = {
                        'i8mm':    'i8mm'    in parts,
                        'dotprod': 'asimddp' in parts,
                        'sve':     'sve'     in parts,
                        'sve2':    'sve2'    in parts,
                        'neon':    'asimd'   in parts,
                        'fp16':    'asimdhp' in parts,
                    }
                    break
    except Exception as e:
        feats = {'error': str(e)}
    feats['cores'] = os.cpu_count()
    feats['arch']  = platform.machine()
    return feats

def optimal_threads():
    """Read from thread sweep result; fallback to nproc-1."""
    try:
        with open(os.path.expanduser('~/armforge/results/optimal_threads.txt')) as f:
            return int(f.read().strip())
    except Exception:
        return max(1, os.cpu_count() - 1)

def report():
    f = detect()
    print("=== ARM64 Feature Report ===")
    for k, v in f.items():
        if isinstance(v, bool):
            print(f"  {'✓' if v else '✗'} {k}")
        else:
            print(f"  {k}: {v}")
    print(f"  Optimal threads: {optimal_threads()}")

if __name__ == '__main__':
    report()
```

### inference/speculative_server.py
```python
"""
Launch llama.cpp server with speculative decoding.

IMPORTANT: On CPU (no GPU), speculative decoding reduces TTFT via draft token
prefill overlap. Throughput (tok/s) gain vs KleidiAI-only is typically flat
or marginal — this is correct behavior, not a bug. The win is latency.

Verified flags (llama.cpp 2026 master):
  --model-draft (-md)  : draft model path
  --draft-max          : max draft tokens per step
  --spec-type          : must be 'draft-simple' for standalone draft model
  -ngl 0               : explicit CPU-only (no GPU offload)
  --mlock              : lock model weights in RAM, prevents swap thrash
  -b 512               : batch size — activates KleidiAI dotprod kernel paths
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from inference.arm_features import optimal_threads

LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MAIN_MODEL   = os.path.expanduser("~/llama.cpp/models/main_model.gguf")
DRAFT_MODEL  = os.path.expanduser("~/llama.cpp/models/draft_model.gguf")

def start(port=8000, context=2048, draft_max=5):
    threads = optimal_threads()

    for path, name in [(LLAMA_SERVER, "llama-server"),
                       (MAIN_MODEL,   "main_model.gguf"),
                       (DRAFT_MODEL,  "draft_model.gguf")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    cmd = [
        LLAMA_SERVER,
        "-m",            MAIN_MODEL,
        "--model-draft", DRAFT_MODEL,
        "--spec-type",   "draft-simple",
        "--draft-max",   str(draft_max),
        "-t",            str(threads),
        "-ngl",          "0",
        "--mlock",
        "-b",            "512",
        "-c",            str(context),
        "--host",        "0.0.0.0",
        "--port",        str(port),
        "--log-format",  "json",
    ]

    print(f"Starting speculative decoding server on :{port}")
    print(f"Threads: {threads} | Draft max: {draft_max} | mlock: ON | batch: 512")
    subprocess.run(cmd)

if __name__ == '__main__':
    start()
```

### inference/sglang_server.py
```python
"""
SGLang CPU server — BONUS DEMO ONLY.
Not included in primary benchmark comparison.
Requires separate install: pip install "sglang[srt]"

Known issues: Some 0.5.x builds crash on first inference request.
If this happens, skip and rely on llama.cpp results only.
"""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from inference.arm_features import optimal_threads

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # ungated, CPU-tested

def start(port=30000):
    threads = optimal_threads()
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",          MODEL_ID,
        "--device",              "cpu",
        "--dtype",               "float32",
        "--quantization",        "w8a8",
        "--host",                "0.0.0.0",
        "--port",                str(port),
        "--context-length",      "2048",
        "--mem-fraction-static", "0.7",
        "--log-level",           "info",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"]      = str(threads)
    env["MKL_NUM_THREADS"]      = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    print(f"Starting SGLang W8A8 CPU server on :{port} (bonus demo)")
    subprocess.run(cmd, env=env)

if __name__ == '__main__':
    start()
```

---

### benchmark/run_bench.py
```python
"""
Benchmark tokens/sec, TTFT, throughput.

Usage:
  python benchmark/run_bench.py --mode baseline  --port 8001
  python benchmark/run_bench.py --mode kleidiai  --port 8000
  python benchmark/run_bench.py --mode optimized --port 8000
  python benchmark/run_bench.py --mode sglang    --port 30000

WARMUP: One throwaway request is sent before measurement starts.
This ensures KV cache paths and JIT code are warm for fair comparison.
"""
import time, json, argparse, statistics, os
from datetime import datetime
import requests

PROMPTS = [
    "Explain how ARM Neoverse processors accelerate AI inference.",
    "What is speculative decoding and why does it reduce latency?",
    "Describe the difference between INT8 and FP16 quantization.",
    "How does the KleidiAI library improve matrix multiplication on ARM?",
    "Write a Python function that computes Fibonacci numbers efficiently.",
]

def check_server(base_url, timeout=5):
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code in (200, 404)
    except Exception:
        return False

def warmup(base_url):
    """Single warmup request — not recorded in results."""
    try:
        requests.post(f"{base_url}/v1/completions",
            json={"prompt": "warmup", "max_tokens": 5, "stream": False},
            timeout=60)
        print("  Warmup OK")
    except Exception as e:
        print(f"  Warmup failed (non-fatal): {e}")

def measure_ttft(base_url, prompt, max_tokens=5, timeout=60):
    """Time To First Token — streaming."""
    start = time.perf_counter()
    try:
        with requests.post(f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": True},
            stream=True, timeout=timeout) as r:
            for chunk in r.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    return time.perf_counter() - start, None
    except Exception as e:
        return None, str(e)
    return None, "no chunks received"

def measure_throughput(base_url, prompt, max_tokens=128, timeout=180):
    """Tokens per second — full generation, non-streaming."""
    start = time.perf_counter()
    try:
        r = requests.post(f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "stream": False},
            timeout=timeout)
        elapsed = time.perf_counter() - start
        data = r.json()
        tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
        return tokens / elapsed, None
    except Exception as e:
        return None, str(e)

def run(mode, port):
    base_url = f"http://localhost:{port}"
    print(f"\n=== ArmForge Benchmark [{mode.upper()}] → {base_url} ===\n")

    if not check_server(base_url):
        print(f"ERROR: No server at {base_url}. Start server first.")
        return None

    print("Running warmup request...")
    warmup(base_url)

    ttfts, tps_list = [], []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:55]}...")
        ttft, e1 = measure_ttft(base_url, prompt)
        tps,  e2 = measure_throughput(base_url, prompt)
        if ttft: ttfts.append(ttft); print(f"  TTFT:       {ttft*1000:.0f} ms")
        if tps:  tps_list.append(tps); print(f"  Throughput: {tps:.2f} tok/s")
        if e1 or e2: print(f"  Error: {e1 or e2}")

    if not tps_list:
        print("No successful measurements.")
        return None

    results = {
        "mode":        mode,
        "port":        port,
        "timestamp":   datetime.now().isoformat(),
        "platform":    "arm64",
        "avg_ttft_ms": round(statistics.mean(ttfts) * 1000, 1) if ttfts else None,
        "avg_tps":     round(statistics.mean(tps_list), 2),
        "min_tps":     round(min(tps_list), 2),
        "max_tps":     round(max(tps_list), 2),
        "samples":     len(PROMPTS),
    }

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/bench_{mode}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"  Avg TTFT:  {results['avg_ttft_ms']} ms")
    print(f"  Avg tok/s: {results['avg_tps']}")
    print(f"  Saved:     {fname}")
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
        choices=["baseline","kleidiai","optimized","sglang"])
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    run(args.mode, args.port)
```

### benchmark/compare.py
```python
"""
Three-row comparison: baseline / KleidiAI-only / KleidiAI+speculative.
Each row isolated to show individual contribution.

Usage: python benchmark/compare.py
"""
import json, glob
from tabulate import tabulate

def load_latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files: return None
    with open(files[-1]) as f: return json.load(f)

def pct(base, val):
    if base and val and base > 0:
        g = (val - base) / base * 100
        return f"{'+'if g>=0 else ''}{g:.0f}%"
    return "N/A"

def compare():
    baseline  = load_latest("results/bench_baseline_*.json")
    kleidiai  = load_latest("results/bench_kleidiai_*.json")
    optimized = load_latest("results/bench_optimized_*.json")
    sglang    = load_latest("results/bench_sglang_*.json")

    if not any([baseline, kleidiai, optimized]):
        print("No results found. Run benchmark scripts first.")
        return

    base_tps  = baseline.get("avg_tps") if baseline else None
    base_ttft = baseline.get("avg_ttft_ms") if baseline else None

    rows = []
    if baseline:
        rows.append([
            "① Baseline (vanilla llama.cpp, KleidiAI OFF)",
            f"{base_tps} tok/s", f"{base_ttft} ms", "—", "—"
        ])
    if kleidiai:
        rows.append([
            "② + KleidiAI dotprod kernels (no speculative)",
            f"{kleidiai.get('avg_tps')} tok/s",
            f"{kleidiai.get('avg_ttft_ms')} ms",
            pct(base_tps,  kleidiai.get("avg_tps")),
            pct(base_ttft, kleidiai.get("avg_ttft_ms")),
        ])
    if optimized:
        rows.append([
            "③ + KleidiAI + Speculative Decoding (TTFT focus)",
            f"{optimized.get('avg_tps')} tok/s",
            f"{optimized.get('avg_ttft_ms')} ms",
            pct(base_tps,  optimized.get("avg_tps")),
            pct(base_ttft, optimized.get("avg_ttft_ms")),
        ])
    if sglang:
        rows.append([
            "  SGLang W8A8 CPU [bonus]",
            f"{sglang.get('avg_tps')} tok/s",
            f"{sglang.get('avg_ttft_ms')} ms",
            pct(base_tps,  sglang.get("avg_tps")),
            "N/A",
        ])

    print("\n=== ArmForge — Optimization Breakdown ===\n")
    print(tabulate(rows,
        headers=["Configuration", "Throughput", "TTFT", "vs Baseline (tps)", "vs Baseline (ttft)"],
        tablefmt="github"))
    print("\nNOTE: Speculative decoding on CPU targets TTFT reduction, not throughput.")
    print("KleidiAI is the primary throughput optimization.")

if __name__ == '__main__':
    compare()
```

---

### dashboard/app.py
```python
"""
FastAPI dashboard — live metrics + benchmark results.
Run: uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import psutil, json, glob, os
from datetime import datetime

app = FastAPI(title="ArmForge Dashboard")
templates = Jinja2Templates(directory="dashboard/templates")

def get_arm_features():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('Features'):
                    return line.split(':')[1].strip().split()
    except Exception:
        pass
    return []

def load_results():
    out = []
    for fp in sorted(glob.glob("results/bench_*.json")):
        try:
            with open(fp) as f: out.append(json.load(f))
        except Exception:
            pass
    return out

def get_optimal_threads():
    try:
        with open('results/optimal_threads.txt') as f: return f.read().strip()
    except Exception:
        return "N/A"

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    features = get_arm_features()
    active   = [f for f in ["i8mm","asimddp","sve","asimd"] if f in features]
    mem      = psutil.virtual_memory()
    return templates.TemplateResponse("index.html", {
        "request":          request,
        "platform":         "arm64",
        "cores":            os.cpu_count(),
        "optimal_threads":  get_optimal_threads(),
        "cpu_pct":          round(psutil.cpu_percent(interval=0.5), 1),
        "mem_used_gb":      round(mem.used / 1e9, 1),
        "mem_total_gb":     round(mem.total / 1e9, 1),
        "arm_features":     active,
        "results":          load_results()[-8:],
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.get("/api/metrics")
async def metrics():
    mem = psutil.virtual_memory()
    return {
        "platform":         "arm64",
        "arch":             os.uname().machine,
        "cpu_count":        os.cpu_count(),
        "optimal_threads":  get_optimal_threads(),
        "cpu_pct":          round(psutil.cpu_percent(interval=0.5), 1),
        "mem_used_gb":      round(mem.used / 1e9, 2),
        "mem_total_gb":     round(mem.total / 1e9, 2),
        "arm_features":     get_arm_features(),
        "bench_results":    load_results(),
        "timestamp":        datetime.now().isoformat(),
    }
```

### dashboard/templates/index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>ArmForge Dashboard</title>
<style>
*{box-sizing:border-box}
body{font-family:monospace;background:#0d1117;color:#e6edf3;padding:2rem;margin:0}
h1{color:#58a6ff;margin:0 0 4px}
h2{color:#58a6ff;font-size:1rem;margin:1.5rem 0 .5rem}
.sub{color:#8b949e;font-size:.85rem;margin:0 0 1.5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:1.5rem}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;text-align:center}
.val{font-size:1.8rem;color:#3fb950;font-weight:bold}
.lbl{font-size:.75rem;color:#8b949e;margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{border:1px solid #30363d;padding:.4rem .75rem;text-align:left}
th{background:#161b22;color:#58a6ff}
.tag{display:inline-block;background:#1f6feb22;color:#58a6ff;padding:2px 8px;border-radius:4px;font-size:.75rem;margin:2px}
.g{color:#3fb950}.m{color:#8b949e}.y{color:#e3b341}
a{color:#58a6ff;text-decoration:none}
code{background:#161b22;padding:2px 6px;border-radius:4px;font-size:.8rem}
.note{background:#1f2937;border-left:3px solid #e3b341;padding:.5rem 1rem;font-size:.8rem;margin:1rem 0;color:#e3b341}
</style>
</head>
<body>
<h1>🦾 ArmForge — ARM64 Inference Dashboard</h1>
<p class="sub">{{ platform }} · {{ cores }} cores · Optimal threads: {{ optimal_threads }} · Updated: {{ timestamp }}</p>

<div class="grid">
  <div class="card"><div class="val">{{ cpu_pct }}%</div><div class="lbl">CPU</div></div>
  <div class="card"><div class="val">{{ mem_used_gb }}G</div><div class="lbl">RAM / {{ mem_total_gb }}G</div></div>
  <div class="card"><div class="val">{{ cores }}</div><div class="lbl">ARM64 cores</div></div>
  <div class="card"><div class="val">{{ optimal_threads }}</div><div class="lbl">Optimal threads</div></div>
</div>

<p>ARM features:
{% for f in arm_features %}<span class="tag">{{ f }}</span>
{% else %}<span class="m">none detected</span>{% endfor %}
</p>

<div class="note">
  ℹ️ <strong>KleidiAI</strong> is the primary throughput optimization (dotprod kernels).
  <strong>Speculative decoding</strong> reduces TTFT (latency) on CPU — throughput gain vs KleidiAI-only is expected to be flat.
</div>

<h2>Benchmark Results</h2>
{% if results %}
<table>
  <tr><th>Mode</th><th>Avg tok/s</th><th>TTFT (ms)</th><th>Samples</th><th>Timestamp</th></tr>
  {% for r in results %}
  <tr>
    <td>{{ r.mode }}</td>
    <td class="g">{{ r.avg_tps or "—" }}</td>
    <td {% if r.avg_ttft_ms and r.avg_ttft_ms < 500 %}class="g"{% else %}class="y"{% endif %}>
      {{ r.avg_ttft_ms or "—" }}</td>
    <td>{{ r.samples }}</td>
    <td class="m" style="font-size:.8rem">{{ r.timestamp[:16] }}</td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p class="m">No results yet. Run in order:<br>
<code>bash scripts/03_tune_threads.sh</code><br>
<code>bash scripts/04_benchmark_baseline.sh</code><br>
<code>bash scripts/05_benchmark_kleidiai.sh</code><br>
<code>bash scripts/06_benchmark_optimized.sh</code><br>
<code>python benchmark/compare.py</code>
</p>
{% endif %}

<p style="margin-top:2rem;font-size:.8rem;color:#8b949e">
  <a href="/api/metrics">/api/metrics (JSON)</a> ·
  ArmForge · ARM AI Optimization Challenge 2026 · Apache 2.0
</p>
</body>
</html>
```

---

## PHASE 6 — Full Run Order

```bash
# ── On local machine ──────────────────────────────────
git init armforge && cd armforge
# Write all files as defined in this spec (agent does this)
git add .
git commit -m "ArmForge v3 — corrected build"
git remote add origin https://github.com/YOUR_USERNAME/armforge.git
git push -u origin main

# ── On Oracle A1 (SSH in, aarch64) ────────────────────
git clone https://github.com/YOUR_USERNAME/armforge
cd armforge

bash scripts/00_bootstrap.sh           # ~5 min — deps, swap, venv
bash scripts/01_build_llamacpp.sh      # ~8 min — KleidiAI build
bash scripts/01b_build_baseline.sh     # ~8 min — baseline build (no KleidiAI)
bash scripts/02_download_models.sh     # ~5 min — 3B + 1B models

# CRITICAL: Run thread sweep before any benchmark
bash scripts/03_tune_threads.sh        # ~3 min — writes optimal_threads.txt

bash scripts/04_benchmark_baseline.sh  # ~4 min — vanilla llama.cpp
bash scripts/05_benchmark_kleidiai.sh  # ~4 min — KleidiAI only
bash scripts/06_benchmark_optimized.sh # ~6 min — KleidiAI + speculative

# View isolated comparison
source ~/armforge_env/bin/activate
python benchmark/compare.py

# Dashboard (keep terminal open)
bash scripts/07_start_dashboard.sh
# → http://YOUR_IP:8080

# ── Optional SGLang bonus ──────────────────────────────
pip install "sglang[srt]"
python inference/sglang_server.py &    # wait ~60s
python benchmark/run_bench.py --mode sglang --port 30000
python benchmark/compare.py
```

---

## Expected Results (Neoverse N1, 2 OCPU)

| Configuration | Expected tok/s | Expected TTFT |
|---|---|---|
| ① Baseline | 4–7 tok/s | 600–900 ms |
| ② + KleidiAI | 6–10 tok/s (+30–50%) | 500–700 ms |
| ③ + KleidiAI + Speculative | ~same as ② | 300–500 ms (−30–40%) |

**Speculative decoding win on CPU is TTFT, not throughput. This is correct and expected.**

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| `--draft-model` not found | Use `--model-draft` |
| `--draft N` not found | Use `--draft-max N` |
| `--spec-type` missing | Add `--spec-type draft-simple` |
| SGLang CUDA build installed | Install CPU torch first (done in 00_bootstrap.sh) |
| SGLang `--dtype int8` rejected | Use `--quantization w8a8` |
| OOM during model load | 4 GB swap created in 00_bootstrap.sh |
| Port already in use | `pkill -f llama-server && sleep 2` |
| KleidiAI not in CMakeCache | `grep KLEIDIAI ~/llama.cpp/build/CMakeCache.txt` — rebuild if missing |
| Oracle A1 no capacity | Try ap-singapore-1, eu-frankfurt-1, ap-tokyo-1 |
| Neoverse N1 = no i8mm | Has dotprod — script auto-detects, uses armv8.2-a+dotprod |
| Benchmark first result slow | Warmup call added in run_bench.py |
| Thread count not optimal | Run 03_tune_threads.sh before any benchmark |
| Model swapped out mid-bench | `--mlock` added to all server calls |

---

## Validation Commands

```bash
uname -m                                      # → aarch64
python3 inference/arm_features.py             # feature report + optimal threads
grep KLEIDIAI ~/llama.cpp/build/CMakeCache.txt # → GGML_CPU_KLEIDIAI:BOOL=ON
cat results/optimal_threads.txt               # → 1, 2, 3, or 4
curl http://localhost:8080/api/metrics | python3 -m json.tool
python benchmark/compare.py                   # three-row table
```

---

*ArmForge v3 · Apache 2.0 · $0 infrastructure · ARM AI Optimization Challenge 2026*
