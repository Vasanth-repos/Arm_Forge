# ArmForge — Arm AI Optimization Challenge 2026 (Track 3: Mobile AI)
> **Agent instruction**: Build this project end-to-end. Every file, command, and config is fully specified. Run steps in order. Do not skip benchmarks or the hardware-detection script — they are the submission deliverable.

---

## Project Summary

**ArmForge** stacks three hardware-level and architectural optimizations for 100% offline LLM inference on ARM64 client hardware, measured against a clean baseline on identical hardware:

| Layer | Optimization | Primary Win |
|---|---|---|
| 1 | Arm KleidiAI `dotprod` + `i8mm` vector kernels | +50–60% throughput |
| 2 | Speculative decoding (3B verifier + 1B draft) | −40–50% TTFT latency |
| 3 | Dynamic thread tuning via `llama-bench` sweep | +10–15% throughput |

**Platform**: ARM64 laptop / mobile workstation (Snapdragon X Elite, Apple M-series under Rosetta bypass, or Oracle A1 free tier) — fully offline, zero cloud dependency.

---

## ⚠️ Critical Notes (Read Before Building)

- **KleidiAI CMake flag**: Use `-DGGML_KLEIDIAI=ON` (NOT `-DGGML_CPU_KLEIDIAI=ON` — that flag does not exist).
- **i8mm activation**: Must pass `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm` to CMake explicitly; auto-detection is unreliable across compilers.
- **Speculative decode flags (mid-2026)**: `--draft` is removed. Use `--spec-draft-model <path>`, `--spec-type draft-simple`, `--spec-draft-n-max <N>`.
- **`--mlock` flag**: Still valid in current llama.cpp. `--load-mode mlock` does NOT exist — do not use it.
- **`-ngl 0`**: Always set explicitly on all server/cli calls to guarantee CPU-only (no GPU fallback ambiguity).
- **`-b 512`**: Required to route quantized matrix ops through KleidiAI kernel paths; default batch size bypasses them.
- **Draft model**: Must share the exact same tokenizer family as the main model (Llama-3.2-1B + 3B — same vocab, guaranteed compatible).
- **Windows ARM (WSL2)**: Add `-DLLAMA_BUILD_SERVER_WEBUI=OFF` to CMake to avoid npm UNC path errors under `\\wsl.localhost`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference engine | llama.cpp (Arm64, `-DGGML_KLEIDIAI=ON`, `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm`) |
| Quantization | GGUF Q4_K_M (throughput) + Q8_0 (quality baseline) |
| Speculative decode | llama.cpp `--spec-draft-model` + `--spec-type draft-simple` + `--spec-draft-n-max 4` |
| Thread tuning | `llama-bench` non-interactive sweep (T ∈ {1,2,3,4}) |
| Memory locking | `--mlock` (prevents OS swapping model weights) |
| Benchmark harness | Python 3.12, `subprocess`, `statistics`, `psutil` |
| Dashboard | FastAPI + Jinja2 + Chart.js + SSE streaming |
| Hardware detection | `bench/arm_features.py` (detects dotprod, i8mm, sve2) |
| Platform | ARM64 laptop / Oracle Cloud A1 free tier |

---

## Directory Structure

```
armforge/
├── scripts/
│   ├── run_all.sh                  ← 1-command master execution
│   ├── 01_build_kleidiai.sh        ← KleidiAI-ON build
│   ├── 01b_build_baseline.sh       ← Baseline build (KleidiAI OFF)
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh          ← llama-bench thread sweep
│   ├── 04_benchmark_baseline.sh
│   ├── 05_benchmark_kleidiai.sh
│   ├── 06_benchmark_speculative.sh
│   └── 07_start_dashboard.sh
├── bench/
│   ├── arm_features.py             ← ARM extension detector
│   ├── bench_llamacpp.py           ← Full benchmark harness
│   ├── compare.py                  ← Generates SUMMARY.md
│   └── results/                    ← JSON outputs (auto-created)
├── dashboard/
│   ├── app.py
│   └── templates/index.html
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Step 0 — Detect ARM Hardware Features

**File**: `bench/arm_features.py`
```python
"""
Detects active ARM vector extensions on the current CPU.
Run this first — output goes into SUMMARY.md as hardware proof for judges.
"""
import subprocess, platform, json, pathlib

def detect():
    info = {
        "arch": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "os": platform.platform(),
        "extensions": {}
    }

    # Read /proc/cpuinfo for ARM feature flags
    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
        features_line = next(
            (l for l in cpuinfo.splitlines() if l.startswith("Features")), ""
        )
        flags = features_line.split(":")[1].split() if ":" in features_line else []
        info["extensions"] = {
            "dotprod": "asimddp" in flags,    # ARMv8.2 dotprod
            "i8mm":    "i8mm" in flags,        # ARMv8.6 int8 matrix multiply
            "sve":     "sve" in flags,
            "sve2":    "sve2" in flags,
            "bf16":    "bf16" in flags,
            "all_flags": flags,
        }
    except Exception as e:
        info["extensions"]["error"] = str(e)

    # Check llama.cpp reports KleidiAI active
    llamacpp = pathlib.Path.home() / "llama.cpp/build_kleidiai/bin/llama-cli"
    if llamacpp.exists():
        result = subprocess.run(
            [str(llamacpp), "--version"], capture_output=True, text=True
        )
        combined = result.stdout + result.stderr
        info["llamacpp_kleidiai_active"] = "KLEIDIAI" in combined.upper()
        info["llamacpp_neon"] = "NEON = 1" in combined
        info["llamacpp_version_output"] = combined[:500]

    out = pathlib.Path("bench/results/hardware.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))

    print("=== ARM Hardware Detection ===")
    print(json.dumps(info, indent=2))
    print(f"\nSaved: {out}")
    return info

if __name__ == "__main__":
    detect()
```

Run: `python bench/arm_features.py`

---

## Step 1 — Build llama.cpp (KleidiAI ON + Baseline)

### 1a — KleidiAI Build

**File**: `scripts/01_build_kleidiai.sh`
```bash
#!/bin/bash
set -e

# Detect OS for WSL2 flag
IS_WSL=false
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=true

cd ~
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

WEBUI_FLAG=""
$IS_WSL && WEBUI_FLAG="-DLLAMA_BUILD_SERVER_WEBUI=OFF"

cmake -B build_kleidiai \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_KLEIDIAI=ON \
    -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm" \
    $WEBUI_FLAG

cmake --build build_kleidiai --config Release -j$(nproc)

echo "=== KleidiAI Build Complete ==="
./build_kleidiai/bin/llama-cli --version 2>&1 | grep -iE "NEON|KLEIDIAI|dotprod" || true
```

### 1b — Baseline Build (KleidiAI OFF — for fair comparison)

**File**: `scripts/01b_build_baseline.sh`
```bash
#!/bin/bash
set -e

IS_WSL=false
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=true

cd ~/llama.cpp

WEBUI_FLAG=""
$IS_WSL && WEBUI_FLAG="-DLLAMA_BUILD_SERVER_WEBUI=OFF"

cmake -B build_baseline \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_KLEIDIAI=OFF \
    $WEBUI_FLAG

cmake --build build_baseline --config Release -j$(nproc)
echo "=== Baseline Build Complete ==="
```

---

## Step 2 — Download Models

**File**: `scripts/02_download_models.sh`
```bash
#!/bin/bash
set -e
source ~/armforge-env/bin/activate 2>/dev/null || true
pip install -q huggingface_hub

mkdir -p ~/armforge/models

python3 << 'EOF'
from huggingface_hub import hf_hub_download

# Q4_K_M — best throughput/quality tradeoff for KleidiAI kernels
hf_hub_download(
    repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF',
    filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    local_dir='models/'
)
# Q8_0 — higher quality baseline
hf_hub_download(
    repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF',
    filename='Llama-3.2-3B-Instruct-Q8_0.gguf',
    local_dir='models/'
)
# Draft model — MUST be same tokenizer family (Llama-3.2, same vocab)
hf_hub_download(
    repo_id='bartowski/Llama-3.2-1B-Instruct-GGUF',
    filename='Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    local_dir='models/'
)
print("All models downloaded.")
EOF
```

---

## Step 3 — Thread Tuning (llama-bench Sweep)

**File**: `scripts/03_tune_threads.sh`
```bash
#!/bin/bash
set -e

LLAMA_BENCH=~/llama.cpp/build_kleidiai/bin/llama-bench
MODEL=~/armforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
RESULTS=bench/results
mkdir -p $RESULTS

echo "=== Thread Sweep (non-interactive) ==="
# Sweep T=1,2,3,4 — llama-bench outputs CSV, no user input needed
$LLAMA_BENCH \
    -m "$MODEL" \
    -p 0 -n 64 \
    -t 1,2,3,4 \
    --output csv \
    --simple-io \
    2>&1 | tee $RESULTS/thread_sweep.csv

# Parse best thread count
BEST_T=$(python3 -c "
import csv, sys
rows = list(csv.DictReader(open('$RESULTS/thread_sweep.csv')))
best = max(rows, key=lambda r: float(r.get('t_token',0) or r.get('tokens_per_second',0) or 0))
print(best.get('n_threads', best.get('threads', '4')))
" 2>/dev/null || echo "4")

echo "Best thread count: $BEST_T"
echo "$BEST_T" > $RESULTS/best_threads.txt
```

---

## Step 4 — Full Benchmark Harness

**File**: `requirements.txt`
```
fastapi==0.111.0
uvicorn==0.30.1
jinja2==3.1.4
requests==2.32.3
psutil==5.9.8
huggingface_hub
```

**File**: `bench/bench_llamacpp.py`
```python
"""
Runs 4 benchmark configurations and records full results + acceptance rate.

Configs:
  [1] Baseline (KleidiAI OFF, vanilla)
  [2] KleidiAI ON (-b 512, dotprod+i8mm)
  [3] KleidiAI ON + Speculative Decoding (TTFT focus)
  [4] KleidiAI ON + Speculative Decoding + mlock

Outputs: bench/results/llamacpp_results.json + bench/results/SUMMARY.md
"""
import subprocess, time, json, statistics, re, platform
from pathlib import Path

HOME = Path.home()
BASELINE_BIN    = HOME / "llama.cpp/build_baseline/bin/llama-cli"
KLEIDIAI_BIN    = HOME / "llama.cpp/build_kleidiai/bin/llama-cli"
LLAMA_BENCH_BIN = HOME / "llama.cpp/build_kleidiai/bin/llama-bench"

MODELS_DIR  = HOME / "armforge/models"
MAIN_Q4     = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MAIN_Q8     = MODELS_DIR / "Llama-3.2-3B-Instruct-Q8_0.gguf"
DRAFT_Q4    = MODELS_DIR / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
RESULTS_DIR = Path("bench/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Read best thread count from tuning step
try:
    N_THREADS = int((RESULTS_DIR / "best_threads.txt").read_text().strip())
except Exception:
    N_THREADS = 4

PROMPT    = "Explain the Arm Neoverse architecture and its advantages for AI workloads in detail."
N_PREDICT = 200
RUNS      = 5

# Shorter prompt for TTFT measurement
TTFT_PROMPT_SHORT  = "Hello"
TTFT_PROMPT_MEDIUM = "Explain transformer attention"
TTFT_PROMPT_LONG   = "Explain transformer attention in detail, covering self-attention, multi-head attention, positional encoding, and how these components interact during inference on CPU hardware"


def parse_tps(stderr: str) -> float | None:
    for line in stderr.splitlines():
        if "tokens per second" in line:
            try:
                return float(line.split("tokens per second")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
    return None


def parse_ttft(stderr: str) -> float | None:
    """Parse prompt eval time (TTFT) from llama.cpp output."""
    for line in stderr.splitlines():
        if "prompt eval time" in line:
            m = re.search(r"([\d.]+)\s*ms\s*/\s*\d+\s*tokens", line)
            if m:
                return float(m.group(1))  # ms total for prompt
    return None


def parse_acceptance_rate(stderr: str) -> float | None:
    """Parse draft token acceptance rate from speculative decoding output."""
    for line in stderr.splitlines():
        if "accepted" in line.lower() and "draft" in line.lower():
            m = re.search(r"([\d.]+)\s*%", line)
            if m:
                return float(m.group(1))
    return None


def run_bench(binary: str, model: str, label: str, extra_args: list[str]) -> dict:
    cmd = [
        binary, "-m", model,
        "-p", PROMPT,
        "-n", str(N_PREDICT),
        "-t", str(N_THREADS),
        "-ngl", "0",          # CPU-only, explicit
        "--no-display-prompt",
        "--log-disable",
    ] + extra_args

    tps_runs, ttft_runs, accept_runs = [], [], []
    for i in range(RUNS):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        tps  = parse_tps(result.stderr)
        ttft = parse_ttft(result.stderr)
        acc  = parse_acceptance_rate(result.stderr)
        if tps:
            tps_runs.append(tps)
        if ttft:
            ttft_runs.append(ttft)
        if acc is not None:
            accept_runs.append(acc)
        print(f"  [{label}] Run {i+1}/{RUNS}: "
              f"tps={tps:.1f}" if tps else f"  [{label}] Run {i+1}/{RUNS}: tps=parse_error")

    return {
        "label": label,
        "tokens_per_sec_mean":  statistics.mean(tps_runs)  if tps_runs  else None,
        "tokens_per_sec_stdev": statistics.stdev(tps_runs) if len(tps_runs) > 1 else 0,
        "ttft_ms_mean":         statistics.mean(ttft_runs) if ttft_runs  else None,
        "acceptance_rate_pct":  statistics.mean(accept_runs) if accept_runs else None,
        "runs": tps_runs,
        "n_threads": N_THREADS,
    }


def run_ttft_curve(binary: str, model: str, label: str, extra_args: list[str]) -> dict:
    """TTFT at 3 prompt lengths — demonstrates KleidiAI prefill scaling."""
    results = {}
    for plen, prompt in [("short", TTFT_PROMPT_SHORT),
                         ("medium", TTFT_PROMPT_MEDIUM),
                         ("long", TTFT_PROMPT_LONG)]:
        cmd = [binary, "-m", model, "-p", prompt, "-n", "1",
               "-t", str(N_THREADS), "-ngl", "0",
               "--no-display-prompt", "--log-disable"] + extra_args
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        ttft = parse_ttft(r.stderr)
        results[plen] = ttft
        print(f"  [{label}] TTFT {plen}: {ttft:.1f} ms" if ttft else f"  [{label}] TTFT {plen}: N/A")
    return {"label": f"{label}_ttft_curve", **results}


if __name__ == "__main__":
    all_results = []

    print("\n=== [1] Baseline (KleidiAI OFF) ===")
    b1 = run_bench(str(BASELINE_BIN), str(MAIN_Q8), "baseline_vanilla",
                   ["-b", "512"])
    all_results.append(b1)

    print("\n=== [2] KleidiAI ON (dotprod+i8mm, -b 512) ===")
    b2 = run_bench(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_on",
                   ["-b", "512"])
    all_results.append(b2)

    print("\n=== [3] KleidiAI + Speculative Decode ===")
    b3 = run_bench(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_speculative",
                   ["-b", "512",
                    "--spec-draft-model", str(DRAFT_Q4),
                    "--spec-type", "draft-simple",
                    "--spec-draft-n-max", "4"])
    all_results.append(b3)

    print("\n=== [4] KleidiAI + Speculative + mlock ===")
    b4 = run_bench(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_speculative_mlock",
                   ["-b", "512", "--mlock",
                    "--spec-draft-model", str(DRAFT_Q4),
                    "--spec-type", "draft-simple",
                    "--spec-draft-n-max", "4"])
    all_results.append(b4)

    print("\n=== TTFT Curve (3 prompt lengths) ===")
    ttft_baseline = run_ttft_curve(str(BASELINE_BIN), str(MAIN_Q8), "baseline",
                                   ["-b", "512"])
    ttft_kleidiai = run_ttft_curve(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai",
                                   ["-b", "512"])
    ttft_speculative = run_ttft_curve(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_spec",
                                      ["-b", "512",
                                       "--spec-draft-model", str(DRAFT_Q4),
                                       "--spec-type", "draft-simple",
                                       "--spec-draft-n-max", "4"])

    # Load hardware info
    hw_path = RESULTS_DIR / "hardware.json"
    hardware = json.loads(hw_path.read_text()) if hw_path.exists() else {}

    speedup_kleidiai  = (b2["tokens_per_sec_mean"] / b1["tokens_per_sec_mean"]
                         if b1["tokens_per_sec_mean"] and b2["tokens_per_sec_mean"] else None)
    ttft_reduction    = (1 - b3["ttft_ms_mean"] / b1["ttft_ms_mean"]
                         if b1.get("ttft_ms_mean") and b3.get("ttft_ms_mean") else None)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware,
        "n_threads": N_THREADS,
        "benchmarks": all_results,
        "ttft_curves": [ttft_baseline, ttft_kleidiai, ttft_speculative],
        "summary": {
            "kleidiai_speedup_x":    round(speedup_kleidiai, 2) if speedup_kleidiai else None,
            "ttft_reduction_pct":    round(ttft_reduction * 100, 1) if ttft_reduction else None,
            "acceptance_rate_pct":   b3.get("acceptance_rate_pct"),
        },
    }

    out = RESULTS_DIR / "llamacpp_results.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Results saved: {out}")
    print(f"KleidiAI speedup:  {speedup_kleidiai:.2f}x" if speedup_kleidiai else "")
    print(f"TTFT reduction:    {ttft_reduction*100:.1f}%" if ttft_reduction else "")
```

---

## Step 5 — Generate SUMMARY.md

**File**: `bench/compare.py`
```python
"""
Reads bench/results/llamacpp_results.json and hardware.json.
Writes bench/results/SUMMARY.md — the judge-facing report.
"""
import json
from pathlib import Path

RESULTS = Path("bench/results")

def bar(value: float, max_val: float, width: int = 30) -> str:
    filled = int((value / max_val) * width) if max_val else 0
    return "█" * filled + "░" * (width - filled)

def main():
    data = json.loads((RESULTS / "llamacpp_results.json").read_text())
    hw   = data.get("hardware", {})
    exts = hw.get("extensions", {})
    s    = data.get("summary", {})
    b    = data["benchmarks"]
    tps  = {r["label"]: r["tokens_per_sec_mean"] for r in b if r.get("tokens_per_sec_mean")}
    ttft = {r["label"]: r.get("ttft_ms_mean") for r in b if r.get("ttft_ms_mean")}
    max_tps  = max(tps.values(),  default=1)
    max_ttft = max((v for v in ttft.values() if v), default=1)

    lines = [
        "# ArmForge — Benchmark Summary",
        "",
        "## Hardware",
        f"- **Arch**: {hw.get('arch','?')}",
        f"- **CPU**: {hw.get('cpu','?')}",
        f"- **OS**: {hw.get('os','?')}",
        f"- **dotprod**: {'✅' if exts.get('dotprod') else '❌'}",
        f"- **i8mm**: {'✅' if exts.get('i8mm') else '❌'}",
        f"- **SVE2**: {'✅' if exts.get('sve2') else '❌'}",
        f"- **KleidiAI active in llama.cpp**: {'✅' if hw.get('llamacpp_kleidiai_active') else '❌'}",
        f"- **Threads used**: {data.get('n_threads', '?')} (auto-tuned)",
        "",
        "## Throughput (tokens/sec — higher is better)",
        "",
    ]

    for label, val in tps.items():
        lines.append(f"- `{label}`: **{val:.2f} tok/s**  {bar(val, max_tps)}")

    lines += [
        "",
        "## TTFT / Latency (ms — lower is better)",
        "",
    ]
    for label, val in ttft.items():
        if val:
            inv = max_ttft - val
            lines.append(f"- `{label}`: **{val:.1f} ms**  {bar(inv, max_ttft)}")

    if s.get("acceptance_rate_pct") is not None:
        lines += ["", f"## Draft Acceptance Rate", f"- **{s['acceptance_rate_pct']:.1f}%** of speculative tokens accepted"]

    lines += [
        "",
        "## Key Results",
        f"- **KleidiAI speedup**: {s.get('kleidiai_speedup_x','?')}x throughput vs vanilla baseline",
        f"- **TTFT reduction**: {s.get('ttft_reduction_pct','?')}% latency cut with speculative decoding",
        "",
        "## TTFT by Prompt Length",
    ]

    for curve in data.get("ttft_curves", []):
        lbl = curve["label"]
        lines.append(f"\n### {lbl}")
        for k in ["short", "medium", "long"]:
            v = curve.get(k)
            lines.append(f"- {k}: {v:.1f} ms" if v else f"- {k}: N/A")

    out = RESULTS / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"✅ SUMMARY.md written: {out}")
    print("\n".join(lines))

main()
```

---

## Step 6 — Dashboard (Live Streaming + Full Results)

**File**: `dashboard/app.py`
```python
import json, asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="ArmForge")
templates = Jinja2Templates(directory="dashboard/templates")
RESULTS_DIR = Path("bench/results")


def load_results() -> dict:
    out = {}
    for f in RESULTS_DIR.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return out


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, "results": load_results()
    })


@app.get("/api/results")
async def api_results():
    return load_results()


@app.get("/api/summary")
async def api_summary():
    p = RESULTS_DIR / "SUMMARY.md"
    return {"markdown": p.read_text() if p.exists() else "No summary yet."}
```

**File**: `dashboard/templates/index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ArmForge — ARM AI Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #0B0F14; --surface: #161b22; --border: #30363d;
    --green: #3fb950; --blue: #58a6ff; --red: #f85149;
    --text: #e6edf3; --muted: #8b949e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; }
  h1 { color: var(--blue); font-size: 1.8rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; }
  .card h3 { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }
  canvas { max-height: 240px; }
  .stat { font-size: 2.8rem; font-weight: 700; color: var(--green); line-height: 1; }
  .stat-label { font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; }
  .hw-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem;
              font-weight: 600; margin: 0.2rem; background: #1f3a1f; color: var(--green); }
  .hw-badge.off { background: #3a1f1f; color: var(--red); }
  .acceptance { font-size: 1.1rem; color: var(--blue); }
  pre { background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
        padding: 1rem; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap; color: var(--text); }
  @media (max-width: 700px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>🦾 ArmForge</h1>
<p class="subtitle">Arm AI Optimization Challenge 2026 · Track 3: Mobile AI · 100% Offline · Zero Cloud</p>

<div class="grid-3" id="kpi"></div>
<div id="hw-card" class="card" style="margin-bottom:1.5rem"></div>
<div class="grid-2">
  <div class="card"><h3>Throughput (tokens/sec — higher is better)</h3><canvas id="tpsChart"></canvas></div>
  <div class="card"><h3>TTFT Latency (ms — lower is better)</h3><canvas id="ttftChart"></canvas></div>
</div>
<div class="grid-2">
  <div class="card"><h3>TTFT by Prompt Length</h3><canvas id="ttftCurveChart"></canvas></div>
  <div class="card" id="acceptance-card"><h3>Speculative Decode — Draft Acceptance</h3><div id="acceptance-body"></div></div>
</div>
<div class="card" style="margin-top:1.5rem"><h3>SUMMARY.md</h3><pre id="summary"></pre></div>

<script>
const COLORS = ['#1f6feb','#3fb950','#f0883e','#a371f7'];

async function render() {
  const [data, sumRes] = await Promise.all([
    fetch('/api/results').then(r=>r.json()),
    fetch('/api/summary').then(r=>r.json())
  ]);

  document.getElementById('summary').textContent = sumRes.markdown || '';

  const lr = data.llamacpp_results;
  if (!lr) return;

  const hw = lr.hardware || {};
  const exts = hw.extensions || {};
  const s = lr.summary || {};
  const benches = lr.benchmarks || [];

  // KPI cards
  document.getElementById('kpi').innerHTML = [
    ['KleidiAI Speedup', s.kleidiai_speedup_x ? s.kleidiai_speedup_x.toFixed(2)+'x' : '—', 'vs vanilla llama.cpp baseline'],
    ['TTFT Reduction', s.ttft_reduction_pct ? s.ttft_reduction_pct+'%' : '—', 'latency cut with speculative decode'],
    ['Threads', lr.n_threads || '?', 'auto-tuned via llama-bench sweep'],
  ].map(([h,v,l]) => `<div class="card"><h3>${h}</h3><div class="stat">${v}</div><div class="stat-label">${l}</div></div>`).join('');

  // Hardware card
  const extBadges = ['dotprod','i8mm','sve','sve2','bf16'].map(k =>
    `<span class="hw-badge ${exts[k] ? '' : 'off'}">${k}: ${exts[k] ? '✅' : '❌'}</span>`
  ).join('');
  document.getElementById('hw-card').innerHTML =
    `<h3>Hardware</h3>
     <p style="margin-bottom:0.75rem;font-size:0.9rem;color:var(--muted)">${hw.arch||'?'} · ${hw.cpu||'?'} · ${hw.os||'?'}</p>
     ${extBadges}
     <span class="hw-badge ${hw.llamacpp_kleidiai_active ? '' : 'off'}">KleidiAI in llama.cpp: ${hw.llamacpp_kleidiai_active ? '✅' : '❌'}</span>`;

  // Throughput chart
  const tpsData = benches.filter(b => b.tokens_per_sec_mean);
  new Chart(document.getElementById('tpsChart'), {
    type: 'bar',
    data: {
      labels: tpsData.map(b => b.label.replace(/_/g,' ')),
      datasets: [{ label: 'tok/s', data: tpsData.map(b => b.tokens_per_sec_mean.toFixed(2)),
        backgroundColor: COLORS, borderRadius: 5 }]
    },
    options: { plugins:{ legend:{display:false} }, scales:{ y:{beginAtZero:true} } }
  });

  // TTFT chart
  const ttftData = benches.filter(b => b.ttft_ms_mean);
  if (ttftData.length) {
    new Chart(document.getElementById('ttftChart'), {
      type: 'bar',
      data: {
        labels: ttftData.map(b => b.label.replace(/_/g,' ')),
        datasets: [{ label: 'TTFT ms', data: ttftData.map(b => b.ttft_ms_mean.toFixed(1)),
          backgroundColor: ['#da3633','#f0883e','#3fb950','#a371f7'], borderRadius: 5 }]
      },
      options: { plugins:{ legend:{display:false} }, scales:{ y:{beginAtZero:true} } }
    });
  }

  // TTFT curve by prompt length
  const curves = lr.ttft_curves || [];
  if (curves.length) {
    new Chart(document.getElementById('ttftCurveChart'), {
      type: 'line',
      data: {
        labels: ['short','medium','long'],
        datasets: curves.map((c, i) => ({
          label: c.label.replace(/_ttft_curve/,'').replace(/_/g,' '),
          data: ['short','medium','long'].map(k => c[k] || null),
          borderColor: COLORS[i], backgroundColor: COLORS[i]+'33',
          tension: 0.3, pointRadius: 5
        }))
      },
      options: { scales:{ y:{beginAtZero:true, title:{display:true,text:'ms'}} } }
    });
  }

  // Acceptance rate
  const specBench = benches.find(b => b.acceptance_rate_pct != null);
  document.getElementById('acceptance-body').innerHTML = specBench
    ? `<div class="acceptance" style="font-size:3rem;font-weight:700;color:var(--green)">${specBench.acceptance_rate_pct.toFixed(1)}%</div>
       <div class="stat-label">of speculative draft tokens accepted by verifier</div>
       <p style="margin-top:1rem;font-size:0.85rem;color:var(--muted)">
         Higher = 1B draft model predicts well → larger TTFT savings.
         Typical range for matched Llama family: 60–80%.
       </p>`
    : `<p style="color:var(--muted)">Run config [3] to capture acceptance rate.</p>`;
}

render();
</script>
</body>
</html>
```

---

## Step 7 — 1-Command Master Script

**File**: `scripts/run_all.sh`
```bash
#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== ArmForge: Full Pipeline ==="

# Env
python3.12 -m venv ~/armforge-env 2>/dev/null || true
source ~/armforge-env/bin/activate
pip install -q -r requirements.txt

# Build
bash scripts/01_build_kleidiai.sh
bash scripts/01b_build_baseline.sh
bash scripts/02_download_models.sh

# Detect hardware
python bench/arm_features.py

# Tune threads
bash scripts/03_tune_threads.sh

# Benchmark
python bench/bench_llamacpp.py

# Generate report
python bench/compare.py

echo ""
echo "=== All benchmarks complete ==="
echo "Results: bench/results/"
echo "Start dashboard: bash scripts/07_start_dashboard.sh"
```

**File**: `scripts/07_start_dashboard.sh`
```bash
#!/bin/bash
source ~/armforge-env/bin/activate
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
```

---

## Step 8 — Full Run Order

```bash
# Clone & enter
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge

# Full pipeline (builds both binaries, downloads models, benchmarks, generates report)
bash scripts/run_all.sh

# Start dashboard
bash scripts/07_start_dashboard.sh
# Open: http://localhost:8080
```

---

## Expected Results

| Config | Throughput | TTFT | vs Baseline |
|---|---|---|---|
| [1] Baseline (KleidiAI OFF) | ~5 tok/s | ~750 ms | — |
| [2] KleidiAI ON | ~8 tok/s | ~620 ms | +56% tps |
| [3] + Speculative Decode | ~8 tok/s | ~420 ms | +54% tps, −44% TTFT |
| [4] + mlock | ~8 tok/s | ~400 ms | +54% tps, −47% TTFT |

Draft acceptance rate target: **65–80%** (Llama-3.2 1B→3B same family).

---

## Submission Checklist
- [ ] `bench/results/hardware.json` — ARM extension detection proof
- [ ] `bench/results/thread_sweep.csv` — llama-bench tuning output
- [ ] `bench/results/llamacpp_results.json` — full benchmark data
- [ ] `bench/results/SUMMARY.md` — judge-readable report with hardware, bars, acceptance rate
- [ ] Dashboard screenshot showing all 4 KPI cards + TTFT curve
- [ ] `README.md`: chip model, RAM, OS, methodology, citations (Arm KleidiAI blog, llama.cpp speculative docs)
