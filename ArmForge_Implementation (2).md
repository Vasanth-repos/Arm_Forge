# ArmForge — Arm AI Optimization Challenge 2026 (Track 3: Mobile AI)
> **Agent instruction**: Build this project end-to-end. Every file, command, and config is fully specified. Run steps in order. Do not skip any step — benchmarks, hardware detection, and the live playground are all submission deliverables.

---

## Project Summary

**ArmForge** stacks four hardware-level and architectural optimizations for 100% offline LLM inference on ARM64 client hardware, measured against a clean baseline on identical hardware:

| Layer | Optimization | Primary Win |
|---|---|---|
| 1 | Arm KleidiAI `dotprod` + `i8mm` vector kernels | +50–60% throughput |
| 2 | Speculative decoding — draft-simple (3B+1B) vs n-gram (zero overhead) | −40–50% TTFT |
| 3 | Dynamic thread tuning via `llama-bench` pp+tg sweep | +10–15% throughput |
| 4 | `numactl` NUMA binding + `--mlock` memory locking | −5–15% latency spikes |

**Platform**: ARM64 laptop / mobile workstation (Snapdragon X Elite, Apple M-series, or Oracle A1 free tier) — fully offline, zero cloud dependency.

---

## ⚠️ Critical Notes (Read Before Building)

- **KleidiAI CMake flag**: Use `-DGGML_KLEIDIAI=ON` (NOT `-DGGML_CPU_KLEIDIAI=ON` — that flag does not exist).
- **i8mm activation**: Must pass `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm` explicitly; auto-detection is unreliable.
- **`-b 512`**: Required to route quantized matrix ops through KleidiAI kernel paths; default batch size bypasses them.
- **Speculative decode flags (mid-2026)**: `--draft` is removed. Use `--spec-draft-model`, `--spec-type draft-simple`, `--spec-draft-n-max`.
- **n-gram speculative decode**: Use `--spec-type ngram-simple` — needs no draft model, zero extra RAM.
- **Acceptance rate log line**: Parse `draft_accept_rate = X.XX` from llama.cpp stderr (not "accepted draft").
- **`--mlock`**: Still valid. `--load-mode mlock` does NOT exist.
- **`-ngl 0`**: Always set explicitly on all calls — guarantees CPU-only, no GPU ambiguity.
- **Warmup run**: Always do 1 silent warmup call before recording — first run includes model load time.
- **`llama-bench` output**: Use `--output csv` and measure both `pp` (prompt processing) and `tg` (token generation) separately.
- **Quantization variable**: Benchmark Q8_0 baseline vs Q8_0 KleidiAI vs Q4_K_M KleidiAI — don't mix quant format and kernel optimization.
- **numactl**: On multi-cluster ARM chips (Snapdragon X Elite), bind with `numactl --cpunodebind=0 --membind=0`.
- **Windows ARM (WSL2)**: Add `-DLLAMA_BUILD_SERVER_WEBUI=OFF` to CMake to avoid npm UNC path errors.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference engine | llama.cpp (`-DGGML_KLEIDIAI=ON`, `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm`) |
| Quantization | GGUF Q8_0 (baseline) + Q4_K_M (KleidiAI throughput) |
| Speculative decode A | `--spec-type draft-simple` (3B verifier + 1B draft, same tokenizer) |
| Speculative decode B | `--spec-type ngram-simple` (zero model overhead, context-based) |
| Thread tuning | `llama-bench` CSV sweep — pp512 + tg128, T ∈ {1,2,3,4} |
| Memory | `--mlock` + `numactl --cpunodebind=0 --membind=0` |
| Benchmark harness | Python 3.12, `subprocess`, `statistics`, `psutil` |
| Serving | `llama-server` OpenAI-compatible endpoint (`:8000`) |
| Dashboard | FastAPI + Jinja2 + Chart.js + SSE live playground |
| Hardware detection | `bench/arm_features.py` (dotprod, i8mm, sve2, perf-per-watt) |
| Container | Docker (`--platform linux/arm64`) |
| Platform | ARM64 laptop / Oracle Cloud A1 free tier |

---

## Directory Structure

```
armforge/
├── scripts/
│   ├── run_all.sh
│   ├── 01_build_kleidiai.sh
│   ├── 01b_build_baseline.sh
│   ├── 02_download_models.sh
│   ├── 03_tune_threads.sh
│   ├── 04_benchmark_all.sh
│   └── 07_start_dashboard.sh
├── bench/
│   ├── arm_features.py          ← ARM extension + perf-per-watt detector
│   ├── bench_llamacpp.py        ← Full harness (6 configs, warmup, pp+tg split)
│   ├── bench_llama_bench.py     ← llama-bench structured CSV runner
│   ├── compare.py               ← Generates SUMMARY.md with ASCII bars
│   └── results/                 ← JSON + CSV outputs (auto-created)
├── dashboard/
│   ├── app.py                   ← FastAPI + SSE streaming playground
│   └── templates/index.html     ← Full UI with live prompt tester
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Step 0 — Detect ARM Hardware Features

**File**: `bench/arm_features.py`
```python
"""
Detects active ARM vector extensions and estimates energy efficiency.
Output: bench/results/hardware.json — included in SUMMARY.md as judge proof.
"""
import subprocess, platform, json, time, pathlib

def detect():
    info = {
        "arch": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "os": platform.platform(),
        "extensions": {},
        "numa_nodes": None,
        "perf_per_watt": None,
    }

    # Parse /proc/cpuinfo for ARM feature flags
    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
        feat_line = next((l for l in cpuinfo.splitlines() if l.startswith("Features")), "")
        flags = feat_line.split(":")[1].split() if ":" in feat_line else []
        info["extensions"] = {
            "dotprod": "asimddp" in flags,
            "i8mm":    "i8mm"    in flags,
            "sve":     "sve"     in flags,
            "sve2":    "sve2"    in flags,
            "bf16":    "bf16"    in flags,
            "all_flags": flags,
        }
    except Exception as e:
        info["extensions"]["error"] = str(e)

    # NUMA topology
    try:
        r = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True)
        info["numa_nodes"] = r.stdout.strip()[:300]
    except FileNotFoundError:
        info["numa_nodes"] = "numactl not installed"

    # Verify KleidiAI active in llama.cpp build
    llamacpp = pathlib.Path.home() / "llama.cpp/build_kleidiai/bin/llama-cli"
    if llamacpp.exists():
        r = subprocess.run([str(llamacpp), "--version"], capture_output=True, text=True)
        out = r.stdout + r.stderr
        info["llamacpp_kleidiai_active"] = "KLEIDIAI" in out.upper()
        info["llamacpp_neon"] = "NEON = 1" in out
        info["llamacpp_version_output"] = out[:500]

    # Rough perf-per-watt estimate using powertop (if available)
    # Records idle power draw; actual inference power measured separately in bench
    try:
        r = subprocess.run(
            ["sudo", "powertop", "--time=3", "--csv=/tmp/pt.csv"],
            capture_output=True, text=True, timeout=10
        )
        info["powertop_available"] = (r.returncode == 0)
    except Exception:
        info["powertop_available"] = False

    out_path = pathlib.Path("bench/results/hardware.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2))
    print("=== ARM Hardware Detection ===")
    print(json.dumps(info, indent=2))
    return info

if __name__ == "__main__":
    detect()
```

---

## Step 1 — Build llama.cpp

### 1a — KleidiAI Build

**File**: `scripts/01_build_kleidiai.sh`
```bash
#!/bin/bash
set -e
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
./build_kleidiai/bin/llama-cli --version 2>&1 | grep -iE "NEON|KLEIDIAI|dotprod|i8mm" || true
```

### 1b — Baseline Build (KleidiAI OFF)

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

# Q8_0 — for fair baseline vs KleidiAI comparison (same quant, different kernels)
hf_hub_download(
    repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF',
    filename='Llama-3.2-3B-Instruct-Q8_0.gguf',
    local_dir='models/'
)
# Q4_K_M — KleidiAI throughput showcase
hf_hub_download(
    repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF',
    filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    local_dir='models/'
)
# Draft model — MUST be same tokenizer family (Llama-3.2, identical vocab)
hf_hub_download(
    repo_id='bartowski/Llama-3.2-1B-Instruct-GGUF',
    filename='Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    local_dir='models/'
)
print("All models downloaded.")
EOF
```

---

## Step 3 — Thread Tuning (llama-bench Structured CSV)

**File**: `scripts/03_tune_threads.sh`
```bash
#!/bin/bash
set -e
LLAMA_BENCH=~/llama.cpp/build_kleidiai/bin/llama-bench
MODEL=~/armforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
RESULTS=bench/results
mkdir -p $RESULTS

echo "=== Thread Sweep: pp512 + tg128 ==="
# --output csv gives structured, parseable output with pp and tg split
# --simple-io ensures non-interactive mode (no stdin required)
$LLAMA_BENCH \
    -m "$MODEL" \
    -p 512 -n 128 \
    -t 1,2,3,4 \
    -b 512 \
    --output csv \
    --simple-io \
    2>&1 | tee $RESULTS/thread_sweep.csv

echo "Thread sweep saved: $RESULTS/thread_sweep.csv"

# Auto-select best thread count for tg (token generation)
BEST_T=$(python3 - << 'PYEOF'
import csv, sys

rows = []
with open("bench/results/thread_sweep.csv") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)

# llama-bench CSV: model,size,params,backend,ngl,n_batch,n_ubatch,type_k,type_v,n_threads,n_gpu_layers,test,t/s
if len(rows) < 2:
    print("4")
    sys.exit()

header = rows[0].split(",")
data   = [dict(zip(header, r.split(","))) for r in rows[1:]]

# Focus on tg (token generation) rows
tg_rows = [r for r in data if "tg" in r.get("test","")]
if not tg_rows:
    tg_rows = data  # fallback: use all rows

best = max(tg_rows, key=lambda r: float(r.get("t/s", 0) or 0))
print(best.get("n_threads", "4"))
PYEOF
)

echo "Best thread count (tg): $BEST_T"
echo "$BEST_T" > $RESULTS/best_threads.txt
```

---

## Step 4 — llama-bench Structured Benchmark

**File**: `bench/bench_llama_bench.py`
```python
"""
Runs llama-bench for all 3 quant configs, captures pp + tg split.
This is the structured benchmark judges can verify independently.
Output: bench/results/llama_bench_results.json
"""
import subprocess, csv, json, io, time
from pathlib import Path

HOME        = Path.home()
BENCH_BIN   = HOME / "llama.cpp/build_kleidiai/bin/llama-bench"
BASE_BIN    = HOME / "llama.cpp/build_baseline/bin/llama-bench"
MODELS_DIR  = HOME / "armforge/models"
RESULTS_DIR = Path("bench/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    N_THREADS = int((RESULTS_DIR / "best_threads.txt").read_text().strip())
except Exception:
    N_THREADS = 4

CONFIGS = [
    # (label, binary, model, extra_args)
    ("baseline_Q8_0",
     str(BASE_BIN),
     str(MODELS_DIR / "Llama-3.2-3B-Instruct-Q8_0.gguf"),
     []),
    ("kleidiai_Q8_0",
     str(BENCH_BIN),
     str(MODELS_DIR / "Llama-3.2-3B-Instruct-Q8_0.gguf"),
     ["-b", "512"]),
    ("kleidiai_Q4_K_M",
     str(BENCH_BIN),
     str(MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
     ["-b", "512"]),
]


def run_llama_bench(label: str, binary: str, model: str, extra: list) -> dict:
    cmd = [
        binary,
        "-m", model,
        "-p", "512",   # pp: prompt processing tokens
        "-n", "128",   # tg: token generation tokens
        "-t", str(N_THREADS),
        "-ngl", "0",
        "--output", "csv",
        "--simple-io",
    ] + extra

    print(f"  Running llama-bench: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Parse CSV output
    pp_tps = tg_tps = None
    try:
        lines = [l for l in result.stdout.splitlines() if l.strip() and not l.startswith("#")]
        if len(lines) >= 2:
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            for row in reader:
                test = row.get("test", "")
                tps  = float(row.get("t/s", 0) or 0)
                if "pp" in test:
                    pp_tps = tps
                elif "tg" in test:
                    tg_tps = tps
    except Exception as e:
        print(f"    Parse error: {e}")
        print(f"    stdout: {result.stdout[:300]}")

    print(f"    pp={pp_tps:.1f} tok/s  tg={tg_tps:.1f} tok/s" if pp_tps else "    parse failed")
    return {"label": label, "pp_tok_s": pp_tps, "tg_tok_s": tg_tps}


if __name__ == "__main__":
    results = []
    for label, binary, model, extra in CONFIGS:
        r = run_llama_bench(label, binary, model, extra)
        results.append(r)

    # Speedup summary
    baseline = next((r for r in results if r["label"] == "baseline_Q8_0"), None)
    for r in results:
        if baseline and baseline["tg_tok_s"] and r["tg_tok_s"]:
            r["tg_speedup_vs_baseline"] = round(r["tg_tok_s"] / baseline["tg_tok_s"], 2)
        if baseline and baseline["pp_tok_s"] and r["pp_tok_s"]:
            r["pp_speedup_vs_baseline"] = round(r["pp_tok_s"] / baseline["pp_tok_s"], 2)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_threads": N_THREADS,
        "results": results,
    }
    Path("bench/results/llama_bench_results.json").write_text(json.dumps(out, indent=2))
    print("\nllama-bench results saved.")
    for r in results:
        print(f"  {r['label']}: pp={r['pp_tok_s']} tg={r['tg_tok_s']} | "
              f"tg speedup={r.get('tg_speedup_vs_baseline','—')}x")
```

---

## Step 5 — Full Benchmark Harness (6 Configs + Warmup + Acceptance Rate)

**File**: `bench/bench_llamacpp.py`
```python
"""
6 benchmark configurations with warmup discard, correct acceptance rate parsing,
TTFT curve at 3 prompt lengths, and numactl binding.

Configs:
  [1] Baseline Q8_0 (KleidiAI OFF)
  [2] KleidiAI Q8_0 (same quant, different kernels — clean comparison)
  [3] KleidiAI Q4_K_M + -b 512
  [4] KleidiAI Q4_K_M + speculative draft-simple (3B+1B)
  [5] KleidiAI Q4_K_M + speculative ngram-simple (zero overhead)
  [6] KleidiAI Q4_K_M + speculative draft-simple + mlock + numactl

Output: bench/results/llamacpp_results.json
"""
import subprocess, time, json, statistics, re, shutil
from pathlib import Path

HOME         = Path.home()
BASELINE_BIN = HOME / "llama.cpp/build_baseline/bin/llama-cli"
KLEIDIAI_BIN = HOME / "llama.cpp/build_kleidiai/bin/llama-cli"
MODELS_DIR   = HOME / "armforge/models"
MAIN_Q8      = MODELS_DIR / "Llama-3.2-3B-Instruct-Q8_0.gguf"
MAIN_Q4      = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DRAFT_Q4     = MODELS_DIR / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
RESULTS_DIR  = Path("bench/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    N_THREADS = int((RESULTS_DIR / "best_threads.txt").read_text().strip())
except Exception:
    N_THREADS = 4

HAS_NUMACTL = shutil.which("numactl") is not None

PROMPT       = "Explain the Arm Neoverse architecture and its advantages for AI workloads in detail."
N_PREDICT    = 200
RUNS         = 6   # run 6, discard first (warmup), record 5
WARMUP_RUNS  = 1

TTFT_PROMPTS = {
    "short":  "Hello",
    "medium": "Explain transformer attention mechanisms",
    "long":   ("Explain transformer attention in detail, covering self-attention, "
                "multi-head attention, positional encoding, and how these components "
                "interact during inference on CPU hardware without GPU acceleration"),
}


def make_cmd(binary: str, model: str, prompt: str, n: int,
             extra: list, use_numactl: bool = False) -> list:
    base = [
        str(binary), "-m", str(model),
        "-p", prompt, "-n", str(n),
        "-t", str(N_THREADS),
        "-ngl", "0",
        "-b", "512",
        "--no-display-prompt",
        "--log-disable",
    ] + extra
    if use_numactl and HAS_NUMACTL:
        return ["numactl", "--cpunodebind=0", "--membind=0"] + base
    return base


def parse_tps(stderr: str) -> float | None:
    for line in stderr.splitlines():
        if "tokens per second" in line:
            try:
                return float(line.split("tokens per second")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass
    return None


def parse_ttft(stderr: str) -> float | None:
    """Parses prompt eval time in ms from llama.cpp stderr."""
    for line in stderr.splitlines():
        if "prompt eval time" in line:
            m = re.search(r"([\d.]+)\s*ms\s*/\s*\d+\s*tokens", line)
            if m:
                return float(m.group(1))
    return None


def parse_acceptance_rate(stderr: str) -> float | None:
    """
    Correct log line format in current llama.cpp:
      draft_accept_rate = 0.72  (or as percentage in some builds: 72.00%)
    """
    for line in stderr.splitlines():
        # Format A: "draft_accept_rate = 0.72"
        m = re.search(r"draft_accept_rate\s*=\s*([\d.]+)", line)
        if m:
            val = float(m.group(1))
            return val * 100 if val <= 1.0 else val  # normalise to %
        # Format B: "accepted X / Y" — fallback
        m2 = re.search(r"accepted\s+(\d+)\s*/\s*(\d+)", line)
        if m2:
            a, b = int(m2.group(1)), int(m2.group(2))
            return (a / b * 100) if b > 0 else None
    return None


def run_config(binary: str, model: str, label: str,
               extra: list, use_numactl: bool = False) -> dict:
    tps_list, ttft_list, accept_list = [], [], []
    total_runs = RUNS + WARMUP_RUNS

    for i in range(total_runs):
        cmd = make_cmd(binary, model, PROMPT, N_PREDICT, extra, use_numactl)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if i < WARMUP_RUNS:
            continue  # discard warmup
        tps  = parse_tps(result.stderr)
        ttft = parse_ttft(result.stderr)
        acc  = parse_acceptance_rate(result.stderr)
        if tps:  tps_list.append(tps)
        if ttft: ttft_list.append(ttft)
        if acc is not None: accept_list.append(acc)
        run_num = i - WARMUP_RUNS + 1
        print(f"  [{label}] Run {run_num}/{RUNS}: tps={tps:.1f if tps else 'N/A'} "
              f"ttft={ttft:.0f if ttft else 'N/A'}ms "
              f"acc={acc:.1f if acc else 'N/A'}%")

    return {
        "label":                 label,
        "tokens_per_sec_mean":   statistics.mean(tps_list)   if tps_list   else None,
        "tokens_per_sec_stdev":  statistics.stdev(tps_list)  if len(tps_list) > 1 else 0,
        "ttft_ms_mean":          statistics.mean(ttft_list)  if ttft_list  else None,
        "acceptance_rate_pct":   statistics.mean(accept_list) if accept_list else None,
        "runs_tps":  tps_list,
        "n_threads": N_THREADS,
        "numactl":   use_numactl and HAS_NUMACTL,
    }


def run_ttft_curve(binary: str, model: str, label: str, extra: list) -> dict:
    out = {"label": f"{label}_ttft_curve"}
    for plen, prompt in TTFT_PROMPTS.items():
        # Warmup once, then measure
        cmd = make_cmd(binary, model, prompt, 1, extra)
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # warmup
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out[plen] = parse_ttft(r.stderr)
        print(f"  [{label}] TTFT {plen}: {out[plen]:.1f} ms" if out[plen] else f"  [{label}] TTFT {plen}: N/A")
    return out


SPEC_DRAFT_ARGS = [
    "--spec-draft-model", str(DRAFT_Q4),
    "--spec-type", "draft-simple",
    "--spec-draft-n-max", "4",
]
SPEC_NGRAM_ARGS = [
    "--spec-type", "ngram-simple",
    "--spec-draft-n-max", "4",
]

if __name__ == "__main__":
    all_benches = []

    print("\n=== [1] Baseline Q8_0 (KleidiAI OFF) ===")
    b1 = run_config(str(BASELINE_BIN), str(MAIN_Q8), "baseline_Q8_vanilla", [])
    all_benches.append(b1)

    print("\n=== [2] KleidiAI Q8_0 (same quant, kernel upgrade) ===")
    b2 = run_config(str(KLEIDIAI_BIN), str(MAIN_Q8), "kleidiai_Q8_0", [])
    all_benches.append(b2)

    print("\n=== [3] KleidiAI Q4_K_M + -b 512 ===")
    b3 = run_config(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_Q4_K_M", [])
    all_benches.append(b3)

    print("\n=== [4] KleidiAI + Speculative draft-simple (3B+1B) ===")
    b4 = run_config(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_spec_draft", SPEC_DRAFT_ARGS)
    all_benches.append(b4)

    print("\n=== [5] KleidiAI + Speculative ngram-simple (zero overhead) ===")
    b5 = run_config(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_spec_ngram", SPEC_NGRAM_ARGS)
    all_benches.append(b5)

    print("\n=== [6] KleidiAI + draft-simple + mlock + numactl ===")
    b6 = run_config(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai_full_stack",
                    SPEC_DRAFT_ARGS + ["--mlock"], use_numactl=True)
    all_benches.append(b6)

    print("\n=== TTFT Curves (3 prompt lengths) ===")
    ttft_curves = [
        run_ttft_curve(str(BASELINE_BIN), str(MAIN_Q8), "baseline",  []),
        run_ttft_curve(str(KLEIDIAI_BIN), str(MAIN_Q4), "kleidiai",  []),
        run_ttft_curve(str(KLEIDIAI_BIN), str(MAIN_Q4), "spec_draft", SPEC_DRAFT_ARGS),
        run_ttft_curve(str(KLEIDIAI_BIN), str(MAIN_Q4), "spec_ngram", SPEC_NGRAM_ARGS),
    ]

    hw = json.loads((RESULTS_DIR / "hardware.json").read_text()) if (RESULTS_DIR / "hardware.json").exists() else {}

    def safe_speedup(a, b):
        if a and b and b > 0: return round(a / b, 2)
        return None

    summary = {
        "kleidiai_kernel_speedup_x":    safe_speedup(b2["tokens_per_sec_mean"], b1["tokens_per_sec_mean"]),
        "kleidiai_q4_speedup_x":        safe_speedup(b3["tokens_per_sec_mean"], b1["tokens_per_sec_mean"]),
        "ttft_reduction_spec_draft_pct": round((1 - b4["ttft_ms_mean"] / b1["ttft_ms_mean"]) * 100, 1)
                                          if b1.get("ttft_ms_mean") and b4.get("ttft_ms_mean") else None,
        "ttft_reduction_spec_ngram_pct": round((1 - b5["ttft_ms_mean"] / b1["ttft_ms_mean"]) * 100, 1)
                                          if b1.get("ttft_ms_mean") and b5.get("ttft_ms_mean") else None,
        "draft_acceptance_rate_pct":    b4.get("acceptance_rate_pct"),
        "ngram_acceptance_rate_pct":    b5.get("acceptance_rate_pct"),
        "full_stack_tps":               b6.get("tokens_per_sec_mean"),
    }

    output = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware":    hw,
        "n_threads":   N_THREADS,
        "has_numactl": HAS_NUMACTL,
        "benchmarks":  all_benches,
        "ttft_curves": ttft_curves,
        "summary":     summary,
    }

    out_path = RESULTS_DIR / "llamacpp_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Results: {out_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
```

---

## Step 6 — Generate SUMMARY.md

**File**: `bench/compare.py`
```python
"""
Reads all bench/results/*.json and writes bench/results/SUMMARY.md.
This is the judge-facing report — self-contained, downloadable.
"""
import json
from pathlib import Path

RESULTS = Path("bench/results")

def bar(value: float, max_val: float, width: int = 28) -> str:
    if not value or not max_val: return "░" * width
    filled = min(int((value / max_val) * width), width)
    return "█" * filled + "░" * (width - filled)

def main():
    lr_path = RESULTS / "llamacpp_results.json"
    lb_path = RESULTS / "llama_bench_results.json"
    hw_path = RESULTS / "hardware.json"

    lr = json.loads(lr_path.read_text()) if lr_path.exists() else {}
    lb = json.loads(lb_path.read_text()) if lb_path.exists() else {}
    hw = json.loads(hw_path.read_text()) if hw_path.exists() else lr.get("hardware", {})

    exts = hw.get("extensions", {})
    s    = lr.get("summary", {})
    b    = lr.get("benchmarks", [])

    tps_map  = {r["label"]: r["tokens_per_sec_mean"] for r in b if r.get("tokens_per_sec_mean")}
    ttft_map = {r["label"]: r["ttft_ms_mean"]         for r in b if r.get("ttft_ms_mean")}
    max_tps  = max(tps_map.values(),  default=1)
    min_ttft = min((v for v in ttft_map.values() if v), default=1)
    max_ttft = max((v for v in ttft_map.values() if v), default=1)

    lines = [
        "# ArmForge — Benchmark Summary",
        "",
        "## Hardware",
        f"- **Arch**: {hw.get('arch','?')}",
        f"- **CPU**: {hw.get('cpu','?')}",
        f"- **OS**: {hw.get('os','?')}",
        f"- **dotprod (i8 dot product)**: {'✅' if exts.get('dotprod') else '❌'}",
        f"- **i8mm (int8 matrix multiply)**: {'✅' if exts.get('i8mm') else '❌'}",
        f"- **SVE**: {'✅' if exts.get('sve') else '❌'}",
        f"- **SVE2**: {'✅' if exts.get('sve2') else '❌'}",
        f"- **BF16**: {'✅' if exts.get('bf16') else '❌'}",
        f"- **KleidiAI active in llama.cpp**: {'✅' if hw.get('llamacpp_kleidiai_active') else '❌'}",
        f"- **NUMA topology**: {hw.get('numa_nodes','not detected')}",
        f"- **Threads used**: {lr.get('n_threads','?')} (auto-tuned via llama-bench)",
        f"- **numactl binding active**: {'✅' if lr.get('has_numactl') else '❌'}",
        "",
        "---",
        "",
        "## llama-bench Results (Structured — pp + tg split)",
        "",
        "| Config | pp (prompt tok/s) | tg (gen tok/s) | tg speedup |",
        "|---|---|---|---|",
    ]
    for r in lb.get("results", []):
        lines.append(
            f"| {r['label']} | {r['pp_tok_s']:.1f if r['pp_tok_s'] else 'N/A'} | "
            f"{r['tg_tok_s']:.1f if r['tg_tok_s'] else 'N/A'} | "
            f"{r.get('tg_speedup_vs_baseline','—')}x |"
        )

    lines += [
        "",
        "---",
        "",
        "## Throughput: All Configs (tokens/sec — higher is better)",
        "",
    ]
    for label, val in tps_map.items():
        lines.append(f"- `{label}`: **{val:.2f} tok/s**  {bar(val, max_tps)}")

    lines += ["", "## TTFT Latency (ms — lower is better)", ""]
    for label, val in ttft_map.items():
        if val:
            # Invert bar so shorter bar = faster (better)
            lines.append(f"- `{label}`: **{val:.1f} ms**  {bar(max_ttft - val, max_ttft)}")

    lines += ["", "## TTFT by Prompt Length", ""]
    for curve in lr.get("ttft_curves", []):
        lbl = curve["label"].replace("_ttft_curve", "")
        lines.append(f"**{lbl}**: "
                     f"short={curve.get('short','N/A')} ms  "
                     f"medium={curve.get('medium','N/A')} ms  "
                     f"long={curve.get('long','N/A')} ms")

    acc_d = s.get("draft_acceptance_rate_pct")
    acc_n = s.get("ngram_acceptance_rate_pct")
    lines += [
        "",
        "## Speculative Decode — Acceptance Rates",
        f"- **draft-simple** (3B+1B): {acc_d:.1f}% accepted" if acc_d else "- draft-simple: N/A",
        f"- **ngram-simple** (zero model overhead): {acc_n:.1f}% accepted" if acc_n else "- ngram-simple: N/A",
        "> Higher acceptance = draft model predicts well → larger TTFT savings. Target: 65–80% for matched Llama family.",
        "",
        "## Key Results Summary",
        f"- **KleidiAI kernel speedup (Q8_0 vs Q8_0)**: {s.get('kleidiai_kernel_speedup_x','?')}x",
        f"- **KleidiAI + Q4_K_M speedup**: {s.get('kleidiai_q4_speedup_x','?')}x vs baseline",
        f"- **TTFT reduction — draft-simple**: {s.get('ttft_reduction_spec_draft_pct','?')}%",
        f"- **TTFT reduction — ngram-simple**: {s.get('ttft_reduction_spec_ngram_pct','?')}%",
        f"- **Full stack (KleidiAI+spec+mlock+numactl)**: {s.get('full_stack_tps','?')} tok/s",
        "",
        "---",
        "_Generated by ArmForge bench/compare.py_",
    ]

    out = RESULTS / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"✅ SUMMARY.md written: {out}")

main()
```

---

## Step 7 — Dashboard with Live SSE Playground

**File**: `dashboard/app.py`
```python
"""
FastAPI dashboard with:
  - /           → full results UI
  - /api/results → JSON all bench results
  - /api/summary → SUMMARY.md text
  - /api/stream  → SSE: streams llama-server output token by token
  - /api/download/summary → download SUMMARY.md
  - /api/download/results → download llamacpp_results.json
"""
import json, asyncio, subprocess, shutil
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="ArmForge")
templates = Jinja2Templates(directory="dashboard/templates")
RESULTS_DIR = Path("bench/results")
HOME        = Path.home()
LLAMA_CLI   = HOME / "llama.cpp/build_kleidiai/bin/llama-cli"
MAIN_Q4     = HOME / "armforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DRAFT_Q4    = HOME / "armforge/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

try:
    N_THREADS = int((RESULTS_DIR / "best_threads.txt").read_text().strip())
except Exception:
    N_THREADS = 4

HAS_NUMACTL = shutil.which("numactl") is not None


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
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/results")
async def api_results():
    return load_results()


@app.get("/api/summary")
async def api_summary():
    p = RESULTS_DIR / "SUMMARY.md"
    return {"markdown": p.read_text() if p.exists() else "Run bench/compare.py first."}


@app.get("/api/download/summary")
async def download_summary():
    p = RESULTS_DIR / "SUMMARY.md"
    return FileResponse(p, media_type="text/markdown", filename="ArmForge_SUMMARY.md")


@app.get("/api/download/results")
async def download_results():
    p = RESULTS_DIR / "llamacpp_results.json"
    return FileResponse(p, media_type="application/json", filename="armforge_results.json")


@app.get("/api/stream")
async def stream_inference(prompt: str = "Tell me about Arm KleidiAI"):
    """
    SSE endpoint: streams llama-cli output token by token to the browser.
    Uses the full-stack config (KleidiAI + speculative + mlock).
    """
    async def generate():
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MAIN_Q4),
            "-p", prompt,
            "-n", "256",
            "-t", str(N_THREADS),
            "-ngl", "0",
            "-b", "512",
            "--mlock",
            "--spec-draft-model", str(DRAFT_Q4),
            "--spec-type", "draft-simple",
            "--spec-draft-n-max", "4",
            "--no-display-prompt",
            "--log-disable",
        ]
        if HAS_NUMACTL:
            cmd = ["numactl", "--cpunodebind=0", "--membind=0"] + cmd

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        yield "data: {\"type\":\"start\"}\n\n"
        async for line in proc.stdout:
            token = line.decode("utf-8", errors="replace")
            payload = json.dumps({"type": "token", "text": token})
            yield f"data: {payload}\n\n"
        await proc.wait()
        yield "data: {\"type\":\"end\"}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
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
    --bg:#0B0F14; --surface:#161b22; --surface2:#1c2128;
    --border:#30363d; --green:#3fb950; --blue:#58a6ff;
    --red:#f85149; --orange:#f0883e; --purple:#a371f7;
    --text:#e6edf3; --muted:#8b949e;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:1.5rem}
  h1{color:var(--blue);font-size:1.7rem;margin-bottom:.2rem}
  .subtitle{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}
  .section{margin-bottom:1.5rem}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem}
  .grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem}
  .card h3{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.8rem}
  canvas{max-height:220px}
  .stat{font-size:2.4rem;font-weight:700;color:var(--green);line-height:1}
  .stat-label{font-size:.75rem;color:var(--muted);margin-top:.4rem}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.7rem;font-weight:600;margin:.15rem}
  .badge.on{background:#1f3a1f;color:var(--green)}
  .badge.off{background:#3a1f1f;color:var(--red)}
  .badge.info{background:#1a2a3a;color:var(--blue)}
  textarea{width:100%;background:var(--surface2);color:var(--text);border:1px solid var(--border);
           border-radius:6px;padding:.75rem;font-size:.85rem;resize:vertical;min-height:72px}
  button{background:var(--blue);color:#000;border:none;border-radius:6px;padding:.5rem 1.2rem;
         font-weight:600;cursor:pointer;font-size:.85rem}
  button:hover{opacity:.85}
  button.dl{background:var(--surface2);color:var(--text);border:1px solid var(--border);margin-left:.5rem}
  #stream-out{background:#0d1117;border:1px solid var(--border);border-radius:6px;
              padding:.75rem;font-size:.8rem;min-height:100px;max-height:260px;
              overflow-y:auto;white-space:pre-wrap;color:var(--green);margin-top:.75rem}
  #stream-meta{font-size:.75rem;color:var(--muted);margin-top:.4rem}
  pre{background:#0d1117;border:1px solid var(--border);border-radius:6px;
      padding:1rem;font-size:.72rem;overflow-x:auto;white-space:pre-wrap;
      color:var(--text);max-height:360px;overflow-y:auto}
  @media(max-width:800px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr}}
</style>
</head>
<body>

<h1>🦾 ArmForge</h1>
<p class="subtitle">Arm AI Optimization Challenge 2026 · Track 3: Mobile AI · 100% Offline · Zero Cloud Dependency</p>

<!-- KPI row -->
<div class="grid-4 section" id="kpi"></div>

<!-- Hardware badges -->
<div class="card section" id="hw-card"></div>

<!-- Charts row 1 -->
<div class="grid-2 section">
  <div class="card"><h3>Throughput — All Configs (tok/s, higher is better)</h3><canvas id="tpsChart"></canvas></div>
  <div class="card"><h3>TTFT Latency — All Configs (ms, lower is better)</h3><canvas id="ttftChart"></canvas></div>
</div>

<!-- Charts row 2 -->
<div class="grid-2 section">
  <div class="card"><h3>TTFT by Prompt Length (prefill scaling)</h3><canvas id="ttftCurveChart"></canvas></div>
  <div class="card"><h3>llama-bench: pp vs tg Split (structured)</h3><canvas id="lbChart"></canvas></div>
</div>

<!-- Acceptance rates -->
<div class="grid-2 section">
  <div class="card" id="accept-card"><h3>Speculative Draft Acceptance Rate</h3></div>
  <div class="card" id="ngram-card"><h3>N-gram Speculative Acceptance Rate</h3></div>
</div>

<!-- Live playground -->
<div class="card section">
  <h3>Live Playground — On-Device Inference (SSE Streaming)</h3>
  <p style="font-size:.8rem;color:var(--muted);margin-bottom:.75rem">
    Runs KleidiAI + speculative decode + mlock + numactl on your ARM64 device in real time.
  </p>
  <textarea id="prompt-input" placeholder="Ask anything...">Tell me about Arm KleidiAI and i8mm matrix multiply.</textarea>
  <div style="margin-top:.6rem">
    <button onclick="runStream()">▶ Run Inference</button>
    <button class="dl" onclick="clearStream()">Clear</button>
  </div>
  <div id="stream-out">Output will appear here...</div>
  <div id="stream-meta"></div>
</div>

<!-- SUMMARY.md + Downloads -->
<div class="card section">
  <h3>SUMMARY.md — Judge Report
    <button class="dl" style="float:right" onclick="location.href='/api/download/summary'">⬇ Download .md</button>
    <button class="dl" style="float:right;margin-right:.5rem" onclick="location.href='/api/download/results'">⬇ Download JSON</button>
  </h3>
  <pre id="summary" style="margin-top:.75rem"></pre>
</div>

<script>
const COLORS=['#1f6feb','#3fb950','#f0883e','#a371f7','#58a6ff','#f85149'];

async function render(){
  const [data,sumRes]=await Promise.all([
    fetch('/api/results').then(r=>r.json()),
    fetch('/api/summary').then(r=>r.json())
  ]);

  document.getElementById('summary').textContent=sumRes.markdown||'';

  const lr=data.llamacpp_results||{};
  const lb=data.llama_bench_results||{};
  const hw=lr.hardware||{};
  const exts=hw.extensions||{};
  const s=lr.summary||{};
  const benches=lr.benchmarks||[];

  // KPI cards
  const kpis=[
    ['KleidiAI Kernel Speedup', s.kleidiai_kernel_speedup_x ? s.kleidiai_kernel_speedup_x.toFixed(2)+'x':'—','Q8_0 vs Q8_0 (pure kernel win)'],
    ['Q4_K_M Speedup',          s.kleidiai_q4_speedup_x    ? s.kleidiai_q4_speedup_x.toFixed(2)+'x'   :'—','KleidiAI Q4_K_M vs baseline'],
    ['TTFT Reduction',          s.ttft_reduction_spec_draft_pct ? s.ttft_reduction_spec_draft_pct+'%'  :'—','speculative draft-simple'],
    ['Threads',                 lr.n_threads||'?','auto-tuned via llama-bench'],
  ];
  document.getElementById('kpi').innerHTML=kpis.map(([h,v,l])=>
    `<div class="card"><h3>${h}</h3><div class="stat">${v}</div><div class="stat-label">${l}</div></div>`
  ).join('');

  // Hardware
  const extBadges=['dotprod','i8mm','sve','sve2','bf16'].map(k=>
    `<span class="badge ${exts[k]?'on':'off'}">${k}:${exts[k]?'✅':'❌'}</span>`
  ).join('');
  document.getElementById('hw-card').innerHTML=
    `<h3>Hardware</h3>
     <p style="color:var(--muted);font-size:.85rem;margin-bottom:.6rem">${hw.arch||'?'} · ${hw.cpu||'?'} · ${hw.os||'?'}</p>
     ${extBadges}
     <span class="badge ${hw.llamacpp_kleidiai_active?'on':'off'}">KleidiAI:${hw.llamacpp_kleidiai_active?'✅':'❌'}</span>
     <span class="badge info">threads:${lr.n_threads||'?'}</span>
     <span class="badge ${lr.has_numactl?'on':'off'}">numactl:${lr.has_numactl?'✅':'❌'}</span>`;

  // Throughput chart
  const tpsData=benches.filter(b=>b.tokens_per_sec_mean);
  new Chart(document.getElementById('tpsChart'),{
    type:'bar',
    data:{
      labels:tpsData.map(b=>b.label.replace(/_/g,' ')),
      datasets:[{label:'tok/s',data:tpsData.map(b=>b.tokens_per_sec_mean.toFixed(2)),
        backgroundColor:COLORS,borderRadius:4}]
    },
    options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}
  });

  // TTFT chart
  const ttftData=benches.filter(b=>b.ttft_ms_mean);
  if(ttftData.length){
    new Chart(document.getElementById('ttftChart'),{
      type:'bar',
      data:{
        labels:ttftData.map(b=>b.label.replace(/_/g,' ')),
        datasets:[{label:'ms',data:ttftData.map(b=>b.ttft_ms_mean.toFixed(1)),
          backgroundColor:COLORS.slice().reverse(),borderRadius:4}]
      },
      options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}
    });
  }

  // TTFT curve
  const curves=lr.ttft_curves||[];
  if(curves.length){
    new Chart(document.getElementById('ttftCurveChart'),{
      type:'line',
      data:{
        labels:['short','medium','long'],
        datasets:curves.map((c,i)=>({
          label:c.label.replace(/_ttft_curve/,'').replace(/_/g,' '),
          data:['short','medium','long'].map(k=>c[k]||null),
          borderColor:COLORS[i],backgroundColor:COLORS[i]+'33',
          tension:.3,pointRadius:5
        }))
      },
      options:{scales:{y:{beginAtZero:true,title:{display:true,text:'ms'}}}}
    });
  }

  // llama-bench pp vs tg
  const lbData=lb.results||[];
  if(lbData.length){
    new Chart(document.getElementById('lbChart'),{
      type:'bar',
      data:{
        labels:lbData.map(r=>r.label.replace(/_/g,' ')),
        datasets:[
          {label:'pp tok/s',data:lbData.map(r=>r.pp_tok_s||0),backgroundColor:'#1f6feb',borderRadius:4},
          {label:'tg tok/s',data:lbData.map(r=>r.tg_tok_s||0),backgroundColor:'#3fb950',borderRadius:4},
        ]
      },
      options:{plugins:{legend:{display:true}},scales:{y:{beginAtZero:true}}}
    });
  }

  // Acceptance rates
  const accD=s.draft_acceptance_rate_pct;
  const accN=s.ngram_acceptance_rate_pct;
  document.getElementById('accept-card').innerHTML+=
    accD!=null
    ? `<div class="stat">${accD.toFixed(1)}%</div>
       <div class="stat-label">draft tokens accepted by 3B verifier (target: 65–80%)</div>`
    : `<p style="color:var(--muted)">Run config [4] to capture.</p>`;
  document.getElementById('ngram-card').innerHTML+=
    accN!=null
    ? `<div class="stat">${accN.toFixed(1)}%</div>
       <div class="stat-label">n-gram predictions accepted — zero model overhead</div>`
    : `<p style="color:var(--muted)">Run config [5] to capture.</p>`;
}

// SSE streaming playground
let evtSource=null;
function runStream(){
  const prompt=document.getElementById('prompt-input').value.trim();
  if(!prompt) return;
  const out=document.getElementById('stream-out');
  const meta=document.getElementById('stream-meta');
  out.textContent='';
  meta.textContent='';
  if(evtSource){evtSource.close();}

  const t0=performance.now();
  let tokenCount=0;
  evtSource=new EventSource('/api/stream?prompt='+encodeURIComponent(prompt));

  evtSource.onmessage=e=>{
    const msg=JSON.parse(e.data);
    if(msg.type==='start'){
      out.textContent='';
    } else if(msg.type==='token'){
      out.textContent+=msg.text;
      out.scrollTop=out.scrollHeight;
      tokenCount++;
      const elapsed=(performance.now()-t0)/1000;
      meta.textContent=`${tokenCount} tokens · ${(tokenCount/elapsed).toFixed(1)} tok/s · ${elapsed.toFixed(1)}s`;
    } else if(msg.type==='end'){
      evtSource.close();
      meta.textContent+=' — done ✅';
    }
  };
  evtSource.onerror=()=>{evtSource.close();meta.textContent+=' [stream closed]';};
}
function clearStream(){
  if(evtSource) evtSource.close();
  document.getElementById('stream-out').textContent='Output will appear here...';
  document.getElementById('stream-meta').textContent='';
}

render();
</script>
</body>
</html>
```

---

## Step 8 — Dockerfile

**File**: `Dockerfile`
```dockerfile
# ARM64 explicit — prevents Docker from silently pulling x86 layers
FROM --platform=linux/arm64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3.12 python3.12-venv \
    python3-pip wget curl numactl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Python env
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

# Build llama.cpp KleidiAI inside container
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp && \
    cmake -B /opt/llama.cpp/build_kleidiai \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_KLEIDIAI=ON \
        -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm" \
        -DLLAMA_BUILD_SERVER_WEBUI=OFF \
        /opt/llama.cpp && \
    cmake --build /opt/llama.cpp/build_kleidiai --config Release -j$(nproc)

ENV LLAMA_CPP_HOME=/opt/llama.cpp
EXPOSE 8080

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**File**: `docker-compose.yml`
```yaml
version: "3.9"
services:
  armforge:
    build:
      context: .
      dockerfile: Dockerfile
    platform: linux/arm64      # REQUIRED — prevents x86 emulation
    ports:
      - "8080:8080"
    volumes:
      - ./bench/results:/app/bench/results
      - ${HOME}/armforge/models:/root/armforge/models:ro
    environment:
      - PYTHONUNBUFFERED=1
```

Build & run: `docker compose up --build`

---

## Step 9 — Master Run Script

**File**: `scripts/run_all.sh`
```bash
#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== ArmForge: Full Pipeline ==="

python3.12 -m venv ~/armforge-env 2>/dev/null || true
source ~/armforge-env/bin/activate
pip install -q -r requirements.txt

bash scripts/01_build_kleidiai.sh
bash scripts/01b_build_baseline.sh
bash scripts/02_download_models.sh

python bench/arm_features.py
bash scripts/03_tune_threads.sh

# llama-bench structured results
python bench/bench_llama_bench.py

# Full harness (6 configs + warmup + acceptance + TTFT curve)
python bench/bench_llamacpp.py

# Generate judge report
python bench/compare.py

echo ""
echo "=== Pipeline complete ==="
echo "Results: bench/results/"
echo "Dashboard: bash scripts/07_start_dashboard.sh → http://localhost:8080"
```

**File**: `scripts/07_start_dashboard.sh`
```bash
#!/bin/bash
source ~/armforge-env/bin/activate
uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
```

---

## Step 10 — One-Command Run

```bash
git clone https://github.com/Vasanth-repos/Armforge.git
cd Armforge
bash scripts/run_all.sh

# Then start dashboard
bash scripts/07_start_dashboard.sh
# Open: http://localhost:8080
```

**Or via Docker** (fully reproducible):
```bash
docker compose up --build
# Open: http://localhost:8080
```

---

## Expected Results

| Config | tg tok/s | TTFT | vs Baseline |
|---|---|---|---|
| [1] Baseline Q8_0 (KleidiAI OFF) | ~5.0 | ~750 ms | — |
| [2] KleidiAI Q8_0 (same quant) | ~7.5 | ~630 ms | **+50% tps** |
| [3] KleidiAI Q4_K_M + -b 512 | ~8.5 | ~610 ms | **+65% tps** |
| [4] + Speculative draft-simple | ~8.3 | ~420 ms | +60% tps, **−44% TTFT** |
| [5] + Speculative ngram-simple | ~8.0 | ~460 ms | +55% tps, **−38% TTFT** |
| [6] Full stack (+mlock+numactl) | ~8.5 | ~400 ms | +65% tps, **−47% TTFT** |

Draft acceptance rate target: **65–80%** (Llama-3.2 1B→3B matched family).
N-gram acceptance rate target: **40–60%** (context-dependent).

---

## Submission Checklist
- [ ] `bench/results/hardware.json` — ARM extension proof + numactl topology
- [ ] `bench/results/thread_sweep.csv` — llama-bench raw CSV
- [ ] `bench/results/llama_bench_results.json` — structured pp+tg split
- [ ] `bench/results/llamacpp_results.json` — 6-config full harness results
- [ ] `bench/results/SUMMARY.md` — downloadable judge report
- [ ] Dashboard screenshot: 4 KPI cards + TTFT curve + acceptance rates + live playground
- [ ] Docker `compose up --build` runs cleanly on ARM64
- [ ] `README.md`: exact chip model, RAM speed, OS version, methodology, citations
  - Arm KleidiAI blog: https://developer.arm.com/blogs/tag/kleidi
  - llama.cpp speculative docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
  - llama-bench docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/bench/README.md