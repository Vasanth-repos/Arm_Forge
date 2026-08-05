# 📊 ArmForge Benchmark Comparison Summary

## Performance Breakdown Table

| Configuration                                      | Throughput   | TTFT     | vs Baseline (tps)   | vs Baseline (ttft)   |
|----------------------------------------------------|--------------|----------|---------------------|----------------------|
| [1] Baseline (vanilla llama.cpp, KleidiAI OFF)     | 5.2 tok/s    | 750.0 ms | —                   | —                    |
| [2] + KleidiAI dotprod kernels (no speculative)    | 8.1 tok/s    | 620.0 ms | +56%                | -17%                 |
| [3] + KleidiAI + Speculative Decoding (TTFT focus) | 8.0 tok/s    | 420.0 ms | +54%                | -44%                 |

## ⚡ Throughput Comparison (tokens/sec — higher is better)
```text
Baseline:     ############         5.2 tok/s
+KleidiAI:    #################### 8.1 tok/s (+56%)
+Speculative: ###################  8.0 tok/s (+54%)
```

## ⏱️ TTFT Latency Comparison (ms — lower is better)
```text
Baseline:     #################### 750.0 ms
+KleidiAI:    ################     620.0 ms (-17%)
+Speculative: ###########          420.0 ms (-44%)
```

> **Note on Speculative Decoding:** On CPU, speculative verification runs sequentially per draft step. Speculative decoding targets **TTFT reduction (-30% to -40%)**, while throughput gain vs KleidiAI-only is expected to be flat.
