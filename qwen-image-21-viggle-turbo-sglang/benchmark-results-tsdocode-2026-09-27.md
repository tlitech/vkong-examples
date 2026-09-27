# Qwen Image 2.1 concurrency benchmark — tsdocode

**Run date:** 2026-09-27  
**Service:** `qwen-image-21-viggle-turbo-sglang` via VKong public HTTPS route  
**GPU:** RTX 4090 24 GB (reported rate: $0.753699/hour)  
**Workload:** 100 total 512×512 image generations, six inference steps, guidance scale 1.0, fixed prompt and unique seeds. Concurrency levels ran as sequential batches; 13 requests at levels 1–4 and 12 at levels 5–8. Requests within each batch were concurrent.

| Concurrency | Samples | Success | Mean latency | Median | p95* | Mean SGLang inference | Mean combined overhead | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 13 | 2.510 s | 2.564 s | 3.916 s | 1.169 s | 1.341 s | 0.398 img/s |
| 2 | 13 | 13 | 2.602 s | 2.496 s | 4.003 s | 1.155 s | 1.447 s | 0.740 img/s |
| 3 | 13 | 13 | 3.609 s | 3.712 s | 4.452 s | 1.159 s | 2.450 s | 0.768 img/s |
| 4 | 13 | 13 | 4.706 s | 5.035 s | 5.839 s | 1.160 s | 3.546 s | 0.766 img/s |
| 5 | 12 | 12 | 5.489 s | 6.231 s | 6.843 s | 1.157 s | 4.331 s | 0.767 img/s |
| 6 | 12 | 12 | 6.289 s | 7.432 s | 8.373 s | 1.163 s | 5.125 s | 0.758 img/s |
| 7 | 12 | 12 | 6.956 s | 8.443 s | 9.385 s | 1.159 s | 5.797 s | 0.761 img/s |
| 8 | 12 | 12 | 7.527 s | 8.744 s | 10.875 s | 1.159 s | 6.368 s | 0.766 img/s |

**Outcome:** 100/100 requests succeeded. Throughput rose from 0.398 images/s at concurrency 1 to roughly 0.74–0.77 images/s at concurrency 2–8, then mostly plateaued. Per-request latency increased with concurrency as requests waited behind other work. SGLang's inference time stayed near 1.16 s on average; the remaining measured time includes queueing, JSON/base64 encoding, HTTPS routing, and response transfer.

\* p95 uses the nearest-rank definition: sorted sample at rank `ceil(0.95 × n)`; with 12–13 samples this is the maximum observation. Per-request measurements and exact run configuration are in [`benchmark-results-tsdocode-2026-09-27.json`](benchmark-results-tsdocode-2026-09-27.json). These observations describe this single warmed-up run, not a production SLA or a VKong network-only latency measurement.
