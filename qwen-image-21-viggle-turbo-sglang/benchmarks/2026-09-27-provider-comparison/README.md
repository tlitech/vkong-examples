# Provider comparison — 2026-09-27

Point-in-time provider-forced checks using VKong CLI `v0.10.16` and the personal
`dotieuthien` workspace. The provider flag is an operator/E2E control, not part
of the public provider-neutral workflow.

## Outcome

| Route | Policy | Outcome |
|---|---|---|
| Vast.ai RTX 5090 | `verified_only: false` | Container ran, but the test was manually stopped before agent Ready. |
| RunPod RTX 5090 | `verified_only: false` | Two rentals reached Ready, then CUDA initialization failed. The retry reported CUDA error 804. |
| RunPod RTX 5090 | `verified_only: true` | Offers appeared but disappeared before allocation. |
| RunPod A100 80 GB | `verified_only: true` | First rental did not reach Ready in about 18 minutes; the retry passed localhost and public generation. |

The successful A100 retry downloaded the partially cached base checkpoint in
115 seconds and the uncached LoRA in 17 seconds. A cached deploy restart took
about 94 seconds. VKong cannot currently isolate provider wrapped-image pull
from the rest of setup, so no container byte-rate is claimed.

## Localhost versus public HTTPS

Both requests used the same 1024×1024 prompt and seed and produced the same PNG
SHA-256 (`f987a84d…56aa6`).

| Path | End-to-end | Server inference | Non-inference overhead | Response |
|---|---:|---:|---:|---:|
| Localhost tunnel | 11.900 s | 5.167 s | 6.732 s | 2,732,837 bytes |
| Public HTTPS | 12.251 s | 2.471 s | 9.780 s | 2,732,838 bytes |

The localhost call was the first post-warmup user request, so its server time is
not a controlled network-only baseline. Public multipart image upload was not
reliable: the first 2.05 MB edit request failed during TLS handshake and its one
retry stalled while sending for more than 90 seconds.

All test rentals were stopped. Final workspace state was `active=0`,
`quota_used=0`, and `reserved_usd=0`.
