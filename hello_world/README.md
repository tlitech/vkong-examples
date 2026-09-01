# hello_world — CPU example (ubuntu:24.04)

Minimal FastAPI app on a CPU instance. All scripts are inline in `.vkong.config`.

| | |
|--|--|
| **Image** | `ubuntu:24.04` |
| **GPU** | None (CPU only) |

## Run

```bash
cd hello_world
vkong run -C . --destroy=false --keep-alive
```

> **Note:** Local port changes each session. Check `vkong` output for the actual port.

When finished, run `vkong app stop hello-world` to destroy the active rental and stop billing.

## Smoke test

```bash
curl -fsS http://127.0.0.1:<local-port>/
# => {"ok": true, "service": "hello", "port": 8000}
```
