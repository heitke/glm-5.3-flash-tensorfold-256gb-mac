# GLM-5.3-Flash on a 256 GB Apple Silicon Mac — a TensorFold recipe

Run zai-org's GLM-5.3-Flash (355B MoE, 4-bit MLX quant with MTP, ~170 GB on disk) on a single
Mac Studio M3 Ultra with 256 GB of unified memory, served by [TensorFold](https://github.com/ashhart/TensorFold).
No cluster, no external GPU box — one machine, one command.

**My rig:** Mac Studio M3 Ultra, 256 GB unified memory, 1 TB SSD, macOS 26.6.
TensorFold v1.1.5 (`f38a6a9d`), Metal backend. The model id served: `glm-5.3-flash-mlx-4bit-mtp` (MTP = multi-token prediction heads carried in the checkpoint, which is what makes speculative decoding work).

## Why this works at all

A 170 GB model on a 256 GB machine sounds like it shouldn't fit — macOS by default only lets the
GPU *wire* (pin, un-pagable) 75 % of unified memory, i.e. 192 GiB. Two things resolve the tension:

1. **The engine doesn't wire everything.** Only the hot working set of each layer must be
   GPU-resident; the rest lives in pageable memory and gets paged on demand. My steady-state
   footprint is ~183–197 GB *process footprint* but only ~189 GB actually wired to the GPU.
2. **macOS lets you raise the cap** with `sudo sysctl -w iogpu.wired_limit_mb=222822` (85 %).
   On a reboot this resets — see the launchd snippet below to persist it.

**The honest answer to the obvious question:** you can run this model *without touching the
sysctl at all.* It just barely fits under the default 192 GiB wired cap, and I ran it that way
for weeks. The trade-off is concurrency — the wired budget leaves room for essentially one
heavy streaming request (a second client's prompt prefill competes for the same pool and you
feel it as latency, sometimes outright cache evictions between sessions). Raising the cap to
85 % gives TensorFold ~28 GiB more headroom and I now run it with the engine's own load limit
set to match, but the single-user experience is identical either way.

## Recipe

### 1. Get the weights

Download `glm-5.3-flash-mlx-4bit-mtp` (4-bit MLX + MTP heads) from Hugging Face
(`zai-org/GLM-5.3-Flash-MLX-4bit` or equivalent — the folder must carry the MTP weights,
a checkpoint without them decodes 1 token/step). ~170 GB free on an external Thunderbolt SSD
works fine; internal disk also works if you have the room.

### 2. (Optional but recommended) raise the GPU wired limit

```bash
sudo sysctl -w iogpu.wired_limit_mb=222822   # 85 % of 256 GiB = 218 GiB
```

Skip this and the model still runs — you just live inside 192 GiB of wired memory.

To persist across reboots, drop this into `/Library/LaunchDaemons/com.user.iogpu-wired-limit.plist`
(sysctl settings reset on boot; `/etc/sysctl.conf` is ignored on modern macOS):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.user.iogpu-wired-limit</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>/usr/sbin/sysctl -w iogpu.wired_limit_mb=222822</string>
    </array>
    <key>RunAtLoad</key><true/>
</dict>
</plist>
```

### 3. Build TensorFold

```bash
git clone https://github.com/ashhart/TensorFold
cd TensorFold
zig build native -Dversion=1.1.5   # match the repo's .zig-version
```

### 4. Serve

```bash
GLM_LOAD_LIMIT_GB=218 zig-out/native/bin/tensorfold-native serve \
    /path/to/glm-5.3-flash-mlx-4bit-mtp \
    --port 8420 --dashboard --name glm-5.3-flash \
    --max-tokens 32768 --context 500000 \
    --idle-unload 900
```

- `GLM_LOAD_LIMIT_GB` is the engine's own admission limit; 218 keeps it in lockstep with the
  85 % sysctl. If you skipped step 2, set `GLM_LOAD_LIMIT_GB=192` (or lower — the engine
  clamps to what it can actually wire) and don't raise it past the OS cap or you'll get Metal
  allocation failures instead of a clean refusal.
- `--idle-unload 900` frees the ~180 GB after 15 idle minutes so the machine stays usable;
  the next request reloads (~3–4 min cold start). Drop the flag if you want it always resident.
- Point any OpenAI-compatible client at `http://localhost:8420/v1`. First token on a cold cache
  is ~1.6 s; a warm continuing turn resumes from the prompt cache in milliseconds.

## Benchmarks (measured, Mac Studio M3 Ultra 256 GB, TensorFold 1.1.5, Metal)

| Metric | Value |
|---|---|
| Decode speed (single stream, MTP speculative on) | **~40 tok/s sustained** (35–75 tok/s observed depending on acceptance rate; the engine's own live gauge reads 66–75) |
| Prefill, cold cache | **~360 tok/s** (3,020-token prompt in 8.4 s) |
| Prefill, cache hit | **~50,000 tok/s effective** (same prompt resumed in 0.06 s — 3,018 of 3,020 tokens from cache) |
| TTFT, short prompt, warm engine | **~0.4 s** |
| TTFT, long prompt behind a cached prefix | ~1.6 s |
| Process footprint, steady state | ~184 GB footprint, ~189 GB GPU-wired, ~198 GB peak |
| Cold start (weights read from Thunderbolt SSD) | ~3–4 min to first token |
| Context window | 500,000 positions configured; the 4-bit checkpoint's KV at that depth fits inside the 218 GiB budget with room to spare |

MTP acceptance on real coding/prose traffic ran ~60–65 % (rounds=1283, accepted≈1306/2010 in one
measured turn — acceptance above 1.0 per round because a round drafts multiple tokens).

## Things I learned the hard way

- **Don't set the engine's load limit above the OS wired cap.** The refusal you want is a clean
  `limit exceeded` message, not a Metal buffer allocation failure mid-layer.
- **Concurrent traffic evicts prompt-cache states.** A second client's cold prefill can evict a
  long session's resident cache state — you'll see `resumed at 0` in the log and a 10-minute
  re-prefill at 260k tokens. Single heavy user = no problem; multiple simultaneous long
  conversations = mind the cache budget (`--context 500000` and the 16 GiB prompt-cache budget
  interact here).
- **Swap fills before you crash.** At the default 75 % cap the system spilled ~5 GB to swap
  under load and kept running fine for weeks. The failure mode when you exceed the wired cap is
  engine-level OOM (Metal allocation failure), not a kernel panic — but if you raise the cap
  much past 85 %, you *can* starve the kernel and watchdog-panic the machine. 85 % is the
  defensible ceiling.
- The dashboard at `http://localhost:8420/dashboard` shows live decode tok/s, MTP acceptance,
  and memory gauges — worth having open while you tune.

## License

MIT for this README. TensorFold is Apache-2.0 (upstream); the GLM-5.3-Flash checkpoint carries
its own license from zai-org — check it before any use beyond research.
