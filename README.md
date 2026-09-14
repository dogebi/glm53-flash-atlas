# GLM-5.3-Flash — Tensor Atlas v4 (offline single file)

`index.html` is a self-contained WebGL2 architecture atlas for **zai-org/GLM-5.3-Flash**
(320B total / 18B active hybrid MoE). Open it from disk — no server, no network, no build step
at runtime. Built with the `tensor-atlas-v4-retarget` skill: the nisten *Tensor Atlas v4* engine
(`LLMViz-DeepSeek-V4.1-Flash`, MIT) is kept whole; only the model half is replaced.

## What the atlas shows

Every byte figure is the sum of tensor byte ranges read from the published **safetensors headers
over HTTP Range**. No weights were downloaded, and nothing is a shape × bytes estimate.

| mode | repository | shards | raw tensors | payload |
|------|------------|-------:|------------:|--------:|
| BF16 | `zai-org/GLM-5.3-Flash-BF16` | 120 | 38,770 | **642.6467 GB** |
| FP8  | `zai-org/GLM-5.3-Flash` (published) | 62 | 76,108 | **328.3268 GB** |
| NVFP4| `nvidia/GLM-5.3-Flash-NVFP4` | 33 | 147,661 | **204.4191 GB** |

The build asserts `page total == measured payload` for all three modes
(`GATE payload == measured ✔`); the page's ledger total is a sum over the same measured numbers.

### Architecture as measured

- **46 stored layers**: a 45-layer stack plus **layer 45 = the MTP layer** (its own MoE block plus
  `enorm` / `hnorm` / `eh_proj`), shown as its own module.
- **Hybrid attention**: 34 gated-linear layers (q/k/v projections with short depthwise convs,
  per-channel decay, output gate, per-head norm, fixed-size recurrent state) and 11 sparse-MLA
  layers at 3, 7, 11 … 43 (q compressed to rank 1536, KV compressed to rank 512, plus a 128-dim,
  32-head **indexer** that keeps the top 2,048 key positions).
- **MoE**: 288 routed experts per layer, top-8, sigmoid scoring with a learned bias
  (`noaux_tc`, `n_group` 1), one always-on shared expert; `moe_intermediate_size` 2048 over a
  4096-wide residual stream. Layers 0–2 are dense (`intermediate_size` 12288).
- **mHC**: every layer carries `hc_attn_*` and `hc_ffn_*` triples (base / fn / scale) — the
  manifold-constrained hyper-connection weights, a category of its own in the ledger.
- **Vision tower**: 24 blocks × 14 tensors (hidden 1024, attention bias, QK norms) plus the patch
  downsample and the merger.
- `vocab_size` 154,880, untied embedding and head, 1M-token context.

### Folding rules (why 147,661 raw tensors become 38,752 logical rows)

- `.weight_scale_inv` and friends fold into their owner tensor.
- Per-expert tensors (`mlp.experts.N.{gate,up,down}_proj.weight`) fold into two banks per layer
  (`experts.gate_up_proj`, `experts.down_proj`); the same rule is applied to all three modes, so the
  ledger lists the same logical tensors whether the repository writes experts fused or one by one.
- Vision blocks fold into one bank per suffix, multiplied by 24 and labelled as such.

## Files

| file | purpose |
|------|---------|
| `index.html` | the deliverable — offline single file, ~660 KB |
| `src.html` | the untouched engine (kept for the shell) |
| `build.py` | retarget builder: fold → data half → gates → panels/regex copy → preflight → write |
| `panels.py` | whole-line panel rewrites (engine's own lines, anchored on its text) |
| `objectcode.py` | line/rewrite helpers with structural checks |
| `preflight.py` | static structural gate (`--selftest` proves the checker can still fail) |
| `measure.py` | safetensors header measurement over HTTP Range |
| `hf-bf16.json`, `hf-fp8.json`, `hf-nvfp4.json` | measured headers (per-tensor bytes + shapes) |
| `hf-config-*.json` | the published configs (BF16 reference, FP8, NVFP4 exclude list) |
| `adapt_step1..7.py` | the retarget steps applied to the copied builder, kept for audit |

## Rebuild

```bash
python3 build.py            # full build: gates + panels + preflight, writes index.html
python3 build.py --data-only # data half only, for inspection
python3 preflight.py .       # structural check on its own
python3 preflight.py --selftest   # prove the checker still fails on a broken page
```

The build refuses to write `index.html` when the payload gate or the preflight fails (exit 3 / 7),
and refuses to start if the preflight's self-test fails (exit 8).

## Verification performed

- payload gate: 642.6467 / 328.3268 / 204.4191 GB == measured headers ✔
- preflight: self-test passed + 10/10 structural checks passed ✔
- browser acceptance: page loads offline, 3 view tabs, 3 precision modes × 2 clicks each — **0 JS
  errors**, no view loss, no errors after switching modes; no DeepSeek/Engram/40-layer leftovers in
  the visible copy ✔

## Model facts and attribution

Architecture is read from the published configs (`Glm5NextForConditionalGeneration`, model type
`glm5_next`); capability and parameter-count claims belong to the model card of
`zai-org/GLM-5.3-Flash`. This atlas is an independent byte audit and does not evaluate the model.

Engine: *Tensor Atlas v4* by nisten (MIT) — see `LICENSE`.
