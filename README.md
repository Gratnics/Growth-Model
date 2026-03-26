# Growth Model

> Growing language models one layer at a time.

**Date:** March 26, 2026

---

## What is this?

Growth Model is a method for building a larger language model from a smaller pre-trained one, without ever training the whole thing at once.

The idea: instead of expanding the full model in one shot, we train one new layer at a time. Each new layer learns from the corresponding layer of the parent model, using pre-cached activations as training data. This means the parent model never needs to be in GPU memory during the spawn phase — only one child layer and two small projection layers sit on the GPU at any given time.

The result in our experiment: **97% GPU memory reduction** during layer spawning (187 MB vs 5,950 MB), and a child model that outperforms its parent after a short fine-tuning pass.

---

## Results (WikiText-2)

| Model | Params | Valid PPL | Test PPL |
|---|---|---|---|
| Parent (pre-trained) | 17.3M | 118.76 | 103.31 |
| Child — spawn only | 32.1M | 196.00 | 173.36 |
| **Child — spawn + fine-tune** | **32.1M** | **95.83** | **84.80** |

Hardware: single NVIDIA RTX 5070 (12 GB VRAM).

---

## How it works

```
Stage 1: Pre-train a small parent model M_p
Stage 2: Cache all layer activations to disk (run M_p once)
Stage 3: Spawn child layers one at a time
          → only ~187 MB GPU memory per layer
Stage 4: Assemble the child model M_c
Stage 5: Short end-to-end fine-tuning to fix inter-layer coherence
```

The key insight is that once the parent's activations are cached, the parent model can be completely removed from GPU memory. Each child layer is then trained using the cached inputs and outputs of the corresponding parent layer as a distillation signal.

---

## Files

```
my_model.py            Transformer architecture (from scratch)
                       RMSNorm, RoPE, GQA, SwiGLU — no external model libs

layer_spawn_custom.py  Layer Spawn implementation
                       Weight expansion + layer-wise distillation

experiment.py          Full experiment pipeline
                       Pre-train → cache → spawn → fine-tune → eval → plot
```

---

## Quick Start

```bash
pip install torch transformers matplotlib

# Run everything (pre-train → spawn → fine-tune → eval → plot)
python experiment.py --mode all

# Or step by step
python experiment.py --mode pretrain
python experiment.py --mode cache
python experiment.py --mode spawn
python experiment.py --mode finetune
python experiment.py --mode eval
python experiment.py --mode plot
```

Results will appear in `results/` as CSV files and PNG graphs.

---

## Architecture

Built from scratch — no Hugging Face model weights used.

- **RMSNorm** — faster than LayerNorm, used in LLaMA / Mistral
- **RoPE** — rotary position embeddings (Su et al., 2024)
- **Grouped Query Attention (GQA)** — fewer KV heads for memory efficiency
- **SwiGLU FFN** — gated feed-forward, used in LLaMA / GPT-4

Default configs:

| Name | d_model | Layers | Params |
|---|---|---|---|
| Small (parent) | 256 | 6 | ~17M |
| Medium (child) | 512 | 6 | ~32M |
| Large | 1024 | 16 | ~350M |
| XL | 2048 | 24 | ~1.3B |

---

## Limitations

- Caching activations to disk is storage-heavy (~28 GB for 6 layers in this experiment)
- End-to-end fine-tuning after assembly is required — spawn-only is worse than the parent
- Only tested at small scale (17M → 32M). Larger scales are untested.

---

## Paper / Blog

Full write-up: *https://gratnics.com/research/Growth-Model-Growing-LLMs-One-Layer-at-a-Time*

---

## License

MIT
