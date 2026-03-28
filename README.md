# Growth Model

> Growing language models one layer at a time.

**Date:** March 28, 2026
**Author:** Gratnics

---

## What is this?

Growth Model is a method for building a larger language model from a smaller pre-trained one, without ever training the whole thing at once.

The idea: instead of expanding the full model in one shot, we train one new layer at a time. Each new layer learns from the corresponding layer of the parent model, using pre-cached activations as a distillation target. The parent model never needs to be in GPU memory during the spawn phase — only one child layer and two small projection layers sit on the GPU at any given time.

**Important caveat:** the memory savings apply specifically to the spawn phase. End-to-end fine-tuning of the assembled model is still required (~975 MB peak), so the pipeline as a whole is not uniformly low-memory. The spawn-phase reduction is nonetheless the central contribution — it is what makes it possible to train individual layers on hardware that could not hold the full model.

---

## Results (WikiText-2)

All experiments ran on a single NVIDIA RTX 5070 (12 GB VRAM).

### Generation 1: Parent → Child

| Model | Params | Valid PPL | Test PPL | Notes |
|---|---|---|---|---|
| Parent (pre-trained) | 17.3M | 87.24 | 76.94 | Scratch, 3 epochs |
| Child — spawn only | 32.1M | 127.92 | 113.63 | Before fine-tuning |
| **Child — spawn + fine-tune** | **32.1M** | **80.22** | **71.34** | After 3-epoch FT |
| Same-arch scratch baseline | 32.1M | 290.78 | 256.22 | Same architecture, no spawn† |
| Standard decoder baseline | 31.7M | 99.65 | 87.35 | Plain decoder, same epochs‡ |

### Generation 2: Child → Next Child

| Model | Params | Valid PPL | Test PPL | Notes |
|---|---|---|---|---|
| Child parent (fine-tuned) | 32.1M | 80.22 | 71.34 | Teacher for gen 2 |
| Next child — spawn only | 42.1M | 81.60 | 72.43 | Before fine-tuning |
| **Next child — spawn + fine-tune** | **42.1M** | **76.45** | **68.23** | After 3-epoch FT |

### Peak GPU Memory During Spawn

| Phase | Peak GPU Memory |
|---|---|
| Parent pre-training | 5,950 MB |
| **Gen 1 layer spawn (per layer)** | **188–190 MB** |
| **Gen 2 layer spawn (per layer)** | **239–242 MB** |
| Child fine-tuning | ~975 MB |
| Next child fine-tuning | ~1,195 MB |

Spawn-phase reduction: **~97%** relative to full-model pre-training.

---

### Reading the numbers honestly

† The same-architecture scratch baseline uses the identical `LayerGrowthModel` structure (interface width 256, internal width 512, 6 layers) initialized randomly and trained end-to-end for 3 epochs — no caching, no spawn. Its validation PPL was still declining steeply at epoch 3 (813 → 346 → 290), meaning it had not converged. A fair quality comparison would require running it significantly longer.

‡ The standard decoder baseline uses a plain decoder-only transformer (d\_model=384, 7 layers, ~31.7M params) — a different architecture — trained for 3 epochs.

The Growth Model pipeline uses approximately **6× more optimization steps** in total (parent pre-training + spawn + fine-tuning ≈ 21,400 steps) compared to the 3,567 steps used by the scratch baselines. Quality comparisons should be read with this in mind.

---

## How it works

```text
Stage 1: Pre-train a small parent model M_p
Stage 2: Cache all layer activations to disk (run M_p once over training data)
Stage 3: Spawn child layers one at a time
          → only ~188–242 MB GPU memory per layer (spawn phase)
          → parent never re-enters GPU memory
Stage 4: Assemble the child model M_c
Stage 5: End-to-end fine-tuning to resolve inter-layer coherence
          → required step, ~975 MB peak GPU memory
```

Once the parent's activations are cached, the parent model is removed from GPU memory entirely. Each child layer is trained using the cached inputs and outputs of the corresponding parent layer as a distillation target. After all layers are saved, the assembled model undergoes fine-tuning to allow layers to coordinate with each other's actual outputs.

---

## Files

```text
my_model.py            Transformer architecture (built from scratch)
                       RMSNorm, RoPE, GQA, SwiGLU — no external model weights

layer_spawn_custom.py  Layer Spawn implementation
                       Weight tiling initialization + layer-wise distillation

experiment.py          Full experiment pipeline
                       Pre-train → cache → spawn → fine-tune → eval → plot
```

---

## Quick Start

```bash
pip install torch transformers matplotlib

# Interactive menu
python experiment.py

# Full parent → child pipeline
python experiment.py --mode parent_to_child

# Set child size explicitly
python experiment.py --mode all --child-multiplier 1.85

# Child → child (second generation)
python experiment.py --mode child_to_child

# Step by step
python experiment.py --mode pretrain
python experiment.py --mode cache
python experiment.py --mode spawn
python experiment.py --mode finetune
python experiment.py --mode eval
python experiment.py --mode plot

# Same-architecture scratch baseline (run before eval/plot for comparison)
python experiment.py --mode standard32m
python experiment.py --mode eval
python experiment.py --mode plot

# Second-generation steps
python experiment.py --mode next_cache
python experiment.py --mode next_spawn
python experiment.py --mode next_finetune
python experiment.py --mode next_eval
python experiment.py --mode next_plot
```

Interactive menu options:

```text
What would you like to build?
  1. Build a child model from a parent model
  2. Build a next child using an existing child model as the parent
  3. Quit
```

Results appear in `results/` as CSV files and PNG graphs.

---

## Architecture

Built from scratch — no Hugging Face model weights used.

- **RMSNorm** — faster than LayerNorm; used in LLaMA / Mistral
- **RoPE** — rotary position embeddings (Su et al., 2021)
- **Grouped Query Attention (GQA)** — fewer KV heads for memory efficiency at inference
- **SwiGLU FFN** — gated feed-forward used in LLaMA / GPT-4

The child model is a `LayerGrowthModel`: the residual stream and embedding dimension remain at `d_p` (parent width), while each transformer block projects up to `d_c` internally via `proj_up` / `proj_down` linear layers. This is a non-standard architecture — see Limitations.

Default configurations:

| Name | Interface `d` | Internal `d` | Layers | Params |
|---|---|---|---|---|
| Parent (Small) | 256 | 256 | 6 | ~17M |
| Child (Growth) | 256 | 512 | 6 | ~32M |
| Next child (Growth) | 256 | 640 | 6 | ~42M |
| Large | 1024 | 1024 | 16 | ~350M |
| XL | 2048 | 2048 | 24 | ~1.3B |

---

## Limitations

- **Spawn-phase savings only.** The ~97% GPU memory reduction applies to the spawn steps. End-to-end fine-tuning (~975 MB) is required and loads the full assembled model.
- **Non-standard architecture.** `LayerGrowthModel` with `proj_up` / `proj_down` projections is difficult to train from scratch — the same-arch baseline reached only PPL 290 after 3 epochs with validation loss still declining. Part of the method's quality advantage may stem from this architectural property rather than the growth procedure itself.
- **Compute is not matched.** The full Growth Model pipeline uses ~6× more optimization steps than the scratch baselines. Quality comparisons are not apples-to-apples.
- **Storage cost.** Caching all layer activations requires ~28.5 GB on disk for 6 layers at this scale.
- **Small scale only.** All experiments used 17M–42M parameter models on WikiText-2. Larger scales and other datasets are untested.
- **Diminishing returns across generations.** Each generation improves over its teacher, but by a smaller margin (8% → 4.7%).

---

## Paper / Blog

Full write-up: *https://gratnics.com/research/GrowthModel*

---

## License

MIT
