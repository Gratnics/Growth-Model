# Growth Model

> Growing language models one layer at a time.

**Date:** March 26, 2026

---

## What is this?

Growth Model is a method for building a larger language model from a smaller pre-trained one, without ever training the whole thing at once.

The idea: instead of expanding the full model in one shot, we train one new layer at a time. Each new layer learns from the corresponding layer of the parent model, using pre-cached activations as training data. This means the parent model never needs to be in GPU memory during the spawn phase - only one child layer and two small projection layers sit on the GPU at any given time.

The result in our experiment: **97% GPU memory reduction** during layer spawning (187 MB vs 5,950 MB), and a child model that outperforms its parent after a short fine-tuning pass.

---

## Historical Results (WikiText-2)

| Model | Params | Valid PPL | Test PPL |
|---|---|---|---|
| Parent (pre-trained) | 17.3M | 118.76 | 103.31 |
| Child - spawn only | 32.1M | 196.00 | 173.36 |
| **Child - spawn + fine-tune** | **32.1M** | **95.83** | **84.80** |

Hardware: single NVIDIA RTX 5070 (12 GB VRAM).

Current default dataset in the codebase: **WikiText-2**.
Default source: **Hugging Face `Salesforce/wikitext` (`wikitext-2-v1`)**.

Layer-cache distillation uses up to the first **20,000** training sequences by default.

If needed, you can switch dataset source with `GROWTHMODEL_DATASET_SOURCE=legacy_zip` or `GROWTHMODEL_DATASET_SOURCE=auto`.

---

## How it works

```text
Stage 1: Pre-train a small parent model M_p
Stage 2: Cache all layer activations to disk (run M_p once)
Stage 3: Spawn child layers one at a time
          - only ~187 MB GPU memory per layer
Stage 4: Assemble the child model M_c
Stage 5: Short end-to-end fine-tuning to fix inter-layer coherence
```

The key insight is that once the parent's activations are cached, the parent model can be completely removed from GPU memory. Each child layer is then trained using the cached inputs and outputs of the corresponding parent layer as a distillation signal.

---

## Files

```text
my_model.py            Transformer architecture (from scratch)
                       RMSNorm, RoPE, GQA, SwiGLU - no external model libs

layer_spawn_custom.py  Layer Spawn implementation
                       Weight expansion + layer-wise distillation

experiment.py          Full experiment pipeline
                       Pre-train -> cache -> spawn -> fine-tune -> eval -> plot
```

---

## Quick Start

```bash
pip install torch transformers matplotlib

# Start with the menu
python experiment.py

# Run the full parent -> child pipeline directly
python experiment.py --mode parent_to_child

# Set the child size explicitly from the command line
python experiment.py --mode all --child-multiplier 1.85

# Run the child -> child pipeline directly
python experiment.py --mode child_to_child

# Or run step by step
python experiment.py --mode pretrain
python experiment.py --mode cache
python experiment.py --mode spawn
python experiment.py --mode finetune
python experiment.py --mode baseline
# alias for the same-architecture scratch baseline
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

Menu options:

```text
What would you like to build?
  1. Build the foundation from a parent model to a child model
  2. Build a child model using an existing child model as the parent
  3. Quit
```

`child_to_child` now uses the previously created child model as the teacher model and builds a next child into separate cache, checkpoint, and result files.

To compare Growth Model against a normally trained model with the same architecture as the grown child, run the same-architecture scratch baseline once before `eval` and `plot`:

```bash
python experiment.py --mode standard32m
python experiment.py --mode eval
python experiment.py --mode plot
```

This baseline uses the same `LayerGrowthModel` structure as the child model: interface width `256`, internal width `512`, `6` layers, and random initialization. It is then trained end-to-end for the same `3` epochs as the child fine-tuning stage, but without cache generation or spawn.

Additional figure outputs now include:

- `perplexity_comparison.png` with the optional same-architecture baseline bar when available
- `validation_ppl_by_epoch.png` for epoch-by-epoch validation PPL
- `spawn_loss_by_layer.png` for per-layer spawn convergence

When you start the first-generation `all` pipeline interactively, the script can also ask how large the child model should be relative to the parent. The chosen child config is saved and reused by later `spawn`, `finetune`, `eval`, and `child_to_child` runs.

Results will appear in `results/` as CSV files and PNG graphs.

---

## Architecture

Built from scratch - no Hugging Face model weights used.

- **RMSNorm** - faster than LayerNorm, used in LLaMA / Mistral
- **RoPE** - rotary position embeddings (Su et al., 2024)
- **Grouped Query Attention (GQA)** - fewer KV heads for memory efficiency
- **SwiGLU FFN** - gated feed-forward, used in LLaMA / GPT-4

Default configs:

| Name | d_model | Layers | Params |
|---|---|---|---|
| Small (parent) | 256 | 6 | ~17M |
| Medium (child) | 512 | 6 | ~32M |
| Same-architecture scratch baseline | interface 256 / internal 512 | 6 | ~32.1M |
| Large | 1024 | 16 | ~350M |
| XL | 2048 | 24 | ~1.3B |

---

## Limitations

- Caching activations to disk is storage-heavy (~28 GB for 6 layers in this experiment)
- End-to-end fine-tuning after assembly is required - spawn-only is worse than the parent
- Only tested at small scale (17M -> 32M). Larger scales are untested.

---

## Paper / Blog

Full write-up: *https://gratnics.com/research/Growth-Model-Growing-LLMs-One-Layer-at-a-Time*

---

## License

MIT
