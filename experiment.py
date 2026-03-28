import os, csv, time, math, json, argparse, urllib.request, sys, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from my_model import MyModel, ModelConfig, TransformerBlock, RMSNorm
from layer_spawn_custom import LayerExpander


class Config:
    DATASET_NAME = "WikiText-103"
    DATASET_PREFIX = "wikitext103"
    DATASET_ARCHIVE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    DATASET_ARCHIVE_NAME = "wikitext-103-v1.zip"
    DATASET_EXTRACT_DIR = "wikitext-103"
    DATASET_SPLIT_FILES = {
        "train": "wiki.train.tokens",
        "valid": "wiki.valid.tokens",
        "test": "wiki.test.tokens",
    }
    PARENT = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=256,
                         n_layers=6, n_heads=8, n_kv_heads=4,
                         ffn_mult=4, dropout=0.1, bias=False)
    DEFAULT_CHILD = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=512,
                                n_layers=6, n_heads=16, n_kv_heads=8,
                                ffn_mult=4, dropout=0.0, bias=False)
    CHILD  = DEFAULT_CHILD
    BASELINE = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=410,
                           n_layers=6, n_heads=5, n_kv_heads=5,
                           ffn_mult=4, dropout=0.0, bias=False)
    DEFAULT_NEXT_CHILD = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=640,
                                     n_layers=6, n_heads=20, n_kv_heads=10,
                                     ffn_mult=4, dropout=0.0, bias=False)
    NEXT_CHILD = DEFAULT_NEXT_CHILD
    PRETRAIN_EPOCHS = 3
    SPAWN_EPOCHS    = 2
    FINETUNE_EPOCHS = 3
    BASELINE_EPOCHS = 3
    BATCH_SIZE      = 16
    SEQ_LEN         = 128
    CACHE_MAX_SAMPLES = 20_000
    DATALOADER_WORKERS = 0 if os.name == "nt" else 2
    LR_PRETRAIN     = 3e-4
    LR_SPAWN        = 5e-5
    LR_FINETUNE     = 1e-4
    GRAD_CLIP       = 1.0
    DATA_DIR    = "./data"
    RESULTS_DIR = "./results"
    CKPT_DIR    = "./checkpoints"
    CACHE_DIR   = "./cache"
    NEXT_CACHE_DIR = "./cache_next_child"
    PARENT_PATH = "./checkpoints/parent.pt"
    CHILD_DIR   = "./checkpoints/child_layers"
    CHILD_FT_PATH = "./checkpoints/child_finetuned.pt"
    CHILD_CONFIG_PATH = "./checkpoints/child_config.json"
    BASELINE_PATH = "./checkpoints/scratch_baseline_32m.pt"
    NEXT_CHILD_DIR = "./checkpoints/next_child_layers"
    NEXT_CHILD_FT_PATH = "./checkpoints/next_child_finetuned.pt"
    NEXT_CHILD_GROWTH_FACTOR = 1.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Data

class TokenDataset(Dataset):
    def __init__(self, token_ids, seq_len, max_samples=None):
        if torch.is_tensor(token_ids):
            self.token_ids = token_ids.to(dtype=torch.long).contiguous()
        else:
            self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.n_samples = max(0, (len(self.token_ids) - seq_len - 1) // seq_len)
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.token_ids[start:start+self.seq_len]


class LayerCacheDataset(Dataset):
    def __init__(self, layer_idx, cache_dir=Config.CACHE_DIR):
        self.inputs  = torch.load(os.path.join(cache_dir, f"layer_{layer_idx}_input.pt"))
        self.outputs = torch.load(os.path.join(cache_dir, f"layer_{layer_idx}_output.pt"))
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.outputs[idx]


# Utils

def gpu_mb():      return torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0.
def gpu_peak_mb(): return torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.
def gpu_res_mb():  return torch.cuda.memory_reserved()/1024**2 if torch.cuda.is_available() else 0.
def reset_peak():
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

def compute_perplexity(model, token_ids, device):
    # Shift happens in-model.
    model.eval()
    dataset = TokenDataset(token_ids, Config.SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=Config.DATALOADER_WORKERS,
                         pin_memory=torch.cuda.is_available())
    total_loss, total_tokens = 0., 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            loss, _ = model(x, labels=x)
            total_loss   += loss.item() * x.numel()
            total_tokens += x.numel()
    return math.exp(total_loss / total_tokens)

def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        print(f"  Skipped empty CSV: {path}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


def runtime_dataset_metadata():
    return {
        "dataset_name": Config.DATASET_NAME,
        "dataset_prefix": Config.DATASET_PREFIX,
        "seq_len": Config.SEQ_LEN,
    }


def checkpoint_metadata_path(path):
    return f"{path}.metadata.json"


def directory_metadata_path(path):
    return os.path.join(path, "metadata.json")


def save_metadata(path, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {path}")


def load_metadata(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_metadata_compatibility(path, expected, label):
    metadata = load_metadata(path)
    if metadata is None:
        print(f"  Warning: {label} metadata not found at {path}; compatibility cannot be verified.")
        return

    mismatches = []
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            mismatches.append(f"{key}={metadata.get(key)!r} (expected {expected_value!r})")

    if mismatches:
        raise RuntimeError(f"{label} metadata mismatch: {', '.join(mismatches)}")


def model_config_to_dict(config):
    return {
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "ffn_mult": config.ffn_mult,
        "dropout": config.dropout,
        "bias": config.bias,
    }


def model_config_from_dict(data):
    return ModelConfig(**data)


def estimate_decoder_model_params(config):
    block = TransformerBlock(config)
    block_params = sum(p.numel() for p in block.parameters())
    return (config.vocab_size * config.d_model + config.d_model + config.n_layers * block_params) / 1e6


def estimate_growth_model_params(parent_config, child_config):
    block = TransformerBlock(child_config)
    block_params = sum(p.numel() for p in block.parameters())
    interface_dim = parent_config.d_model
    total_params = (
        parent_config.vocab_size * interface_dim
        + interface_dim
        + parent_config.n_layers * (block_params + 2 * interface_dim * child_config.d_model)
    )
    return total_params / 1e6


def derive_next_child_config(child_config):
    head_dim = child_config.d_model // child_config.n_heads
    step = head_dim * 2
    target_d_model = math.ceil(child_config.d_model * Config.NEXT_CHILD_GROWTH_FACTOR / step) * step
    target_d_model = max(child_config.d_model + step, target_d_model)
    n_heads = target_d_model // head_dim
    n_kv_heads = max(1, n_heads // 2)
    return ModelConfig(
        vocab_size=child_config.vocab_size,
        max_seq_len=child_config.max_seq_len,
        d_model=target_d_model,
        n_layers=child_config.n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_mult=child_config.ffn_mult,
        dropout=child_config.dropout,
        bias=child_config.bias,
    )


def apply_runtime_child_config(child_config, persist=False):
    Config.CHILD = child_config
    Config.NEXT_CHILD = derive_next_child_config(child_config)
    if persist:
        os.makedirs(Config.CKPT_DIR, exist_ok=True)
        with open(Config.CHILD_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(model_config_to_dict(child_config), f, indent=2)


def load_saved_child_config():
    if not os.path.exists(Config.CHILD_CONFIG_PATH):
        return None
    with open(Config.CHILD_CONFIG_PATH, "r", encoding="utf-8") as f:
        return model_config_from_dict(json.load(f))


def parse_child_growth_input(raw_value):
    text = raw_value.strip().lower().replace("x", "")
    if not text:
        return None
    if text.endswith("%"):
        return 1.0 + float(text[:-1]) / 100.0
    value = float(text)
    return 1.0 + value / 100.0 if value > 5 else value


def choose_child_config_for_multiplier(parent_config, target_multiplier):
    parent_params = estimate_decoder_model_params(parent_config)
    target_params = parent_params * target_multiplier
    head_dim = parent_config.d_model // parent_config.n_heads
    step = head_dim * 2
    start_d_model = ((parent_config.d_model // step) + 1) * step
    max_d_model = max(4096, start_d_model)
    best_config, best_params, best_error = None, None, float("inf")

    for d_model in range(start_d_model, max_d_model + step, step):
        n_heads = d_model // head_dim
        if n_heads < 2 or n_heads % 2 != 0:
            continue
        n_kv_heads = max(1, n_heads // 2)
        if n_heads % n_kv_heads != 0:
            continue
        candidate = ModelConfig(
            vocab_size=parent_config.vocab_size,
            max_seq_len=parent_config.max_seq_len,
            d_model=d_model,
            n_layers=parent_config.n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_mult=parent_config.ffn_mult,
            dropout=Config.DEFAULT_CHILD.dropout,
            bias=parent_config.bias,
        )
        candidate_params = estimate_growth_model_params(parent_config, candidate)
        error = abs(candidate_params - target_params)
        if error < best_error:
            best_config, best_params, best_error = candidate, candidate_params, error

    return best_config, best_params, parent_params, target_params


def prompt_for_child_multiplier():
    current_child_params = estimate_growth_model_params(Config.PARENT, Config.CHILD)
    parent_params = estimate_decoder_model_params(Config.PARENT)
    default_multiplier = current_child_params / parent_params

    print("\nHow much larger should the child model be than the parent?")
    print(f"  Current child target: ~{current_child_params:.1f}M params ({default_multiplier:.2f}x parent)")
    print("  Enter a total multiplier like 1.85, or a percent increase like 85 for +85%.")
    print("  Press Enter to keep the current child size.")

    while True:
        try:
            raw_value = input("\nChild growth: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo child growth input received. Keeping the current child config.")
            return None
        if raw_value == "":
            return None
        try:
            multiplier = parse_child_growth_input(raw_value)
        except ValueError:
            print("Invalid value. Use something like 1.85 or 85.")
            continue
        if multiplier is None or multiplier <= 1.0:
            print("Please enter a value greater than 1.0 or a positive percent increase.")
            continue
        return multiplier


def configure_runtime_child_settings(mode, child_multiplier_arg):
    saved_child_config = load_saved_child_config()
    if saved_child_config is not None:
        apply_runtime_child_config(saved_child_config, persist=False)

    multiplier = child_multiplier_arg
    if isinstance(multiplier, str):
        multiplier = parse_child_growth_input(multiplier)
    should_prompt = mode == "all" and sys.stdin.isatty()
    if multiplier is None and should_prompt:
        multiplier = prompt_for_child_multiplier()

    if multiplier is not None:
        child_config, child_params, parent_params, target_params = choose_child_config_for_multiplier(
            Config.PARENT, multiplier)
        if child_config is None:
            raise ValueError(f"Could not find a child config for multiplier {multiplier}.")
        apply_runtime_child_config(child_config, persist=True)
        print("\nResolved parent -> child growth target")
        print(f"  Requested : {multiplier:.2f}x parent (~{target_params:.1f}M params)")
        print(f"  Selected  : d_model={Config.CHILD.d_model}, heads={Config.CHILD.n_heads}, kv_heads={Config.CHILD.n_kv_heads}")
        print(f"  Estimate  : ~{child_params:.1f}M params")
    elif saved_child_config is not None:
        saved_child_params = estimate_growth_model_params(Config.PARENT, Config.CHILD)
        print("\nUsing saved child config")
        print(f"  d_model={Config.CHILD.d_model}, heads={Config.CHILD.n_heads}, kv_heads={Config.CHILD.n_kv_heads}")
        print(f"  Estimate: ~{saved_child_params:.1f}M params")


def layer_forward(model, layer_idx, x):
    if hasattr(model, "proj_ups") and hasattr(model, "proj_downs"):
        return model.proj_downs[layer_idx](model.blocks[layer_idx](model.proj_ups[layer_idx](x)))
    return model.blocks[layer_idx](x)


def load_base_parent(device):
    ensure_metadata_compatibility(
        checkpoint_metadata_path(Config.PARENT_PATH),
        {
            **runtime_dataset_metadata(),
            "artifact": "parent_checkpoint",
            "model_config": model_config_to_dict(Config.PARENT),
        },
        "Parent checkpoint",
    )
    parent = MyModel(Config.PARENT).to(device)
    parent.load_state_dict(torch.load(Config.PARENT_PATH, map_location=device))
    return parent


def prepare_layer_parts_dir(path):
    os.makedirs(path, exist_ok=True)


def run_cache_for_model(data_cache, teacher_model, cache_dir, label):
    print("\n" + "="*55 + f"\n  Cache Generation: {label}\n" + "="*55)
    os.makedirs(cache_dir, exist_ok=True)
    device = Config.DEVICE

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    print(f"{label}: {teacher_model.count_params():.1f}M parameters")
    print(f"GPU memory usage (cache generation): {gpu_res_mb():.0f}MB")

    full_train_samples = max(0, (len(data_cache["train"]) - Config.SEQ_LEN - 1) // Config.SEQ_LEN)
    dataset = TokenDataset(data_cache["train"], Config.SEQ_LEN,
                           max_samples=Config.CACHE_MAX_SAMPLES)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                         shuffle=False, num_workers=Config.DATALOADER_WORKERS,
                         pin_memory=torch.cuda.is_available())

    n = len(teacher_model.blocks)
    layer_inputs  = [[] for _ in range(n)]
    layer_outputs = [[] for _ in range(n)]

    if len(dataset) < full_train_samples:
        print(f"\nUsing the first {len(dataset):,} / {full_train_samples:,} train samples for cache distillation.")
    print(f"\nGenerating cache for {len(dataset):,} samples...")
    t0 = time.time()

    with torch.no_grad():
        for bi, x in enumerate(loader):
            x = x.to(device)
            h = teacher_model.tok_emb(x)
            for l in range(n):
                layer_inputs[l].append(h.cpu())
                h = layer_forward(teacher_model, l, h)
                layer_outputs[l].append(h.cpu())
            if bi % 100 == 0:
                print(f"  Batch {bi}/{len(loader)}")

    print(f"  Computation complete: {time.time()-t0:.0f}s")

    total_mb = 0
    for l in range(n):
        inp = torch.cat(layer_inputs[l],  dim=0)
        out = torch.cat(layer_outputs[l], dim=0)
        pi  = os.path.join(cache_dir, f"layer_{l}_input.pt")
        po  = os.path.join(cache_dir, f"layer_{l}_output.pt")
        torch.save(inp, pi); torch.save(out, po)
        mb = (os.path.getsize(pi)+os.path.getsize(po))/1024**2
        total_mb += mb
        print(f"  Layer {l}: {mb:.0f}MB")

    save_metadata(
        directory_metadata_path(cache_dir),
        {
            **runtime_dataset_metadata(),
            "artifact": "layer_cache",
            "teacher_label": label,
            "teacher_layers": n,
            "teacher_interface_dim": teacher_model.tok_emb.weight.shape[1],
            "cached_samples": len(dataset),
            "cache_sample_cap": Config.CACHE_MAX_SAMPLES,
        },
    )
    print(f"\nTotal cache size: {total_mb:.0f}MB")

    del teacher_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Released teacher model from GPU. Remaining: {gpu_res_mb():.0f}MB")


WORKFLOW_MENU = {
    "1": "parent_to_child",
    "2": "child_to_child",
    "3": "quit",
    "parent_to_child": "parent_to_child",
    "child_to_child": "child_to_child",
    "quit": "quit",
    "q": "quit",
}


def print_banner():
    print("\nGrowth Model - Validation Experiment (Revised)")
    print(f"  Device   : {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  Parent   : d_model={Config.PARENT.d_model}, n_layers={Config.PARENT.n_layers}")
    print(f"  Child    : d_model={Config.CHILD.d_model},  n_layers={Config.CHILD.n_layers}")
    print(f"  Scratch  : d_model={Config.BASELINE.d_model}, n_layers={Config.BASELINE.n_layers}")
    print(f"  Next     : d_model={Config.NEXT_CHILD.d_model}, n_layers={Config.NEXT_CHILD.n_layers}")


def select_start_mode():
    print("\nWhat would you like to build?")
    print("  1. Build the foundation from a parent model to a child model")
    print("  2. Build a child model using an existing child model as the parent")
    print("  3. Quit")
    while True:
        try:
            choice = input("\nSelect an option [1-3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nNo selection received. Exiting.")
            return "quit"
        mode = WORKFLOW_MENU.get(choice)
        if mode:
            return mode
        print("Invalid selection. Please enter 1, 2, or 3.")


def resolve_mode(mode):
    if mode == "menu":
        return select_start_mode()
    return mode


def handle_workflow_mode(mode):
    if mode == "quit":
        print("\nExiting without running the experiment.")
        return True, None
    if mode == "parent_to_child":
        print("\nSelected workflow: parent model -> child model")
        return False, "all"
    if mode == "child_to_child":
        print("\nSelected workflow: child model -> child model")
        return False, "next_all"
    return False, mode


# Data prep

def tokenize_file(path):
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"  {os.path.basename(path)}: {len(ids):,} tokens")
    return ids

def ensure_dataset_split_paths():
    archive_path = os.path.join(Config.DATA_DIR, Config.DATASET_ARCHIVE_NAME)
    extract_root = os.path.join(Config.DATA_DIR, Config.DATASET_EXTRACT_DIR)
    split_paths = {
        split: os.path.join(extract_root, filename)
        for split, filename in Config.DATASET_SPLIT_FILES.items()
    }

    missing_splits = [split for split, path in split_paths.items() if not os.path.exists(path)]
    if not missing_splits:
        return split_paths

    if not os.path.exists(archive_path):
        print(f"  Downloading archive: {Config.DATASET_ARCHIVE_NAME}...")
        urllib.request.urlretrieve(Config.DATASET_ARCHIVE_URL, archive_path)
    else:
        print(f"  Using existing archive: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        archive_members = set(archive.namelist())
        for split in missing_splits:
            member = f"{Config.DATASET_EXTRACT_DIR}/{Config.DATASET_SPLIT_FILES[split]}"
            if member not in archive_members:
                raise FileNotFoundError(f"Missing dataset member in archive: {member}")
            print(f"  Extracting: {member}")
            archive.extract(member, Config.DATA_DIR)

    return split_paths

def prepare_data():
    print("\n" + "="*55 + f"\n  Data Preparation ({Config.DATASET_NAME})\n" + "="*55)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    split_paths = ensure_dataset_split_paths()
    cache = {}
    for split, path in split_paths.items():
        print(f"  Using dataset file: {path}")
        cp = os.path.join(Config.DATA_DIR, f"{Config.DATASET_PREFIX}_{split}.pt")
        if os.path.exists(cp):
            print(f"  Loading cache: {split}")
            loaded = torch.load(cp)
            cache[split] = loaded.to(dtype=torch.long).contiguous() if torch.is_tensor(loaded) \
                else torch.tensor(loaded, dtype=torch.long)
        else:
            cache[split] = tokenize_file(path)
            torch.save(cache[split], cp)
    return cache


# Pretrain

def run_pretrain(data_cache):
    print("\n" + "="*55 + "\n  Step 1: Parent Model Pre-training\n" + "="*55)
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    device = Config.DEVICE
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    model = MyModel(Config.PARENT).to(device)
    print(f"Parent model: {model.count_params():.1f}M parameters")

    dataset = TokenDataset(data_cache["train"], Config.SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                         shuffle=True, num_workers=Config.DATALOADER_WORKERS,
                         pin_memory=torch.cuda.is_available())
    print(f"Training samples: {len(dataset):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR_PRETRAIN,
                             betas=(0.9,0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=len(loader)*Config.PRETRAIN_EPOCHS,
        eta_min=Config.LR_PRETRAIN/10)

    log_rows, epoch_rows, step = [], [], 0
    for epoch in range(Config.PRETRAIN_EPOCHS):
        model.train()
        epoch_loss, t0 = 0., time.time()
        for x in loader:
            x = x.to(device)
            loss, _ = model(x, labels=x)  # Next-token loss.
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            opt.step(); sch.step()
            epoch_loss += loss.item(); step += 1
            if step % 100 == 0:
                lr = sch.get_last_lr()[0]
                log_rows.append({"step":step,"epoch":epoch+1,
                                  "train_loss":round(loss.item(),4),
                                  "lr":f"{lr:.2e}","gpu_mb":round(gpu_mb(),1)})
                print(f"  epoch {epoch+1} | step {step:5d} | loss {loss.item():.4f}"
                      f" | lr {lr:.2e} | GPU {gpu_mb():.0f}MB")

        val_ppl = compute_perplexity(model, data_cache["valid"], device)
        avg = epoch_loss / len(loader)
        epoch_rows.append({"epoch":epoch+1,
                           "avg_train_loss":round(avg,4),
                           "val_ppl":round(val_ppl,6)})
        print(f"\n  Epoch {epoch+1} complete | avg loss: {avg:.4f}"
              f" | valid PPL: {val_ppl:.2f} | {time.time()-t0:.0f}s"
              f" | GPU reserved: {gpu_res_mb():.0f}MB")
        for row in log_rows:
            if row["epoch"]==epoch+1 and "val_ppl" not in row:
                row["val_ppl"] = val_ppl

    torch.save(model.state_dict(), Config.PARENT_PATH)
    save_metadata(
        checkpoint_metadata_path(Config.PARENT_PATH),
        {
            **runtime_dataset_metadata(),
            "artifact": "parent_checkpoint",
            "model_config": model_config_to_dict(Config.PARENT),
        },
    )
    save_csv(os.path.join(Config.RESULTS_DIR, "pretrain_log.csv"), log_rows)
    save_csv(os.path.join(Config.RESULTS_DIR, "pretrain_epoch_metrics.csv"), epoch_rows)
    print(f"\nSaved parent model: {Config.PARENT_PATH}")


# Cache

def run_cache(data_cache):
    device = Config.DEVICE
    parent = load_base_parent(device)
    run_cache_for_model(data_cache, parent, Config.CACHE_DIR, "Parent model")


# Spawn

def run_spawn(_data_cache=None):
    device = "cpu"
    parent_cpu = load_base_parent(device)
    run_spawn_from_teacher(
        label="Parent model",
        teacher_model=parent_cpu,
        cache_dir=Config.CACHE_DIR,
        target_config=Config.CHILD,
        layer_dir=Config.CHILD_DIR,
        result_csv="spawn_log.csv",
        step_result_csv="spawn_step_log.csv",
    )


def build_spawn_modules_from_teacher(teacher_model, layer_idx, target_config, device):
    expander = LayerExpander()
    interface_dim = teacher_model.tok_emb.weight.shape[1]
    target_dim = target_config.d_model

    teacher_block = teacher_model.blocks[layer_idx]
    child_block = expander.expand_block(teacher_block, target_config).to(device)
    proj_up   = nn.Linear(interface_dim, target_dim, bias=False).to(device)
    proj_down = nn.Linear(target_dim, interface_dim, bias=False).to(device)

    with torch.no_grad():
        if hasattr(teacher_model, "proj_ups") and hasattr(teacher_model, "proj_downs"):
            proj_up.weight.copy_(
                expander.expand_linear(teacher_model.proj_ups[layer_idx].weight, target_dim, interface_dim))
            proj_down.weight.copy_(
                expander.expand_linear(teacher_model.proj_downs[layer_idx].weight, interface_dim, target_dim))
        else:
            proj_up.weight.zero_()
            proj_down.weight.zero_()
            eye_dim = min(interface_dim, target_dim)
            proj_up.weight[:eye_dim, :eye_dim].copy_(torch.eye(eye_dim, device=device))
            proj_down.weight[:eye_dim, :eye_dim].copy_(torch.eye(eye_dim, device=device))

    return child_block, proj_up, proj_down, interface_dim


def run_spawn_from_teacher(label, teacher_model, cache_dir, target_config, layer_dir, result_csv, step_result_csv):
    print("\n" + "="*55 + f"\n  Layer Spawn: {label}\n" + "="*55)

    ensure_metadata_compatibility(
        directory_metadata_path(cache_dir),
        {
            **runtime_dataset_metadata(),
            "artifact": "layer_cache",
            "teacher_label": label,
            "teacher_layers": len(teacher_model.blocks),
            "teacher_interface_dim": teacher_model.tok_emb.weight.shape[1],
        },
        f"{label} cache",
    )

    for l in range(len(teacher_model.blocks)):
        if not os.path.exists(os.path.join(cache_dir, f"layer_{l}_input.pt")):
            print(f"Cache not found in {cache_dir}. Generate it first.")
            return

    prepare_layer_parts_dir(layer_dir)
    device = Config.DEVICE
    teacher_model.eval()
    spawn_log = []
    spawn_step_rows = []

    for layer_idx in range(len(teacher_model.blocks)):
        print(f"\n--- Layer {layer_idx} ---")
        reset_peak()

        cache_ds = LayerCacheDataset(layer_idx, cache_dir=cache_dir)
        cache_ld = DataLoader(cache_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=0)

        c_block, proj_up, proj_down, interface_dim = build_spawn_modules_from_teacher(
            teacher_model, layer_idx, target_config, device)

        mem_start = gpu_mb()
        print(f"  GPU memory usage (child layer only): {mem_start:.0f}MB")

        opt = torch.optim.AdamW(
            list(c_block.parameters())+list(proj_up.parameters())+list(proj_down.parameters()),
            lr=Config.LR_SPAWN, weight_decay=0.01)

        c_block.train(); proj_up.train(); proj_down.train()

        layer_step = 0
        for epoch in range(Config.SPAWN_EPOCHS):
            total_loss = 0.
            for h_in, h_out_teacher in cache_ld:
                layer_step += 1
                h_in = h_in.to(device); h_out_teacher = h_out_teacher.to(device)
                h_large = proj_up(h_in)
                h_child = c_block(h_large)
                h_down  = proj_down(h_child)
                loss_mse = F.mse_loss(h_down, h_out_teacher)
                pn = F.layer_norm(h_out_teacher, [interface_dim])
                cn = F.layer_norm(h_down,        [interface_dim])
                teacher_probs = F.softmax(pn, dim=-1)
                loss_kl = F.kl_div(
                    F.log_softmax(cn, dim=-1),
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1).mean()
                loss = loss_mse + 0.1*loss_kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(c_block.parameters())+list(proj_up.parameters())+list(proj_down.parameters()),
                    Config.GRAD_CLIP)
                opt.step(); total_loss += loss.item()
                spawn_step_rows.append({
                    "layer_idx": layer_idx,
                    "epoch": epoch+1,
                    "step_in_layer": layer_step,
                    "loss": round(loss.item(), 6),
                })
            avg = total_loss/len(cache_ld)
            print(f"  Epoch {epoch+1} | loss: {avg:.6f}")

        peak = gpu_peak_mb()
        spawn_log.append({"layer_idx":layer_idx, "gpu_only_child_mb":round(mem_start,1),
                           "gpu_peak_mb":round(peak,1), "final_loss":round(avg,6)})
        print(f"  GPU peak (child layer only): {peak:.0f}MB")

        sp = os.path.join(layer_dir, f"layer_{layer_idx}.pt")
        torch.save({"block":c_block.state_dict(),"proj_up":proj_up.state_dict(),
                    "proj_down":proj_down.state_dict()}, sp)

        del c_block, proj_up, proj_down, opt, cache_ds
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    save_metadata(
        directory_metadata_path(layer_dir),
        {
            **runtime_dataset_metadata(),
            "artifact": "spawned_layers",
            "teacher_label": label,
            "teacher_layers": len(teacher_model.blocks),
            "target_config": model_config_to_dict(target_config),
        },
    )
    save_csv(os.path.join(Config.RESULTS_DIR, result_csv), spawn_log)
    save_csv(os.path.join(Config.RESULTS_DIR, step_result_csv), spawn_step_rows)
    print("\nLayer Spawn complete.")


# Child / Next Child

class LayerGrowthModel(nn.Module):
    def __init__(self, parent, internal_config, layer_dir, copy_parent_norm=False):
        super().__init__()
        device = Config.DEVICE
        interface_dim = parent.tok_emb.weight.shape[1]
        internal_dim  = internal_config.d_model
        vocab_size = parent.tok_emb.weight.shape[0]

        ensure_metadata_compatibility(
            directory_metadata_path(layer_dir),
            {
                **runtime_dataset_metadata(),
                "artifact": "spawned_layers",
                "teacher_layers": len(parent.blocks),
                "target_config": model_config_to_dict(internal_config),
            },
            f"Layer directory {layer_dir}",
        )

        self.tok_emb  = nn.Embedding(vocab_size, interface_dim)
        with torch.no_grad():
            self.tok_emb.weight.copy_(parent.tok_emb.weight.detach())
        self.norm_out = RMSNorm(interface_dim)
        if copy_parent_norm and hasattr(parent, "norm_out") and parent.norm_out.weight.shape[0] == interface_dim:
            self.norm_out.load_state_dict(parent.norm_out.state_dict())
        self.lm_head  = nn.Linear(interface_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.blocks     = nn.ModuleList()
        self.proj_ups   = nn.ModuleList()
        self.proj_downs = nn.ModuleList()

        for l in range(len(parent.blocks)):
            ckpt = torch.load(os.path.join(layer_dir, f"layer_{l}.pt"), map_location=device)
            block = TransformerBlock(internal_config)
            block.load_state_dict(ckpt["block"])
            pu = nn.Linear(interface_dim, internal_dim, bias=False)
            pd = nn.Linear(internal_dim, interface_dim, bias=False)
            pu.load_state_dict(ckpt["proj_up"])
            pd.load_state_dict(ckpt["proj_down"])
            self.blocks.append(block)
            self.proj_ups.append(pu)
            self.proj_downs.append(pd)

    def count_params(self): return sum(p.numel() for p in self.parameters())/1e6

    def forward(self, input_ids, labels=None):
        x = self.tok_emb(input_ids)
        for layer_idx in range(len(self.blocks)):
            x = layer_forward(self, layer_idx, x)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            sl = logits[:,:-1,:].contiguous()
            tl = labels[:,1: ].contiguous()
            loss = F.cross_entropy(sl.view(-1,sl.size(-1)), tl.view(-1))
        return loss, logits


class ChildModel(LayerGrowthModel):
    def __init__(self, parent):
        super().__init__(parent, Config.CHILD, Config.CHILD_DIR)


class NextChildModel(LayerGrowthModel):
    def __init__(self, parent):
        super().__init__(parent, Config.NEXT_CHILD, Config.NEXT_CHILD_DIR, copy_parent_norm=True)


def load_child_parent(device, require_finetuned=False):
    if not os.path.exists(Config.PARENT_PATH):
        raise FileNotFoundError(f"Missing parent checkpoint: {Config.PARENT_PATH}")
    if not os.path.exists(Config.CHILD_DIR):
        raise FileNotFoundError(f"Missing child layer directory: {Config.CHILD_DIR}")

    base_parent = load_base_parent(device)
    child_parent = ChildModel(base_parent).to(device)

    if os.path.exists(Config.CHILD_FT_PATH):
        ensure_metadata_compatibility(
            checkpoint_metadata_path(Config.CHILD_FT_PATH),
            {
                **runtime_dataset_metadata(),
                "artifact": "child_finetuned_checkpoint",
            },
            "Child fine-tuned checkpoint",
        )
        child_parent.load_state_dict(torch.load(Config.CHILD_FT_PATH, map_location=device))
        label = "Child parent (fine-tuned)"
    elif require_finetuned:
        raise FileNotFoundError(f"Missing fine-tuned child checkpoint: {Config.CHILD_FT_PATH}")
    else:
        label = "Child parent (spawn only)"

    print(f"Using {label.lower()} as the teacher model.")
    child_parent.eval()
    return child_parent, label


# Finetune

def run_finetune_for_model(
    data_cache,
    model,
    model_label,
    save_path,
    result_csv,
    epoch_result_csv,
    epochs=Config.FINETUNE_EPOCHS,
    lr=Config.LR_FINETUNE,
    checkpoint_artifact="finetuned_checkpoint",
):
    print("\n" + "="*55 + f"\n  Fine-tuning: {model_label}\n" + "="*55)
    device = Config.DEVICE
    model = model.to(device)
    model.train()
    print(f"{model_label}: {model.count_params():.1f}M parameters")
    print(f"GPU memory usage (at fine-tune start): {gpu_res_mb():.0f}MB")

    dataset = TokenDataset(data_cache["train"], Config.SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                         shuffle=True, num_workers=Config.DATALOADER_WORKERS,
                         pin_memory=torch.cuda.is_available())

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             betas=(0.9,0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=len(loader)*epochs,
        eta_min=lr/10)

    log_rows, epoch_rows, step = [], [], 0
    for epoch in range(epochs):
        epoch_loss, t0 = 0., time.time()
        for x in loader:
            x = x.to(device)
            loss, _ = model(x, labels=x)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            opt.step(); sch.step()
            epoch_loss += loss.item(); step += 1
            if step % 100 == 0:
                lr = sch.get_last_lr()[0]
                log_rows.append({"step":step,"epoch":epoch+1,
                                  "train_loss":round(loss.item(),4),
                                  "lr":f"{lr:.2e}","gpu_mb":round(gpu_mb(),1)})
                print(f"  epoch {epoch+1} | step {step:5d} | loss {loss.item():.4f}"
                      f" | lr {lr:.2e} | GPU {gpu_mb():.0f}MB")

        val_ppl = compute_perplexity(model, data_cache["valid"], device)
        avg = epoch_loss/len(loader)
        epoch_rows.append({"epoch":epoch+1,
                           "avg_train_loss":round(avg,4),
                           "val_ppl":round(val_ppl,6)})
        print(f"\n  Epoch {epoch+1} complete | avg loss: {avg:.4f}"
              f" | valid PPL: {val_ppl:.2f} | {time.time()-t0:.0f}s")
        for row in log_rows:
            if row["epoch"]==epoch+1 and "val_ppl" not in row:
                row["val_ppl"] = val_ppl

    torch.save(model.state_dict(), save_path)
    save_metadata(
        checkpoint_metadata_path(save_path),
        {
            **runtime_dataset_metadata(),
            "artifact": checkpoint_artifact,
            "model_label": model_label,
        },
    )
    save_csv(os.path.join(Config.RESULTS_DIR, result_csv), log_rows)
    save_csv(os.path.join(Config.RESULTS_DIR, epoch_result_csv), epoch_rows)
    print(f"\nSaved fine-tuned model: {save_path}")

def run_finetune(data_cache):
    device = Config.DEVICE
    parent = load_base_parent(device)
    child = ChildModel(parent).to(device)
    run_finetune_for_model(
        data_cache=data_cache,
        model=child,
        model_label="Child model",
        save_path=Config.CHILD_FT_PATH,
        result_csv="finetune_log.csv",
        epoch_result_csv="finetune_epoch_metrics.csv",
        checkpoint_artifact="child_finetuned_checkpoint",
    )


def run_scratch_baseline(data_cache):
    device = Config.DEVICE
    baseline = MyModel(Config.BASELINE).to(device)
    run_finetune_for_model(
        data_cache=data_cache,
        model=baseline,
        model_label="Scratch baseline (~32.1M, no Spawn)",
        save_path=Config.BASELINE_PATH,
        result_csv="scratch_baseline_log.csv",
        epoch_result_csv="scratch_baseline_epoch_metrics.csv",
        epochs=Config.BASELINE_EPOCHS,
        lr=Config.LR_FINETUNE,
        checkpoint_artifact="scratch_baseline_checkpoint",
    )


def run_next_cache(data_cache):
    try:
        child_parent, label = load_child_parent(Config.DEVICE)
    except FileNotFoundError as e:
        print(e)
        print("Run the parent -> child flow first so a child model exists.")
        return
    run_cache_for_model(data_cache, child_parent, Config.NEXT_CACHE_DIR, label)


def run_next_spawn(_data_cache=None):
    try:
        child_parent, label = load_child_parent("cpu")
    except FileNotFoundError as e:
        print(e)
        print("Run the parent -> child flow first so a child model exists.")
        return
    run_spawn_from_teacher(
        label=label,
        teacher_model=child_parent,
        cache_dir=Config.NEXT_CACHE_DIR,
        target_config=Config.NEXT_CHILD,
        layer_dir=Config.NEXT_CHILD_DIR,
        result_csv="next_spawn_log.csv",
        step_result_csv="next_spawn_step_log.csv",
    )


def run_next_finetune(data_cache):
    try:
        child_parent, label = load_child_parent(Config.DEVICE)
    except FileNotFoundError as e:
        print(e)
        print("Run the parent -> child flow first so a child model exists.")
        return
    try:
        next_child = NextChildModel(child_parent).to(Config.DEVICE)
    except FileNotFoundError as e:
        print(e)
        print("Run --mode next_spawn first.")
        return
    run_finetune_for_model(
        data_cache=data_cache,
        model=next_child,
        model_label=f"Next child model from {label.lower()}",
        save_path=Config.NEXT_CHILD_FT_PATH,
        result_csv="next_finetune_log.csv",
        epoch_result_csv="next_finetune_epoch_metrics.csv",
        checkpoint_artifact="next_child_finetuned_checkpoint",
    )


# Eval

def run_eval(data_cache):
    print("\n" + "="*55 + "\n  Step 5: Evaluation (3-model comparison)\n" + "="*55)
    device = Config.DEVICE

    parent = load_base_parent(device)
    parent.eval()

    child_spawn = ChildModel(parent).to(device)
    child_spawn.eval()

    child_ft = None
    if os.path.exists(Config.CHILD_FT_PATH):
        ensure_metadata_compatibility(
            checkpoint_metadata_path(Config.CHILD_FT_PATH),
            {
                **runtime_dataset_metadata(),
                "artifact": "child_finetuned_checkpoint",
            },
            "Child fine-tuned checkpoint",
        )
        child_ft = ChildModel(parent).to(device)
        child_ft.load_state_dict(torch.load(Config.CHILD_FT_PATH, map_location=device))
        child_ft.eval()

    scratch_baseline = None
    if os.path.exists(Config.BASELINE_PATH):
        ensure_metadata_compatibility(
            checkpoint_metadata_path(Config.BASELINE_PATH),
            {
                **runtime_dataset_metadata(),
                "artifact": "scratch_baseline_checkpoint",
            },
            "Scratch baseline checkpoint",
        )
        scratch_baseline = MyModel(Config.BASELINE).to(device)
        scratch_baseline.load_state_dict(torch.load(Config.BASELINE_PATH, map_location=device))
        scratch_baseline.eval()

    print(f"Parent model : {parent.count_params():.1f}M")
    print(f"Child (spawn): {child_spawn.count_params():.1f}M")
    if child_ft: print(f"Child (FT)   : {child_ft.count_params():.1f}M")
    if scratch_baseline: print(f"Scratch 32M  : {scratch_baseline.count_params():.1f}M")

    results = {}
    for split in ["valid", "test"]:
        p_ppl  = compute_perplexity(parent,      data_cache[split], device)
        cs_ppl = compute_perplexity(child_spawn, data_cache[split], device)
        row = {"parent_ppl":round(p_ppl,2), "child_spawn_ppl":round(cs_ppl,2)}
        d1 = p_ppl - cs_ppl
        print(f"\n  [{split}]")
        print(f"    Parent model  : {p_ppl:.2f}")
        print(f"    Child (spawn) : {cs_ppl:.2f}  ({'improved' if d1>0 else 'worse'} {abs(d1):.2f})")
        if child_ft:
            ft_ppl = compute_perplexity(child_ft, data_cache[split], device)
            row["child_ft_ppl"] = round(ft_ppl,2)
            d2 = p_ppl - ft_ppl
            d3 = cs_ppl - ft_ppl
            print(f"    Child (FT)    : {ft_ppl:.2f}  "
                  f"({'improved' if d2>0 else 'worse'} vs parent: {abs(d2):.2f})")
            print(f"    spawn->FT delta: {d3:.2f}")
        if scratch_baseline:
            sb_ppl = compute_perplexity(scratch_baseline, data_cache[split], device)
            row["scratch_baseline_ppl"] = round(sb_ppl,2)
            d4 = (ft_ppl - sb_ppl) if child_ft else (cs_ppl - sb_ppl)
            ref_label = "child FT" if child_ft else "child spawn"
            print(f"    Scratch 32M   : {sb_ppl:.2f}  "
                  f"({'better' if d4>0 else 'worse'} than {ref_label}: {abs(d4):.2f})")
        results[split] = row

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    rp = os.path.join(Config.RESULTS_DIR, "eval_results.json")
    with open(rp,"w") as f: json.dump(results, f, indent=2)
    print(f"\n  Saved results: {rp}")
    return results


def run_next_eval(data_cache):
    print("\n" + "="*55 + "\n  Evaluation: child parent -> next child\n" + "="*55)
    device = Config.DEVICE

    try:
        child_parent, label = load_child_parent(device)
    except FileNotFoundError as e:
        print(e)
        print("Run the parent -> child flow first so a child model exists.")
        return

    try:
        next_spawn = NextChildModel(child_parent).to(device)
    except FileNotFoundError as e:
        print(e)
        print("Run --mode next_spawn first.")
        return
    next_spawn.eval()

    next_ft = None
    if os.path.exists(Config.NEXT_CHILD_FT_PATH):
        ensure_metadata_compatibility(
            checkpoint_metadata_path(Config.NEXT_CHILD_FT_PATH),
            {
                **runtime_dataset_metadata(),
                "artifact": "next_child_finetuned_checkpoint",
            },
            "Next child fine-tuned checkpoint",
        )
        next_ft = NextChildModel(child_parent).to(device)
        next_ft.load_state_dict(torch.load(Config.NEXT_CHILD_FT_PATH, map_location=device))
        next_ft.eval()

    print(f"{label:16}: {child_parent.count_params():.1f}M")
    print(f"Next child (spawn): {next_spawn.count_params():.1f}M")
    if next_ft: print(f"Next child (FT)   : {next_ft.count_params():.1f}M")

    results = {}
    for split in ["valid", "test"]:
        teacher_ppl = compute_perplexity(child_parent, data_cache[split], device)
        spawn_ppl   = compute_perplexity(next_spawn,   data_cache[split], device)
        row = {"teacher_ppl":round(teacher_ppl,2), "next_child_spawn_ppl":round(spawn_ppl,2)}
        d1 = teacher_ppl - spawn_ppl
        print(f"\n  [{split}]")
        print(f"    {label:<16}: {teacher_ppl:.2f}")
        print(f"    Next child (spawn): {spawn_ppl:.2f}  ({'improved' if d1>0 else 'worse'} {abs(d1):.2f})")
        if next_ft:
            ft_ppl = compute_perplexity(next_ft, data_cache[split], device)
            row["next_child_ft_ppl"] = round(ft_ppl,2)
            d2 = teacher_ppl - ft_ppl
            d3 = spawn_ppl - ft_ppl
            print(f"    Next child (FT)   : {ft_ppl:.2f}  "
                  f"({'improved' if d2>0 else 'worse'} vs teacher: {abs(d2):.2f})")
            print(f"    spawn->FT delta: {d3:.2f}")
        results[split] = row

    rp = os.path.join(Config.RESULTS_DIR, "next_eval_results.json")
    with open(rp,"w") as f: json.dump(results, f, indent=2)
    print(f"\n  Saved results: {rp}")
    return results


# Plots

def plot_loss_curve(path, title, color, out_name):
    import matplotlib.pyplot as plt

    steps, losses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["train_loss"]))

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(steps, losses, linewidth=1.5, color=color)
    ax.set_xlabel("Step"); ax.set_ylabel("Training Loss"); ax.set_title(title)
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_name, dpi=150); plt.close()


def plot_memory_bars(spawn_csv, title, out_name):
    import matplotlib.pyplot as plt

    layers, peaks = [], []
    with open(spawn_csv) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer_idx"]))
            peaks.append(float(row["gpu_peak_mb"]))

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(layers, peaks, color="#16a34a", alpha=0.8)
    ax.set_xlabel("Layer Index"); ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y"); ax.set_xticks(layers)
    fig.tight_layout()
    fig.savefig(out_name, dpi=150); plt.close()


def load_epoch_series(epoch_csv=None, step_csv=None):
    epoch_map = {}

    if epoch_csv and os.path.exists(epoch_csv):
        with open(epoch_csv) as f:
            for row in csv.DictReader(f):
                epoch_map[int(row["epoch"])] = float(row["val_ppl"])
    elif step_csv and os.path.exists(step_csv):
        with open(step_csv) as f:
            for row in csv.DictReader(f):
                if row.get("val_ppl"):
                    epoch_map[int(row["epoch"])] = float(row["val_ppl"])

    if not epoch_map:
        return [], []

    epochs = sorted(epoch_map.keys())
    values = [epoch_map[e] for e in epochs]
    return epochs, values


def plot_validation_ppl_over_epochs(results_dir, out_name):
    import matplotlib.pyplot as plt

    series = []

    parent_epochs, parent_vals = load_epoch_series(
        epoch_csv=os.path.join(results_dir, "pretrain_epoch_metrics.csv"),
        step_csv=os.path.join(results_dir, "pretrain_log.csv"),
    )
    if parent_epochs:
        series.append(("Parent", parent_epochs, parent_vals, "#2563eb"))

    child_epochs, child_vals = load_epoch_series(
        epoch_csv=os.path.join(results_dir, "finetune_epoch_metrics.csv"),
        step_csv=os.path.join(results_dir, "finetune_log.csv"),
    )
    eval_json = os.path.join(results_dir, "eval_results.json")
    if child_epochs and os.path.exists(eval_json):
        with open(eval_json) as f:
            ev = json.load(f)
        spawn_start = ev.get("valid", {}).get("child_spawn_ppl")
        if spawn_start is not None:
            child_epochs = [0] + child_epochs
            child_vals = [spawn_start] + child_vals
    if child_epochs:
        series.append(("Child (spawn+FT)", child_epochs, child_vals, "#16a34a"))

    scratch_epochs, scratch_vals = load_epoch_series(
        epoch_csv=os.path.join(results_dir, "scratch_baseline_epoch_metrics.csv"),
        step_csv=os.path.join(results_dir, "scratch_baseline_log.csv"),
    )
    if scratch_epochs:
        series.append(("Scratch baseline (~32.1M)", scratch_epochs, scratch_vals, "#ef4444"))

    if not series:
        return False

    fig, ax = plt.subplots(figsize=(8,5))
    for label, epochs, values, color in series:
        ax.plot(epochs, values, marker="o", linewidth=2, color=color, label=label)
    if parent_vals:
        ax.axhline(parent_vals[-1], linestyle="--", linewidth=1.5, color="#1d4ed8",
                   alpha=0.8, label="Parent final PPL")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity (lower is better)")
    ax.set_title("Validation PPL by Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_name, dpi=150); plt.close()
    return True


def plot_spawn_loss_by_layer(step_csv, title, out_name):
    import matplotlib.pyplot as plt

    if not os.path.exists(step_csv):
        return False

    layer_series = {}
    with open(step_csv) as f:
        for row in csv.DictReader(f):
            layer_idx = int(row["layer_idx"])
            layer_series.setdefault(layer_idx, {"x": [], "y": []})
            layer_series[layer_idx]["x"].append(int(row["step_in_layer"]))
            layer_series[layer_idx]["y"].append(float(row["loss"]))

    if not layer_series:
        return False

    fig, ax = plt.subplots(figsize=(9,5))
    for layer_idx in sorted(layer_series.keys()):
        ax.plot(layer_series[layer_idx]["x"], layer_series[layer_idx]["y"],
                linewidth=1.5, label=f"Layer {layer_idx}")

    ax.set_xlabel("Optimization Step Within Layer")
    ax.set_ylabel("Spawn Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_name, dpi=150); plt.close()
    return True


def plot_perplexity_bars(eval_json, title, out_name, series_specs):
    import matplotlib.pyplot as plt

    with open(eval_json) as f:
        ev = json.load(f)

    splits = list(ev.keys())
    active_specs = [spec for spec in series_specs if spec[1] in ev[splits[0]]]
    if not active_specs:
        return False

    series_values = []
    for label, key, color in active_specs:
        values = [ev[s][key] for s in splits]
        series_values.append((label, values, color))

    x = list(range(len(splits)))
    width = 0.8 / len(series_values)
    offsets = [(-0.4 + width / 2) + i * width for i in range(len(series_values))]

    fig, ax = plt.subplots(figsize=(8,5))
    for offset, (label, values, color) in zip(offsets, series_values):
        ax.bar([i + offset for i in x], values, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(out_name, dpi=150); plt.close()
    return True


def run_plot():
    print("\n" + "="*55 + "\n  Plot Generation\n" + "="*55)
    try:
        import matplotlib; matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not found."); return

    rd = Config.RESULTS_DIR

    # Loss curves.
    for fname, title, color in [
        ("pretrain_log.csv","Parent Model Pre-training Loss Curve","#2563eb"),
        ("finetune_log.csv","Child Model Fine-tuning Loss Curve",  "#16a34a"),
        ("scratch_baseline_log.csv","Scratch Baseline Training Loss Curve", "#ef4444"),
    ]:
        path = os.path.join(rd, fname)
        if not os.path.exists(path): continue
        out = os.path.join(rd, fname.replace("_log.csv","_loss.png"))
        plot_loss_curve(path, title, color, out)
        print(f"  Saved: {os.path.basename(out)}")

    # Memory plot.
    spawn_csv = os.path.join(rd, "spawn_log.csv")
    if os.path.exists(spawn_csv):
        plot_memory_bars(spawn_csv, "GPU Memory: Layer Spawn vs Full Training",
                         os.path.join(rd, "spawn_memory.png"))
        print("  Saved: spawn_memory.png")

    # Perplexity plot.
    eval_json = os.path.join(rd, "eval_results.json")
    if os.path.exists(eval_json):
        plot_perplexity_bars(
            eval_json=eval_json,
            title="Perplexity Comparison: Parent vs Child Models",
            out_name=os.path.join(rd, "perplexity_comparison.png"),
            series_specs=[
                ("Parent", "parent_ppl", "#3b82f6"),
                ("Child (spawn only)", "child_spawn_ppl", "#f97316"),
                ("Child (spawn+FT)", "child_ft_ppl", "#16a34a"),
                ("Scratch baseline (~32.1M)", "scratch_baseline_ppl", "#ef4444"),
            ],
        )
        print("  Saved: perplexity_comparison.png")

    if plot_validation_ppl_over_epochs(rd, os.path.join(rd, "validation_ppl_by_epoch.png")):
        print("  Saved: validation_ppl_by_epoch.png")

    if plot_spawn_loss_by_layer(os.path.join(rd, "spawn_step_log.csv"),
                                "Spawn Loss Convergence by Layer",
                                os.path.join(rd, "spawn_loss_by_layer.png")):
        print("  Saved: spawn_loss_by_layer.png")

    print(f"\n  Saved all plots to {rd}/.")


def run_next_plot():
    print("\n" + "="*55 + "\n  Plot Generation: child parent -> next child\n" + "="*55)
    try:
        import matplotlib; matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not found."); return

    rd = Config.RESULTS_DIR

    finetune_csv = os.path.join(rd, "next_finetune_log.csv")
    if os.path.exists(finetune_csv):
        out = os.path.join(rd, "next_finetune_loss.png")
        plot_loss_curve(finetune_csv, "Next Child Fine-tuning Loss Curve", "#16a34a", out)
        print("  Saved: next_finetune_loss.png")

    spawn_csv = os.path.join(rd, "next_spawn_log.csv")
    if os.path.exists(spawn_csv):
        plot_memory_bars(spawn_csv, "GPU Memory: Child Parent -> Next Child",
                         os.path.join(rd, "next_spawn_memory.png"))
        print("  Saved: next_spawn_memory.png")

    eval_json = os.path.join(rd, "next_eval_results.json")
    if os.path.exists(eval_json):
        plot_perplexity_bars(
            eval_json=eval_json,
            title="Perplexity Comparison: Child Parent vs Next Child",
            out_name=os.path.join(rd, "next_perplexity_comparison.png"),
            series_specs=[
                ("Child parent", "teacher_ppl", "#3b82f6"),
                ("Next child (spawn)", "next_child_spawn_ppl", "#f97316"),
                ("Next child (FT)", "next_child_ft_ppl", "#16a34a"),
            ],
        )
        print("  Saved: next_perplexity_comparison.png")

    if plot_spawn_loss_by_layer(os.path.join(rd, "next_spawn_step_log.csv"),
                                "Spawn Loss Convergence by Layer (Next Child)",
                                os.path.join(rd, "next_spawn_loss_by_layer.png")):
        print("  Saved: next_spawn_loss_by_layer.png")

    print(f"\n  Saved next-generation plots to {rd}/.")


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["menu","parent_to_child","child_to_child","quit",
                 "data","pretrain","cache","spawn","finetune","baseline","eval","plot","all",
                 "next_cache","next_spawn","next_finetune","next_eval","next_plot","next_all"],
        default="menu")
    parser.add_argument("--child-multiplier",
        help="Target child size, e.g. 1.85 for 1.85x or 85%% / 85 for a +85%% increase.")
    args = parser.parse_args()

    mode = resolve_mode(args.mode)
    should_exit, mode = handle_workflow_mode(mode)
    if should_exit:
        raise SystemExit(0)

    configure_runtime_child_settings(mode, args.child_multiplier)
    print_banner()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    needs_data = {"data","pretrain","cache","spawn","finetune","baseline","eval","all",
                  "next_cache","next_finetune","next_eval","next_all"}
    if mode in needs_data:
        data_cache = prepare_data()

    if mode in ("pretrain","all"): run_pretrain(data_cache)
    if mode in ("cache","all"):    run_cache(data_cache)
    if mode in ("spawn","all"):    run_spawn()
    if mode in ("finetune","all"): run_finetune(data_cache)
    if mode == "baseline":         run_scratch_baseline(data_cache)
    if mode in ("eval","all"):     run_eval(data_cache)
    if mode in ("plot","all"):     run_plot()
    if mode in ("next_cache","next_all"):    run_next_cache(data_cache)
    if mode in ("next_spawn","next_all"):    run_next_spawn()
    if mode in ("next_finetune","next_all"): run_next_finetune(data_cache)
    if mode in ("next_eval","next_all"):     run_next_eval(data_cache)
    if mode in ("next_plot","next_all"):     run_next_plot()

    print("\nDone. Check the results/ folder.")
