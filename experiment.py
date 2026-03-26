import os, csv, time, math, json, argparse, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from my_model import MyModel, ModelConfig, TransformerBlock, RMSNorm
from layer_spawn_custom import LayerExpander


class Config:
    PARENT = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=256,
                         n_layers=6, n_heads=8, n_kv_heads=4,
                         ffn_mult=4, dropout=0.1, bias=False)
    CHILD  = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=512,
                         n_layers=6, n_heads=16, n_kv_heads=8,
                         ffn_mult=4, dropout=0.0, bias=False)
    PRETRAIN_EPOCHS = 3
    SPAWN_EPOCHS    = 2
    FINETUNE_EPOCHS = 3
    BATCH_SIZE      = 16
    SEQ_LEN         = 128
    LR_PRETRAIN     = 3e-4
    LR_SPAWN        = 5e-5
    LR_FINETUNE     = 1e-4
    GRAD_CLIP       = 1.0
    DATA_DIR    = "./data"
    RESULTS_DIR = "./results"
    CKPT_DIR    = "./checkpoints"
    CACHE_DIR   = "./cache"
    PARENT_PATH = "./checkpoints/parent.pt"
    CHILD_DIR   = "./checkpoints/child_layers"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Data

class TokenDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = []
        for i in range(0, len(token_ids) - seq_len - 1, seq_len):
            self.data.append(torch.tensor(token_ids[i:i+seq_len], dtype=torch.long))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class LayerCacheDataset(Dataset):
    def __init__(self, layer_idx):
        self.inputs  = torch.load(os.path.join(Config.CACHE_DIR, f"layer_{layer_idx}_input.pt"))
        self.outputs = torch.load(os.path.join(Config.CACHE_DIR, f"layer_{layer_idx}_output.pt"))
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
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)
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
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


# Data prep

def tokenize_file(path):
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    with open(path, "r", encoding="utf-8") as f: text = f.read()
    ids = tok.encode(text)
    print(f"  {os.path.basename(path)}: {len(ids):,} tokens")
    return ids

def prepare_data():
    print("\n" + "="*55 + "\n  Data Preparation\n" + "="*55)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    urls = {
        "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "test":  "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    }
    cache = {}
    for split, url in urls.items():
        path = os.path.join(Config.DATA_DIR, f"wikitext2_{split}.txt")
        if not os.path.exists(path):
            print(f"  Downloading: {split}...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"  Using existing file: {path}")
        cp = path.replace(".txt", ".pt")
        if os.path.exists(cp):
            print(f"  Loading cache: {split}")
            cache[split] = torch.load(cp)
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
                         shuffle=True, num_workers=2, pin_memory=True)
    print(f"Training samples: {len(dataset):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR_PRETRAIN,
                             betas=(0.9,0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=len(loader)*Config.PRETRAIN_EPOCHS,
        eta_min=Config.LR_PRETRAIN/10)

    log_rows, step = [], 0
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
        print(f"\n  Epoch {epoch+1} complete | avg loss: {avg:.4f}"
              f" | valid PPL: {val_ppl:.2f} | {time.time()-t0:.0f}s"
              f" | GPU reserved: {gpu_res_mb():.0f}MB")
        for row in log_rows:
            if row["epoch"]==epoch+1 and "val_ppl" not in row:
                row["val_ppl"] = val_ppl

    torch.save(model.state_dict(), Config.PARENT_PATH)
    save_csv(os.path.join(Config.RESULTS_DIR, "pretrain_log.csv"), log_rows)
    print(f"\nSaved parent model: {Config.PARENT_PATH}")


# Cache

def run_cache(data_cache):
    print("\n" + "="*55 + "\n  Step 2: Generate Parent Output Cache\n" + "="*55)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    device = Config.DEVICE

    parent = MyModel(Config.PARENT).to(device)
    parent.load_state_dict(torch.load(Config.PARENT_PATH, map_location=device))
    parent.eval()
    print(f"Parent model: {parent.count_params():.1f}M parameters")
    print(f"GPU memory usage (cache generation): {gpu_res_mb():.0f}MB")

    dataset = TokenDataset(data_cache["train"], Config.SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                         shuffle=False, num_workers=2, pin_memory=True)

    n = Config.PARENT.n_layers
    layer_inputs  = [[] for _ in range(n)]
    layer_outputs = [[] for _ in range(n)]

    print(f"\nGenerating cache for {len(dataset):,} samples...")
    t0 = time.time()

    with torch.no_grad():
        for bi, x in enumerate(loader):
            x = x.to(device)
            h = parent.tok_emb(x)
            for l in range(n):
                layer_inputs[l].append(h.cpu())
                h = parent.blocks[l](h)
                layer_outputs[l].append(h.cpu())
            if bi % 100 == 0:
                print(f"  Batch {bi}/{len(loader)}")

    print(f"  Computation complete: {time.time()-t0:.0f}s")

    total_mb = 0
    for l in range(n):
        inp = torch.cat(layer_inputs[l],  dim=0)
        out = torch.cat(layer_outputs[l], dim=0)
        pi  = os.path.join(Config.CACHE_DIR, f"layer_{l}_input.pt")
        po  = os.path.join(Config.CACHE_DIR, f"layer_{l}_output.pt")
        torch.save(inp, pi); torch.save(out, po)
        mb = (os.path.getsize(pi)+os.path.getsize(po))/1024**2
        total_mb += mb
        print(f"  Layer {l}: {mb:.0f}MB")

    print(f"\nTotal cache size: {total_mb:.0f}MB")

    # Free parent GPU memory.
    del parent
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Released parent model from GPU. Remaining: {gpu_res_mb():.0f}MB")


# Spawn

def run_spawn(_data_cache=None):
    print("\n" + "="*55 + "\n  Step 3: Layer Spawn (using cache)\n" + "="*55)

    for l in range(Config.PARENT.n_layers):
        if not os.path.exists(os.path.join(Config.CACHE_DIR, f"layer_{l}_input.pt")):
            print("Cache not found. Run --mode cache first.")
            return

    os.makedirs(Config.CHILD_DIR, exist_ok=True)
    device = Config.DEVICE

    # Parent stays on CPU.
    parent_cpu = MyModel(Config.PARENT)
    parent_cpu.load_state_dict(torch.load(Config.PARENT_PATH, map_location="cpu"))
    parent_cpu.eval()

    expander  = LayerExpander()
    spawn_log = []

    for layer_idx in range(Config.PARENT.n_layers):
        print(f"\n--- Layer {layer_idx} ---")
        reset_peak()

        cache_ds = LayerCacheDataset(layer_idx)
        cache_ld = DataLoader(cache_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True, num_workers=0)

        p_block   = parent_cpu.blocks[layer_idx]
        c_block   = expander.expand_block(p_block, Config.CHILD).to(device)
        p_dim     = Config.PARENT.d_model
        c_dim     = Config.CHILD.d_model
        proj_up   = nn.Linear(p_dim, c_dim, bias=False).to(device)
        proj_down = nn.Linear(c_dim, p_dim, bias=False).to(device)

        mem_start = gpu_mb()
        print(f"  GPU memory usage (child layer only): {mem_start:.0f}MB")

        opt = torch.optim.AdamW(
            list(c_block.parameters())+list(proj_up.parameters())+list(proj_down.parameters()),
            lr=Config.LR_SPAWN, weight_decay=0.01)

        c_block.train(); proj_up.train(); proj_down.train()

        for epoch in range(Config.SPAWN_EPOCHS):
            total_loss = 0.
            for h_in, h_out_parent in cache_ld:
                h_in = h_in.to(device); h_out_parent = h_out_parent.to(device)
                h_large = proj_up(h_in)
                h_child = c_block(h_large)
                h_down  = proj_down(h_child)
                loss_mse = F.mse_loss(h_down, h_out_parent)
                pn = F.layer_norm(h_out_parent, [p_dim])
                cn = F.layer_norm(h_down,       [p_dim])
                loss_kl = F.kl_div(F.log_softmax(cn,dim=-1),
                                   F.softmax(pn,dim=-1), reduction="batchmean")
                loss = loss_mse + 0.1*loss_kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(c_block.parameters())+list(proj_up.parameters())+list(proj_down.parameters()),
                    Config.GRAD_CLIP)
                opt.step(); total_loss += loss.item()
            avg = total_loss/len(cache_ld)
            print(f"  Epoch {epoch+1} | loss: {avg:.6f}")

        peak = gpu_peak_mb()
        spawn_log.append({"layer_idx":layer_idx, "gpu_only_child_mb":round(mem_start,1),
                           "gpu_peak_mb":round(peak,1), "final_loss":round(avg,6)})
        print(f"  GPU peak (child layer only): {peak:.0f}MB")

        sp = os.path.join(Config.CHILD_DIR, f"layer_{layer_idx}.pt")
        torch.save({"block":c_block.state_dict(),"proj_up":proj_up.state_dict(),
                    "proj_down":proj_down.state_dict()}, sp)

        del c_block, proj_up, proj_down, opt, cache_ds
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    save_csv(os.path.join(Config.RESULTS_DIR, "spawn_log.csv"), spawn_log)
    print("\nLayer Spawn complete (parent model is not on the GPU)")


# Child

class ChildModel(nn.Module):
    def __init__(self, parent):
        super().__init__()
        device = Config.DEVICE
        p_dim = Config.PARENT.d_model
        c_dim = Config.CHILD.d_model
        self.tok_emb  = parent.tok_emb
        self.norm_out = RMSNorm(p_dim)
        self.lm_head  = parent.lm_head
        self.blocks     = nn.ModuleList()
        self.proj_ups   = nn.ModuleList()
        self.proj_downs = nn.ModuleList()
        for l in range(Config.PARENT.n_layers):
            ckpt = torch.load(os.path.join(Config.CHILD_DIR, f"layer_{l}.pt"),
                              map_location=device)
            block = TransformerBlock(Config.CHILD)
            block.load_state_dict(ckpt["block"])
            pu = nn.Linear(p_dim, c_dim, bias=False)
            pd = nn.Linear(c_dim, p_dim, bias=False)
            pu.load_state_dict(ckpt["proj_up"])
            pd.load_state_dict(ckpt["proj_down"])
            self.blocks.append(block)
            self.proj_ups.append(pu)
            self.proj_downs.append(pd)

    def count_params(self): return sum(p.numel() for p in self.parameters())/1e6

    def forward(self, input_ids, labels=None):
        x = self.tok_emb(input_ids)
        for block, up, down in zip(self.blocks, self.proj_ups, self.proj_downs):
            x = down(block(up(x)))
        x = self.norm_out(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            sl = logits[:,:-1,:].contiguous()
            tl = labels[:,1: ].contiguous()
            loss = F.cross_entropy(sl.view(-1,sl.size(-1)), tl.view(-1))
        return loss, logits


# Finetune

def run_finetune(data_cache):
    print("\n" + "="*55 + "\n  Step 4: End-to-end Fine-tuning\n" + "="*55)
    device = Config.DEVICE
    parent = MyModel(Config.PARENT).to(device)
    parent.load_state_dict(torch.load(Config.PARENT_PATH, map_location=device))
    child = ChildModel(parent).to(device)
    child.train()
    print(f"Child model: {child.count_params():.1f}M parameters")
    print(f"GPU memory usage (at fine-tune start): {gpu_res_mb():.0f}MB")

    dataset = TokenDataset(data_cache["train"], Config.SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                         shuffle=True, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(child.parameters(), lr=Config.LR_FINETUNE,
                             betas=(0.9,0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=len(loader)*Config.FINETUNE_EPOCHS,
        eta_min=Config.LR_FINETUNE/10)

    log_rows, step = [], 0
    for epoch in range(Config.FINETUNE_EPOCHS):
        epoch_loss, t0 = 0., time.time()
        for x in loader:
            x = x.to(device)
            loss, _ = child(x, labels=x)  # Next-token loss.
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(child.parameters(), Config.GRAD_CLIP)
            opt.step(); sch.step()
            epoch_loss += loss.item(); step += 1
            if step % 100 == 0:
                lr = sch.get_last_lr()[0]
                log_rows.append({"step":step,"epoch":epoch+1,
                                  "train_loss":round(loss.item(),4),
                                  "lr":f"{lr:.2e}","gpu_mb":round(gpu_mb(),1)})
                print(f"  epoch {epoch+1} | step {step:5d} | loss {loss.item():.4f}"
                      f" | lr {lr:.2e} | GPU {gpu_mb():.0f}MB")

        val_ppl = compute_perplexity(child, data_cache["valid"], device)
        avg = epoch_loss/len(loader)
        print(f"\n  Epoch {epoch+1} complete | avg loss: {avg:.4f}"
              f" | valid PPL: {val_ppl:.2f} | {time.time()-t0:.0f}s")
        for row in log_rows:
            if row["epoch"]==epoch+1 and "val_ppl" not in row:
                row["val_ppl"] = val_ppl

    ft_path = os.path.join(Config.CKPT_DIR, "child_finetuned.pt")
    torch.save(child.state_dict(), ft_path)
    save_csv(os.path.join(Config.RESULTS_DIR, "finetune_log.csv"), log_rows)
    print(f"\nSaved fine-tuned child model: {ft_path}")


# Eval

def run_eval(data_cache):
    print("\n" + "="*55 + "\n  Step 5: Evaluation (3-model comparison)\n" + "="*55)
    device = Config.DEVICE

    parent = MyModel(Config.PARENT).to(device)
    parent.load_state_dict(torch.load(Config.PARENT_PATH, map_location=device))
    parent.eval()

    child_spawn = ChildModel(parent).to(device)
    child_spawn.eval()

    ft_path = os.path.join(Config.CKPT_DIR, "child_finetuned.pt")
    child_ft = None
    if os.path.exists(ft_path):
        child_ft = ChildModel(parent).to(device)
        child_ft.load_state_dict(torch.load(ft_path, map_location=device))
        child_ft.eval()

    print(f"Parent model : {parent.count_params():.1f}M")
    print(f"Child (spawn): {child_spawn.count_params():.1f}M")
    if child_ft: print(f"Child (FT)   : {child_ft.count_params():.1f}M")

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
        results[split] = row

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    rp = os.path.join(Config.RESULTS_DIR, "eval_results.json")
    with open(rp,"w") as f: json.dump(results, f, indent=2)
    print(f"\n  Saved results: {rp}")
    return results


# Plots

def run_plot():
    print("\n" + "="*55 + "\n  Plot Generation\n" + "="*55)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found."); return

    rd = Config.RESULTS_DIR

    # Loss curves.
    for fname, title, color in [
        ("pretrain_log.csv","Parent Model Pre-training Loss Curve","#2563eb"),
        ("finetune_log.csv","Child Model Fine-tuning Loss Curve",  "#16a34a"),
    ]:
        path = os.path.join(rd, fname)
        if not os.path.exists(path): continue
        steps, losses = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"])); losses.append(float(row["train_loss"]))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(steps, losses, linewidth=1.5, color=color)
        ax.set_xlabel("Step"); ax.set_ylabel("Training Loss"); ax.set_title(title)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        out = os.path.join(rd, fname.replace("_log.csv","_loss.png"))
        fig.savefig(out, dpi=150); plt.close()
        print(f"  Saved: {os.path.basename(out)}")

    # Memory plot.
    spawn_csv = os.path.join(rd, "spawn_log.csv")
    if os.path.exists(spawn_csv):
        layers, peaks = [], []
        with open(spawn_csv) as f:
            for row in csv.DictReader(f):
                layers.append(int(row["layer_idx"]))
                peaks.append(float(row["gpu_peak_mb"]))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(layers, peaks, color="#16a34a", alpha=0.8, label="Layer Spawn (child only)")
        ax.axhline(y=5950, color="red", linestyle="--", linewidth=1.5,
                   label="Full pre-training (5,950 MB)")
        ax.set_xlabel("Layer Index"); ax.set_ylabel("Peak GPU Memory (MB)")
        ax.set_title("GPU Memory: Layer Spawn vs Full Training")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y"); ax.set_xticks(layers)
        fig.tight_layout()
        fig.savefig(os.path.join(rd,"spawn_memory.png"),dpi=150); plt.close()
        print("  Saved: spawn_memory.png")

    # Perplexity plot.
    eval_json = os.path.join(rd, "eval_results.json")
    if os.path.exists(eval_json):
        with open(eval_json) as f: ev = json.load(f)
        splits = list(ev.keys())
        has_ft = "child_ft_ppl" in ev[splits[0]]
        p_ppls  = [ev[s]["parent_ppl"] for s in splits]
        cs_ppls = [ev[s].get("child_spawn_ppl", ev[s].get("child_ppl",0)) for s in splits]
        x = list(range(len(splits)))
        width = 0.25 if has_ft else 0.35
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar([i-width for i in x], p_ppls,  width, label="Parent",            color="#3b82f6",alpha=0.85)
        ax.bar([i       for i in x], cs_ppls, width, label="Child (spawn only)", color="#f97316",alpha=0.85)
        if has_ft:
            ft_ppls = [ev[s]["child_ft_ppl"] for s in splits]
            ax.bar([i+width for i in x], ft_ppls, width,
                   label="Child (spawn+FT)", color="#16a34a", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in splits])
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("Perplexity Comparison: Parent vs Child Models")
        ax.legend(); ax.grid(True,alpha=0.3,axis="y"); fig.tight_layout()
        fig.savefig(os.path.join(rd,"perplexity_comparison.png"),dpi=150); plt.close()
        print("  Saved: perplexity_comparison.png")

    print(f"\n  Saved all plots to {rd}/.")


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["data","pretrain","cache","spawn","finetune","eval","plot","all"],
        default="all")
    args = parser.parse_args()

    print("\nGrowth Model - Validation Experiment (Revised)")
    print(f"  Device   : {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  Parent   : d_model={Config.PARENT.d_model}, n_layers={Config.PARENT.n_layers}")
    print(f"  Child    : d_model={Config.CHILD.d_model},  n_layers={Config.CHILD.n_layers}")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    needs_data = {"data","pretrain","cache","spawn","finetune","eval","all"}
    if args.mode in needs_data:
        data_cache = prepare_data()

    if args.mode in ("pretrain","all"): run_pretrain(data_cache)
    if args.mode in ("cache","all"):    run_cache(data_cache)
    if args.mode in ("spawn","all"):    run_spawn()
    if args.mode in ("finetune","all"): run_finetune(data_cache)
    if args.mode in ("eval","all"):     run_eval(data_cache)
    if args.mode in ("plot","all"):     run_plot()

    print("\nDone. Check the results/ folder.")
