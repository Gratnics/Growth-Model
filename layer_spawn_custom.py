import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import math
import argparse
from tqdm import tqdm

# Model parts.
from my_model import (
    MyModel, ModelConfig,
    RMSNorm, RotaryEmbedding,
    GroupedQueryAttention, SwiGLU, TransformerBlock
)


# Config

class Config:
    # Parent.
    PARENT = ModelConfig(
        vocab_size  = 50257,
        max_seq_len = 256,
        d_model     = 256,    # Parent width
        n_layers    = 6,
        n_heads     = 8,
        n_kv_heads  = 4,
        ffn_mult    = 4,
        dropout     = 0.1,
        bias        = False,
    )

    # Child.
    # Double the width.
    CHILD = ModelConfig(
        vocab_size  = 50257,
        max_seq_len = 256,
        d_model     = 512,    # Child width
        n_layers    = 6,      # Same depth
        n_heads     = 16,
        n_kv_heads  = 8,
        ffn_mult    = 4,
        dropout     = 0.0,
        bias        = False,
    )

    # Paths.
    SAVE_DIR        = "./checkpoints"
    PARENT_PATH     = "./checkpoints/parent_model.pt"
    CHILD_PATH      = "./checkpoints/child_model.pt"
    LAYER_PARTS_DIR = "./checkpoints/layer_parts"

    # Train.
    PRETRAIN_EPOCHS = 2
    SPAWN_EPOCHS    = 3
    BATCH_SIZE      = 4
    MAX_SEQ_LEN     = 64
    LR_PRETRAIN     = 3e-4
    LR_SPAWN        = 5e-5   # Spawn LR

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Data

class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.data = []
        for i in range(0, len(token_ids) - seq_len, seq_len // 2):
            chunk = token_ids[i: i + seq_len]
            if len(chunk) == seq_len:
                self.data.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # LM target


def get_tokens() -> list[int]:
    """Load GPT-2 tokens or fallback."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception:
        # Fallback data.
        import random
        return [random.randint(0, 50256) for _ in range(50000)]

    texts = [
        "The transformer architecture has revolutionized natural language processing tasks.",
        "Machine learning models learn patterns from data through gradient descent optimization.",
        "Neural networks consist of layers of interconnected nodes that transform input data.",
        "Large language models are trained on vast amounts of text data from the internet.",
        "The attention mechanism computes weighted sums over value vectors using query and key similarity.",
        "Self-supervised learning enables models to learn from unlabeled data by predicting missing parts.",
        "Gradient checkpointing trades computation for memory by recomputing activations during backprop.",
        "Mixed precision training uses float16 for forward pass and float32 for weight updates.",
        "Rotary position embeddings encode relative positions by rotating query and key vectors.",
        "The feed-forward network in transformers applies two linear transformations with a nonlinearity.",
    ] * 200

    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
    return all_tokens


# Pretrain

def pretrain():
    print("\n" + "=" * 55)
    print("  ①: ")
    print("=" * 55)

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    device = Config.DEVICE
    print(f": {device}")

    # Model.
    model = MyModel(Config.PARENT).to(device)
    print(f": {model.count_params():.1f}M ")

    # Data.
    tokens  = get_tokens()
    dataset = TextDataset(tokens, Config.MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    print(f": {len(dataset)}")

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR_PRETRAIN,
        betas=(0.9, 0.95), weight_decay=0.1
    )
    total_steps = len(loader) * Config.PRETRAIN_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=Config.LR_PRETRAIN / 10
    )

    # Train.
    model.train()
    for epoch in range(Config.PRETRAIN_EPOCHS):
        total_loss = 0.0
        for step, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            x, y = x.to(device), y.to(device)

            loss, _ = model(x, labels=y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {step:4d} | loss {loss.item():.4f} | lr {lr_now:.2e}")

        print(f"Epoch {epoch+1}  | loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), Config.PARENT_PATH)
    print(f"\n : {Config.PARENT_PATH}")


# Spawn

class LayerExpander(nn.Module):
    """Expand parent weights."""

    @staticmethod
    def expand_linear(
        small_w: torch.Tensor,
        out_dim: int,
        in_dim: int
    ) -> torch.Tensor:
        """Tile and scale a weight matrix."""
        s_out, s_in = small_w.shape
        ratio_out = (out_dim + s_out - 1) // s_out
        ratio_in  = (in_dim  + s_in  - 1) // s_in
        expanded = small_w.repeat(ratio_out, ratio_in)
        # Trim.
        expanded = expanded[:out_dim, :in_dim]
        # Scale.
        scale = math.sqrt(s_out * s_in) / math.sqrt(out_dim * in_dim)
        return expanded * scale

    @staticmethod
    def expand_bias(
        small_b: torch.Tensor,
        out_dim: int
    ) -> torch.Tensor:
        ratio = (out_dim + small_b.shape[0] - 1) // small_b.shape[0]
        return small_b.repeat(ratio)[:out_dim]

    def expand_block(
        self,
        parent_block: TransformerBlock,
        child_config: ModelConfig,
    ) -> TransformerBlock:
        """Expand one block."""
        child_block = TransformerBlock(child_config)

        p_dim = parent_block.norm1.weight.shape[0]  # Parent width
        c_dim = child_config.d_model                # Child width

        with torch.no_grad():
            # Norms.
            child_block.norm1.weight.copy_(
                self.expand_bias(parent_block.norm1.weight, c_dim))
            child_block.norm2.weight.copy_(
                self.expand_bias(parent_block.norm2.weight, c_dim))

            # Attn.
            pa = parent_block.attn
            ca = child_block.attn

            # Projections.
            for p_linear, c_linear in [
                (pa.q_proj, ca.q_proj),
                (pa.k_proj, ca.k_proj),
                (pa.v_proj, ca.v_proj),
                (pa.out_proj, ca.out_proj),
            ]:
                c_o, c_i = c_linear.weight.shape
                c_linear.weight.copy_(
                    self.expand_linear(p_linear.weight, c_o, c_i))
                if c_linear.bias is not None and p_linear.bias is not None:
                    c_linear.bias.copy_(
                        self.expand_bias(p_linear.bias, c_o))

            # FFN.
            pf = parent_block.ffn
            cf = child_block.ffn

            for p_linear, c_linear in [
                (pf.gate_proj, cf.gate_proj),
                (pf.up_proj,   cf.up_proj),
                (pf.down_proj, cf.down_proj),
            ]:
                c_o, c_i = c_linear.weight.shape
                c_linear.weight.copy_(
                    self.expand_linear(p_linear.weight, c_o, c_i))

        return child_block


def spawn_single_layer(
    layer_idx: int,
    parent_model: MyModel,
    tokens: list[int],
    device: str,
):
    """Spawn one layer."""
    print(f"\n---  {layer_idx}  ---")

    p_block = parent_model.blocks[layer_idx]
    p_block.eval()

    expander  = LayerExpander()
    c_block   = expander.expand_block(p_block, Config.CHILD).to(device)

    p_dim = Config.PARENT.d_model
    c_dim = Config.CHILD.d_model

    # Projections.
    proj_up   = nn.Linear(p_dim, c_dim, bias=False).to(device)
    proj_down = nn.Linear(c_dim, p_dim, bias=False).to(device)

    # Identity init.
    with torch.no_grad():
        nn.init.eye_(proj_up.weight[  :p_dim, :p_dim])
        nn.init.eye_(proj_down.weight[:p_dim, :p_dim])

    optimizer = torch.optim.AdamW(
        list(c_block.parameters()) +
        list(proj_up.parameters()) +
        list(proj_down.parameters()),
        lr=Config.LR_SPAWN, weight_decay=0.01
    )

    dataset = TextDataset(tokens, Config.MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    parent_model.eval()
    c_block.train()

    for epoch in range(Config.SPAWN_EPOCHS):
        total_loss = 0.0

        for step, (x, _) in enumerate(tqdm(loader, desc=f"  {layer_idx} Epoch{epoch+1}")):
            x = x.to(device)

            with torch.no_grad():
                # Parent activations.
                hidden = parent_model.tok_emb(x)

                # Up to target layer.
                for i in range(layer_idx):
                    hidden = parent_model.blocks[i](hidden)

                hidden_in = hidden.clone()  # Input copy.

                # Parent target.
                hidden_out_parent = p_block(hidden_in)

            # Child path.
            hidden_large = proj_up(hidden_in)           # p_dim -> c_dim
            hidden_out_child = c_block(hidden_large)
            hidden_down  = proj_down(hidden_out_child)  # c_dim -> p_dim

            # Distill.
            loss_distill = F.mse_loss(hidden_down, hidden_out_parent)

            # KL.
            p_norm = F.layer_norm(hidden_out_parent, [p_dim])
            c_norm = F.layer_norm(hidden_down,       [p_dim])
            loss_kl = F.kl_div(
                F.log_softmax(c_norm, dim=-1),
                F.softmax(p_norm, dim=-1),
                reduction="batchmean"
            )

            loss = loss_distill + 0.1 * loss_kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(c_block.parameters()) +
                list(proj_up.parameters()) +
                list(proj_down.parameters()),
                1.0
            )
            optimizer.step()
            total_loss += loss.item()

        print(f"  {layer_idx} Epoch{epoch+1} | loss: {total_loss/len(loader):.6f}")

    # Save.
    os.makedirs(Config.LAYER_PARTS_DIR, exist_ok=True)
    save_path = os.path.join(Config.LAYER_PARTS_DIR, f"layer_{layer_idx}.pt")
    torch.save({
        "block":     c_block.state_dict(),
        "proj_up":   proj_up.state_dict(),
        "proj_down": proj_down.state_dict(),
        "layer_idx": layer_idx,
        "p_dim":     p_dim,
        "c_dim":     c_dim,
    }, save_path)
    print(f"   {layer_idx} : {save_path}")


def spawn_all_layers():
    print("\n" + "=" * 55)
    print("  ②: Layer Spawn")
    print("=" * 55)

    device = Config.DEVICE

    if not os.path.exists(Config.PARENT_PATH):
        print("  --mode pretrain ")
        return

    parent = MyModel(Config.PARENT).to(device)
    parent.load_state_dict(torch.load(Config.PARENT_PATH, map_location=device))
    parent.eval()
    print(f" : {parent.count_params():.1f}M ")

    tokens = get_tokens()

    for layer_idx in range(Config.PARENT.n_layers):
        spawn_single_layer(layer_idx, parent, tokens, device)

    print("\n ")


# Child

class ChildModel(nn.Module):
    """Layer-spawned child model."""
    def __init__(self, parent: MyModel):
        super().__init__()

        p_dim = Config.PARENT.d_model
        c_dim = Config.CHILD.d_model
        device = Config.DEVICE

        # Shared parts.
        self.tok_emb  = parent.tok_emb
        self.norm_out = RMSNorm(p_dim)  # Parent width
        self.lm_head  = parent.lm_head

        # Spawned layers.
        self.blocks     = nn.ModuleList()
        self.proj_ups   = nn.ModuleList()
        self.proj_downs = nn.ModuleList()

        for layer_idx in range(Config.PARENT.n_layers):
            path = os.path.join(Config.LAYER_PARTS_DIR, f"layer_{layer_idx}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{layer_idx}: {path}")

            ckpt = torch.load(path, map_location=device)

            block = TransformerBlock(Config.CHILD)
            block.load_state_dict(ckpt["block"])

            proj_up   = nn.Linear(p_dim, c_dim, bias=False)
            proj_down = nn.Linear(c_dim, p_dim, bias=False)
            proj_up.load_state_dict(ckpt["proj_up"])
            proj_down.load_state_dict(ckpt["proj_down"])

            self.blocks.append(block)
            self.proj_ups.append(proj_up)
            self.proj_downs.append(proj_down)

    def count_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple:
        x = self.tok_emb(input_ids)

        for block, up, down in zip(self.blocks, self.proj_ups, self.proj_downs):
            x_large = up(x)         # p_dim -> c_dim
            x_large = block(x_large)
            x = down(x_large)       # c_dim -> p_dim

        x = self.norm_out(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:  ].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return loss, logits


# Test

def test():
    print("\n" + "=" * 55)
    print("  ③: ")
    print("=" * 55)

    device = Config.DEVICE

    # Parent.
    parent = MyModel(Config.PARENT).to(device)
    parent.load_state_dict(torch.load(Config.PARENT_PATH, map_location=device))
    parent.eval()

    # Child.
    try:
        child = ChildModel(parent).to(device)
        child.eval()
    except FileNotFoundError as e:
        print(f" {e}")
        print(" --mode spawn ")
        return

    print(f": {parent.count_params():.1f}M ")
    print(f": {child.count_params():.1f}M ")

    # Sample.
    tokens = get_tokens()
    x = torch.tensor(tokens[:32]).unsqueeze(0).to(device)
    y = torch.tensor(tokens[:32]).unsqueeze(0).to(device)

    with torch.no_grad():
        p_loss, _ = parent(x, labels=y)
        c_loss, _ = child(x, labels=y)

    print(f"\n")
    print(f"  : {p_loss.item():.4f}")
    print(f"  : {c_loss.item():.4f}")

    # Params.
    parent_params = parent.count_params()
    child_params  = child.count_params()
    print(f"\n")
    print(f"  : {parent_params:.1f}M")
    print(f"  : {child_params:.1f}M  ({child_params/parent_params:.1f})")

    print("\n ")
    print("\n...")
    torch.save(child.state_dict(), Config.CHILD_PATH)
    print(f"  → {Config.CHILD_PATH}")


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pretrain", "spawn", "test", "all"],
        default="all",
    )
    args = parser.parse_args()

    print(f"\n Layer Spawn")
    print(f"   : {Config.DEVICE}")
    print(f"   : {Config.PARENT.d_model} × {Config.PARENT.n_layers}")
    print(f"   : {Config.CHILD.d_model} × {Config.CHILD.n_layers}")

    if args.mode in ("pretrain", "all"):
        pretrain()

    if args.mode in ("spawn", "all"):
        spawn_all_layers()

    if args.mode in ("test", "all"):
        test()

    print("\n ")
