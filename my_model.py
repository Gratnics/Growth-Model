import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# Config

@dataclass
class ModelConfig:
    """Model settings."""
    vocab_size: int   = 50257   # GPT-2 vocab
    max_seq_len: int  = 2048    # Context
    d_model: int      = 512     # Width
    n_layers: int     = 8       # Layers
    n_heads: int      = 8       # Q heads
    n_kv_heads: int   = 4       # KV heads
    ffn_mult: int     = 4       # FFN mult
    dropout: float    = 0.0     # Dropout
    bias: bool        = False   # Use bias

    # Presets.
    # tiny  : d_model=128,  n_layers=4,  n_heads=4  → ~10M
    # small : d_model=512,  n_layers=8,  n_heads=8  → ~85M
    # medium: d_model=1024, n_layers=16, n_heads=16 → ~350M
    # large : d_model=2048, n_layers=24, n_heads=16 → ~1.3B
    # xl    : d_model=4096, n_layers=32, n_heads=32 → ~7B


# RMSNorm

class RMSNorm(nn.Module):
    """RMS norm."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# RoPE

class RotaryEmbedding(nn.Module):
    """Rotary embedding."""
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim

        # Freqs.
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Cache.
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Outer product.
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for sin/cos.
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        # Apply RoPE.
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot


# GQA

class GroupedQueryAttention(nn.Module):
    """Grouped-query attention."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_heads // config.n_kv_heads  # Q per KV
        self.head_dim   = config.d_model // config.n_heads
        self.dropout    = config.dropout

        # Projections.
        self.q_proj = nn.Linear(config.d_model, config.n_heads    * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)

        # Output.
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

        # Positional encoding.
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape  # Batch, time, channels.

        # Project.
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

        # Reshape.
        q = q.view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        # RoPE
        q, k = self.rope(q, k, T)

        # Repeat KV.
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, T, head_dim)
        v = v.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, T, head_dim)

        # SDPA.
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),  # Auto-causal.
        )

        # Merge heads.
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out_proj(attn_out)


# SwiGLU

class SwiGLU(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU width.
        hidden_dim = int(config.d_model * config.ffn_mult * 2 / 3)
        # Round for GPU.
        hidden_dim = (hidden_dim + 63) // 64 * 64

        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.up_proj   = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate.
        gate = F.silu(self.gate_proj(x))  # Swish(gate)
        up   = self.up_proj(x)
        return self.down_proj(gate * up)


# Block

class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1   = RMSNorm(config.d_model)
        self.attn    = GroupedQueryAttention(config)
        self.norm2   = RMSNorm(config.d_model)
        self.ffn     = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attn.
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # FFN.
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# Model

class MyModel(nn.Module):
    """Decoder-only transformer."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings.
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # Output norm.
        self.norm_out = RMSNorm(config.d_model)

        # LM head.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights.
        self.lm_head.weight = self.tok_emb.weight

        # Init.
        self.apply(self._init_weights)

        # GPT-2 scaled init.
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        print(f": {self.count_params():.2f}M ")

    def _init_weights(self, module: nn.Module):
        """Init weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self) -> float:
        """Count params."""
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run forward."""
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f" {T}  {self.config.max_seq_len} "

        # Embed.
        x = self.tok_emb(input_ids)   # (B, T, d_model)

        # Blocks.
        for block in self.blocks:
            x = block(x, mask)

        # Norm.
        x = self.norm_out(x)           # (B, T, d_model)

        # Head.
        logits = self.lm_head(x)       # (B, T, vocab_size)

        # Loss.
        loss = None
        if labels is not None:
            # Shift labels.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1: ].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # Masked labels.
            )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop window.
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                       else input_ids[:, -self.config.max_seq_len:]

            _, logits = self(idx_cond)
            logits = logits[:, -1, :]  # Last step.

            # Temperature.
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k.
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, -1:]] = float('-inf')

            # Sample.
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # EOS.
            if next_token.item() == 50256:  # GPT-2 EOS
                break

        return input_ids


# Presets

def get_tiny_config() -> ModelConfig:
    """~10M"""
    return ModelConfig(d_model=128, n_layers=4, n_heads=4, n_kv_heads=2)

def get_small_config() -> ModelConfig:
    """~85M"""
    return ModelConfig(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4)

def get_medium_config() -> ModelConfig:
    """~350M"""
    return ModelConfig(d_model=1024, n_layers=16, n_heads=16, n_kv_heads=8)

def get_large_config() -> ModelConfig:
    """~1.3B"""
    return ModelConfig(d_model=2048, n_layers=24, n_heads=16, n_kv_heads=8)

def get_xl_config() -> ModelConfig:
    """~7B"""
    return ModelConfig(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8)


# Demo

if __name__ == "__main__":
    import os
    from transformers import GPT2Tokenizer

    print("=" * 50)
    print(" Transformer ")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f": {device}")

    # Presets.
    print("\n")
    configs = {
        "tiny  ": get_tiny_config(),
        "small ": get_small_config(),
        "medium": get_medium_config(),
        "large ": get_large_config(),
        "xl    ": get_xl_config(),
    }
    for name, cfg in configs.items():
        m = MyModel(cfg)
        print(f"  {name}: {m.count_params():.1f}M ")

    # Tiny run.
    print("\ntiny")
    config = get_tiny_config()
    model  = MyModel(config).to(device)

    # Forward.
    B, T = 2, 64
    dummy_input  = torch.randint(0, config.vocab_size, (B, T)).to(device)
    dummy_labels = torch.randint(0, config.vocab_size, (B, T)).to(device)

    loss, logits = model(dummy_input, labels=dummy_labels)
    print(f"    : {dummy_input.shape}")
    print(f"    : {logits.shape}")
    print(f"        : {loss.item():.4f}")
    print(f"  : {math.log(config.vocab_size):.2f}")

    # Generate.
    print("\n")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"  : {prompt}")
    print(f"   : {tokenizer.decode(generated[0], skip_special_tokens=True)}")

    print("\n ")
    print("\n")
    print("  python pretrain.py  # ")
    print("  python layer_spawn.py --mode spawn  # Layer Spawn ")
