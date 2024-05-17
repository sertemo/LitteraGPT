import torch
import torch.nn as nn
from torch.nn import functional as F

from litteragpt.transformer import config as c
from litteragpt.transformer.tokenizer import tokenizer


class Head(nn.Module):
    """One Head of self-attention"""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key: nn.Linear = nn.Linear(c.n_embd, head_size, bias=False)
        self.query: nn.Linear = nn.Linear(c.n_embd, head_size, bias=False)
        self.value: nn.Linear = nn.Linear(c.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(c.block_size, c.block_size)))
        self.dropout: nn.Dropout = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Computes self-attention scores affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out: torch.Tensor = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj: nn.Linear = nn.Linear(c.n_embd, c.n_embd)
        self.dropout: nn.Dropout = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(c.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size)
        self.ffwd: FeedForward = FeedForward(n_embd)
        self.layer_norm1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.layer_norm2: nn.LayerNorm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplicamos Layer Norm ANTES de computar el multihead y el FForward.
        Difiere del paper original pero ahora se hace así"""
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table: nn.Embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size, embedding_dim=c.n_embd
        )
        self.position_embedding_table: nn.Embedding = nn.Embedding(
            c.block_size, c.n_embd
        )
        self.blocks: nn.Sequential = nn.Sequential(
            *[Block(c.n_embd, n_head=c.n_head) for _ in range(c.n_layer)]
        )
        self.layer_norm_final: nn.LayerNorm = nn.LayerNorm(c.n_embd)
        self.lm_head: nn.Linear = nn.Linear(c.n_embd, tokenizer.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=c.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, one_sentence: bool = False
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -c.block_size :]
            # get the predictions
            logits, _ = self(idx_cond)
            # Los logits salen como (B,T,C) si metemos x
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # B,C
            # sample from the distribution
            idx_next: torch.Tensor = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            if one_sentence:
                next_idx_list: list[int] = [int(idx_next.item())]
                next_decoded: str = tokenizer.decode(next_idx_list)
                if next_decoded == ".":
                    break

        return idx
