import torch

from torch import nn
from torch.nn import functional as F

from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.positional_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim)
        
        residue = x

        x = self.layernorm_1(x)

        # Self attention
        x = self.attention(x, causal_mask=True)

        x += residue

        # Feedforward layer
        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x  = x * torch.sigmoid(1.702 * x) # Tzv. QuickGELU activation function 

        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        #super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) # vocabulary_size=49408, embedding_size=768, max_seq_len=77

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor: # LongTensor because of tokens=positions in vocabulary
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, dim=768)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (batch_size, seq_len, dim)
        output = self.layernorm(state)

        return output
