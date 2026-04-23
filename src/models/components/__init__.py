import torch
import torch.nn as nn

class AgentEncoder(nn.Module):
    """
    Simplified MLP/Transformer encoder for agent history.
    Encodes history of (x, y, vx, vy, heading) relative to t=0.
    """
    def __init__(self, input_dim=5, hidden_size=256, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Simple GRU for history sequence
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, history):
        """
        Args:
            history: [Batch, N, T_hist, D_feat]
        Returns:
            embeddings: [Batch, N, Hidden_Size]
        """
        B, N, T, D = history.shape
        # Flatten B, N to process all agents independently in history
        x = history.reshape(B * N, T, D)
        
        _, h_n = self.gru(x) # h_n is [num_layers, B*N, hidden_size]
        
        # Take the last layer hidden state
        out = h_n[-1] # [B*N, hidden_size]
        out = self.norm(out)
        
        # Reshape back
        return out.reshape(B, N, self.hidden_size)

class TrajectoryDecoder(nn.Module):
    """
    Causal transformer decoder for multi-agent motion tokens.
    """
    def __init__(
        self,
        vocab_size=170,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        ff_size=1024,
        max_agents=3,
        max_timesteps=40,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_agents = max_agents
        self.max_timesteps = max_timesteps

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.time_embedding = nn.Embedding(max_timesteps, hidden_size)
        self.agent_embedding = nn.Embedding(max_agents, hidden_size)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            activation="gelu",
            batch_first=True,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output head
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def create_causal_mask(self, length, device):
        """
        Creates a standard triangular attention mask.
        In an interleaved sequence, this naturally allows joint prediction:
        Agent n sees its predecessors at the same timestep and all agents at previous timesteps.
        """
        # (L, L): True means BLOCK (additive mask)
        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
        return mask

    def forward(self, motion_tokens, agent_embeddings, agent_ids, time_ids):
        """
        Standard forward pass (used during training).
        Args:
            motion_tokens: [Batch, L]
            agent_embeddings: [Batch, N, Hidden_Size] (memory)
            agent_ids: [Batch, L]
            time_ids: [Batch, L]
        """
        B, L = motion_tokens.shape
        
        # 1. Token Embeddings + Positional/Identity
        x = self.token_embedding(motion_tokens) # [B, L, H]
        x = x + self.agent_embedding(agent_ids) + self.time_embedding(time_ids)
        
        # 2. Causal Mask
        mask = self.create_causal_mask(L, x.device)
        
        # 3. Decode
        # memory: agent_embeddings [B, N, H] 
        out = self.transformer_decoder(x, agent_embeddings, tgt_mask=mask)
        
        return self.output_head(out)

    def forward_step(self, token, agent_id, time_id, memory, past_kv=None):
        """
        Incremental forward step for one token with KV caching.
        Implements Post-Normalization to match nn.TransformerDecoderLayer(norm_first=False).
        Args:
            token: [Batch, 1] integer token
            agent_id: [Batch, 1] integer agent ID
            time_id: [Batch, 1] integer time ID
            memory: [Batch, N, Hidden_Size] (history embeddings)
            past_kv: List of (past_k, past_v) for each layer
        Returns:
            logits: [Batch, 1, Vocab]
            current_kv: Updated list of (past_k, past_v)
        """
        B = token.shape[0]
        
        # 1. Embeddings (Raw)
        x = self.token_embedding(token) # [B, 1, H]
        x = x + self.agent_embedding(agent_id) + self.time_embedding(time_id)
        
        new_kv = []
        
        # 2. Iterate through layers
        for i, layer in enumerate(self.transformer_decoder.layers):
            # --- Self Attention (Post-Norm Logic) ---
            # In Post-Norm, Attention receives unnormalized 'x'
            raw_q = x 
            
            if past_kv is not None:
                pk, pv = past_kv[i]
                k = torch.cat([pk, raw_q], dim=1) # Keys/Values are also unnormalized
                v = torch.cat([pv, raw_q], dim=1)
            else:
                k, v = raw_q, raw_q
            
            new_kv.append((k.detach(), v.detach()))
            
            # self_attn(query, key, value)
            attn_output, _ = layer.self_attn(raw_q, k, v)
            
            # Post-Norm Step 1: Residual Add -> Dropout -> Norm
            x = x + layer.dropout1(attn_output)
            x = layer.norm1(x)
            
            # --- Cross Attention (Post-Norm Logic) ---
            # q_cross: normalized x, k/v: memory (history embeddings)
            # Standard PyTorch cross-attn in Post-Norm happens AFTER norm1
            attn_output2, _ = layer.multihead_attn(x, memory, memory)
            
            # Post-Norm Step 2: Residual Add -> Dropout -> Norm
            x = x + layer.dropout2(attn_output2)
            x = layer.norm2(x)
            
            # --- Feed Forward (Post-Norm Logic) ---
            # _ff_block in PyTorch includes linear1 -> act -> dropout -> linear2 -> dropout
            ff_output = layer._ff_block(x)
            
            # Post-Norm Step 3: Residual Add -> (Dropout in _ff_block) -> Norm
            x = x + ff_output
            x = layer.norm3(x)
            
        logits = self.output_head(x)
        return logits, new_kv
