import torch
import torch.nn as nn
from src.models.components import AgentEncoder, TrajectoryDecoder

class MotionLM(nn.Module):
    """
    Simplified MotionLM Model for multi-agent interaction modeling.
    Uses AgentEncoder for history and TrajectoryDecoder for future tokens.
    """
    def __init__(
        self,
        hidden_size=256,
        num_encoder_layers=2,
        num_decoder_layers=4,
        num_heads=8,
        ff_size=1024,
        vocab_size=170,
        max_agents=3,
        max_timesteps=40,
        history_dim=5,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = AgentEncoder(
            input_dim=history_dim,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers
        )
        self.decoder = TrajectoryDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            max_agents=max_agents,
            max_timesteps=max_timesteps,
            dropout=dropout
        )

    def forward(self, history, motion_tokens, agent_ids, time_ids):
        """
        Full forward pass for training/inference.
        Args:
            history: [Batch, N, T_hist, D_feat]
            motion_tokens: [Batch, N*T_pred]
            agent_ids: [Batch, N*T_pred]
            time_ids: [Batch, N*T_pred]
        """
        # 1. Encode agent histories
        agent_embeddings = self.encoder(history) # [B, N, H]
        
        # 2. Decode future tokens autoregressively
        logits = self.decoder(motion_tokens, agent_embeddings, agent_ids, time_ids) # [B, N*T, Vocab]
        
        return logits

def test_motion_lm():
    print("Testing Simplified MotionLM...")
    batch_size = 4
    num_agents = 3
    num_timesteps = 10
    history_len = 5
    seq_len = num_agents * num_timesteps
    
    model = MotionLM(
        vocab_size=170, 
        max_agents=num_agents, 
        max_timesteps=num_timesteps,
        history_dim=5
    )
    
    # Fake history: [B, N, T_hist, 5]
    history = torch.randn(batch_size, num_agents, history_len, 5)
    
    # Fake tokens: [B, N*T]
    motion_tokens = torch.randint(0, 170, (batch_size, seq_len))
    
    # Interleaved IDs: [A1_T1, A2_T1, A3_T1, A1_T2, ...]
    agent_ids = torch.arange(num_agents).repeat(num_timesteps).repeat(batch_size, 1)
    time_ids = torch.arange(num_timesteps).repeat_interleave(num_agents).repeat(batch_size, 1)
    
    logits = model(history, motion_tokens, agent_ids, time_ids)
    print(f"MotionLM logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 170), f"Wrong output shape: {logits.shape}"
    print("✅ MotionLM test passed!")

if __name__ == "__main__":
    test_motion_lm()
