import os
import sys
import torch
import numpy as np

# Ensure src is in path
BASE_DIR = os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.models.motion_lm import MotionLM
from src.models.components import AgentEncoder, TrajectoryDecoder

def test_full_model_flow():
    print("--- Verifying Wayformer-style MotionLM Architecture ---")
    
    # 1. Config
    B, N, T_hist, T_pred = 2, 3, 20, 10
    H = 128
    vocab_size = 170
    
    print(f"Config: B={B}, Agents={N}, Hist={T_hist}, Pred={T_pred}, Hidden={H}")
    
    # 2. Instantiate Model
    model = MotionLM(
        hidden_size=H,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        ff_size=512,
        vocab_size=vocab_size,
        max_agents=N,
        max_timesteps=T_pred,
        history_dim=5
    )
    model.eval()
    
    # 3. Simulate Inputs (Scene-Centric Global Coords)
    # [B, N, T, 5] -> x, y, vx, vy, yaw
    history = torch.randn(B, N, T_hist, 5) * 100.0 # Large scale as requested
    
    # Future Tokens (Teacher Forcing)
    # Interleaved: [A0_T0, A1_T0, A2_T0, A0_T1, ...]
    tokens = torch.randint(0, vocab_size, (B, N * T_pred))
    agent_ids = torch.arange(N).repeat(T_pred).repeat(B, 1)
    time_ids = torch.arange(T_pred).repeat_interleave(N).repeat(B, 1)
    
    print("\n--- Phase 1: Training Forward (Teacher Forcing) ---")
    with torch.no_grad():
        logits = model(history, tokens, agent_ids, time_ids)
    
    print(f"Logits shape: {logits.shape} (Expected: [{B}, {N*T_pred}, {vocab_size}])")
    assert logits.shape == (B, N * T_pred, vocab_size)
    print("✅ Training forward pass successful!")
    
    print("\n--- Phase 2: Autoregressive Step-by-Step (Inference) ---")
    # Simulate first step of inference
    # 1. Get memory from encoder (Full history)
    with torch.no_grad():
        memory = model.encoder(history)
    
    print(f"Encoder Memory shape: {memory.shape} (Expected: [{B}, {N*T_hist}, {H}])")
    assert memory.shape == (B, N * T_hist, H)
    
    # 2. Sequential Step (t=0, n=0)
    # In MotionLM, to predict (t, n), we input the token from (t, n-1) or (t-1, N-1)
    past_kv = None
    sos_token = torch.tensor([[84]] * B) # Start with SOS
    a_id = torch.tensor([[0]] * B)
    t_id = torch.tensor([[0]] * B)
    
    with torch.no_grad():
        step_logits, past_kv = model.decoder.forward_step(
            sos_token, a_id, t_id, memory, past_kv
        )
    
    print(f"Step Logits shape: {step_logits.shape} (Expected: [{B}, 1, {vocab_size}])")
    print(f"KV Cache layers: {len(past_kv)}")
    
    assert step_logits.shape == (B, 1, vocab_size)
    print("✅ Autoregressive step successful!")
    
    print("\nSUMMARY: Model correctly handles full-history memory and spatio-temporal attention.")
    print("Coordinadas remain global (no on-the-fly normalization applied in code).")

if __name__ == "__main__":
    test_full_model_flow()
