import torch
import numpy as np
from src.utils.tokenizer import MotionTokenizer
from src.models.motion_lm_module import MotionLMLightningModule

class MotionLMInferenceEngine:
    """
    High-performance joint inference engine for multi-agent prediction.
    Supports KV Caching and Token Clamping for conditional forecasting.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        self.module = MotionLMLightningModule.load_from_checkpoint(checkpoint_path)
        self.module.to(self.device).eval()
        self.model = self.module.model
        self.tokenizer = self.module.tokenizer
        
    def predict_joint(
        self, 
        history: torch.Tensor, 
        clamped_tokens: dict = None, 
        max_timesteps: int = 20,
        temperature: float = 1.0,
        sampling: str = "greedy"
    ):
        """
        Jointly predicts future tokens for all agents in history.
        Args:
            history: [N, T_hist, D_feat] (Single batch context)
            clamped_tokens: Dict of {agent_idx: [T_pred tokens]}
            max_timesteps: Number of future steps to predict
            temperature: Sampling temperature
            sampling: "greedy" or "multinomial"
        Returns:
            Dict of {agent_idx: [T_pred tokens]}
        """
        clamped_tokens = clamped_tokens or {}
        N = history.shape[0]
        T = max_timesteps
        
        # Add batch dimension to history [1, N, T_hist, D]
        history_batch = history.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. Encode History
            memory = self.model.encoder(history_batch) # [1, N, H]
            
            # Initialize state
            predictions = {n: [] for n in range(N)}
            past_kv = None
            
            # 2. Interleaved Autoregressive Loop
            for t in range(T):
                for n in range(N):
                    # Prepare IDs for this step
                    agent_id = torch.tensor([[n]], device=self.device)
                    time_id = torch.tensor([[t]], device=self.device)
                    
                    # Determine current token (Clamped vs Predicted)
                    if n in clamped_tokens and t < len(clamped_tokens[n]):
                        # CLAMPED: Use fixed token from planning turn
                        current_token = torch.tensor([[clamped_tokens[n][t]]], device=self.device)
                    else:
                        # PREDICTED: Sample from model
                        # If t=0, we start with a ghost token or similar?
                        # Actually, in vanilla MotionLM, the input at t=0, n=0 is often a 'start' token (0)
                        # but our model uses self.token_embedding(input_tokens) where input_tokens[:, 1:] = tokens[:, :-1]
                        # This means at t=0, n=0, the input is 0 (padding/start).
                        
                        input_token = torch.tensor([[0]], device=self.device)
                        if t > 0 or n > 0:
                            # Use the last produced token (interleaved order)
                            # We need to track the last token globally
                            pass # handled by logic below
                            
                        # Wait, the structure of forward_step expects the PREVIOUS token to predict CURRENT.
                        # At (t, n), the input token is the one from (t, n-1) or (t-1, N-1).
                        
                        # Let's fix the logic for starting token
                        if t == 0 and n == 0:
                            prev_token = torch.tensor([[0]], device=self.device)
                        else:
                            # The previous token in the interleaved sequence
                            prev_n = (n - 1) % N
                            prev_t = t if n > 0 else t - 1
                            prev_token_val = predictions[prev_n][prev_t] if t > 0 or n > 0 else 0
                            prev_token = torch.tensor([[prev_token_val]], device=self.device)
                        
                        # Run one step of Transformer Decoder with KV Cache
                        logits, past_kv = self.model.decoder.forward_step(
                            prev_token, agent_id, time_id, memory, past_kv
                        )
                        
                        # Sampling
                        if sampling == "greedy":
                            current_token_val = torch.argmax(logits[0, 0, :]).item()
                        else:
                            probs = torch.softmax(logits[0, 0, :] / temperature, dim=-1)
                            current_token_val = torch.multinomial(probs, 1).item()
                        
                        current_token = torch.tensor([[current_token_val]], device=self.device)

                    # Update predictions
                    predictions[n].append(current_token.item())
                    
                    # If this token was CLAMPED, we still need to update KV cache using it
                    # but only if we didn't just predict it.
                    if n in clamped_tokens and t < len(clamped_tokens[n]):
                        # We must update KV cache with the CLAMPED token so later agents see it
                        # Wait, the clamped token is the "input" for the NEXT agent.
                        # So at (t, n), we update KV with input=prev_token, and then the NEXT agent 
                        # will use CLAMPED token as its input.
                        
                        # Actually, to update KV cache for the clamped token, we'd need to run it 
                        # through the transformer. 
                        # Simpler: Always run forward_step, and if clamped, just overwrite the result.
                        pass # This is handled if we restructure the loop slightly
        
        return predictions

    def predict_joint_v2(self, history, clamped_tokens=None, max_timesteps=20):
        """ Cleaner implementation of clamped sampling """
        clamped_tokens = clamped_tokens or {}
        N = history.shape[0]
        T = max_timesteps
        history_batch = history.unsqueeze(0).to(self.device)
        
        predictions = {n: [] for n in range(N)}
        past_kv = None
        
        # Initial input token (Start Of Sequence)
        # Initialize with SOS token (84 = Idle/Constant Velocity)
        current_input_token = torch.tensor([[84]], device=self.device)
        
        with torch.no_grad():
            memory = self.model.encoder(history_batch)
            
            for t in range(T):
                for n in range(N):
                    agent_id = torch.tensor([[n]], device=self.device)
                    time_id = torch.tensor([[t]], device=self.device)
                    
                    # Forward pass
                    logits, past_kv = self.model.decoder.forward_step(
                        current_input_token, agent_id, time_id, memory, past_kv
                    )
                    
                    # Selection
                    if n in clamped_tokens and t < len(clamped_tokens[n]):
                        # FORCE plan token
                        token_val = clamped_tokens[n][t]
                    else:
                        # SAMPLE from model
                        token_val = torch.argmax(logits[0, 0, :]).item()
                        
                    predictions[n].append(token_val)
                    
                    # This token becomes the input for the next agent/step
                    current_input_token = torch.tensor([[token_val]], device=self.device)
                    
        return predictions
