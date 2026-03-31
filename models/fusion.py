import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMoELayer(nn.Module):
    """
    Soft MoE implementation for feature-level fusion.
    Reference: 'From Sparse to Soft Mixtures of Experts' (2023).
    """
    def __init__(self, d_model, num_experts, slots_per_expert):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.num_slots = num_experts * slots_per_expert
        
        # Learnable parameters for the dispatch/combine routing
        self.phi = nn.Parameter(torch.randn(d_model, self.num_slots) * 0.02)
        
        # Experts: Small MLP specialized in different patterns
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, N, D = x.shape
        
        # Compute routing logits
        logits = torch.matmul(x, self.phi) # (B, N, S)
        
        # Dispatch weights: Normalize over input tokens (each slot gets parts of tokens)
        dispatch_weights = F.softmax(logits, dim=1) # (B, N, S)
        
        # Combine weights: Normalize over slots (each token is formed by parts of slots)
        combine_weights = F.softmax(logits, dim=2) # (B, N, S)
        
        # 1. Dispatch tokens to slots
        # slots_input: (B, S, D)
        slots_input = torch.matmul(dispatch_weights.transpose(1, 2), x)
        
        # 2. Parallel processing by experts
        slots_output = []
        for i in range(self.num_experts):
            start = i * self.slots_per_expert
            end = (i + 1) * self.slots_per_expert
            expert_in = slots_input[:, start:end, :] # (B, slots_per_expert, D)
            expert_out = self.experts[i](expert_in)
            slots_output.append(expert_out)
        
        slots_output = torch.cat(slots_output, dim=1) # (B, S, D)
        
        # 3. Combine slots back to original token shape
        out = torch.matmul(combine_weights, slots_output) # (B, N, D)
        return out

class PackerMoEFusion(nn.Module):
    """
    Hybrid Fusion Architecture:
    1. TokenPacker (Cross-Attention): Efficiently condenses heterogeneous model embeddings.
    2. Soft MoE: Differentiable expert fusion for optimal feature combination.
    """
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 num_queries=16, d_fusion=256, num_experts=4, device='cpu'):
        super().__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        
        # --- Stage 1: TokenPacker (Injection) ---
        # Fixed queries to condense all model outputs
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_fusion) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_fusion)
        
        # Individual projectors to align base models to d_fusion
        self.projectors = nn.ModuleDict()
        for name, model in self.models_dict.items():
            # Freeze base models (as they are pre-trained)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            
            # Infer hidden dim by running a dummy pass
            dummy_x = torch.zeros(1, seq_len, n_features).to(device)
            with torch.no_grad():
                h = model.forward_hidden(dummy_x)
                self.projectors[name] = nn.Linear(h.shape[-1], d_fusion)
        
        # --- Stage 2: Soft MoE (Fusing) ---
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm2 = nn.LayerNorm(d_fusion)
        
        # Final prediction head
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_queries * d_fusion, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, pred_len * n_features)
        )
        self.to(device)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Extract and project embeddings from base models
        all_tokens = []
        with torch.no_grad():
            for name, model in self.models_dict.items():
                h = model.forward_hidden(x) # (B, T_i, D_i)
                proj_h = self.projectors[name](h) # (B, T_i, d_fusion)
                all_tokens.append(proj_h)
        
        # (B, Total_Tokens, d_fusion)
        kv = torch.cat(all_tokens, dim=1)
        
        # 2. TokenPacker: Inject fine-grained info into global queries
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        packed_tokens = self.norm1(q + attn_out)
        
        # 3. Soft MoE: Expert-level fusion in the compressed space
        fused_tokens = self.soft_moe(packed_tokens)
        fused_tokens = self.norm2(packed_tokens + fused_tokens) # Residual connection
        
        # 4. Global Feature to Prediction
        out = self.output_head(fused_tokens)
        return out.view(B, self.pred_len, self.n_features)
