import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftMoELayer(nn.Module):
    """
    Soft MoE implementation for feature-level fusion.
    Optimized for (Batch, Channel, Dim) token structure.
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
        
        # Dispatch weights: Normalize over input tokens
        dispatch_weights = F.softmax(logits, dim=1) # (B, N, S)
        
        # Combine weights: Normalize over slots
        combine_weights = F.softmax(logits, dim=2) # (B, N, S)
        
        # 1. Dispatch tokens to slots
        slots_input = torch.matmul(dispatch_weights.transpose(1, 2), x) # (B, S, D)
        
        # 2. Parallel processing by experts
        slots_output = []
        for i in range(self.num_experts):
            start = i * self.slots_per_expert
            end = (i + 1) * self.slots_per_expert
            expert_in = slots_input[:, start:end, :]
            expert_out = self.experts[i](expert_in)
            slots_output.append(expert_out)
        
        slots_output = torch.cat(slots_output, dim=1) # (B, S, D)
        
        # 3. Combine slots back to original token shape
        out = torch.matmul(combine_weights, slots_output) # (B, N, D)
        return out

class PackerMoEFusion(nn.Module):
    """
    Channel-Centric Fusion Architecture:
    1. Experts output (B, C, D) -> Time is embedded into D.
    2. TokenPacker uses C queries to aggregate cross-model channel features.
    3. Soft MoE performs differentiable expert fusion per channel token.
    4. Channel-wise prediction head for final output.
    """
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 num_queries=None, d_fusion=256, num_experts=4, device='cpu'):
        super().__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        
        # Use n_features as default num_queries for channel-specific fusion
        self.num_queries = num_queries if num_queries is not None else n_features
        
        # --- Stage 1: TokenPacker (Injection) ---
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, d_fusion) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_fusion)
        
        # Individual projectors to align base models (B, C, D_i) -> (B, C, d_fusion)
        self.projectors = nn.ModuleDict()
        for name, model in self.models_dict.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            
            # Infer hidden dim from dummy pass
            dummy_x = torch.zeros(1, seq_len, n_features).to(device)
            with torch.no_grad():
                h = model.forward_hidden(dummy_x) # Expected (1, C, D_i)
                self.projectors[name] = nn.Linear(h.shape[-1], d_fusion)
        
        # --- Stage 2: Soft MoE (Fusing) ---
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm2 = nn.LayerNorm(d_fusion)
        
        # --- Stage 3: Prediction Head ---
        # Instead of flattening, we use a channel-wise linear head
        self.output_head = nn.Linear(d_fusion, pred_len)
        
        self.to(device)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Extract embeddings (B, C, D_i) and project to (B, C, d_fusion)
        all_tokens = []
        with torch.no_grad():
            for name, model in self.models_dict.items():
                h = model.forward_hidden(x) 
                proj_h = self.projectors[name](h)
                all_tokens.append(proj_h)
        
        # Concatenate tokens from all experts: (B, Num_Models * C, d_fusion)
        kv = torch.cat(all_tokens, dim=1)
        
        # 2. TokenPacker: Aggregate info from all models into channel queries
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        packed_tokens = self.norm1(q + attn_out) # (B, num_queries, d_fusion)
        
        # 3. Soft MoE: Fuse features in the latent space
        fused_tokens = self.soft_moe(packed_tokens)
        fused_tokens = self.norm2(packed_tokens + fused_tokens)
        
        # 4. Channel-wise prediction: (B, num_queries, pred_len)
        out = self.output_head(fused_tokens)
        
        # If queries match features, result is (B, n_features, pred_len)
        # Transpose to (B, pred_len, n_features) for standard output
        if self.num_queries == self.n_features:
            return out.transpose(1, 2)
        else:
            # Fallback for arbitrary query counts
            return out.view(B, -1, self.n_features)[:, :self.pred_len, :]
