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

class FlattenMapper(nn.Module):
    """
    Aligns expert outputs (B, C, D) or (B, C, P, D) to a unified (B, Token, d_fusion).
    Flattens the last two dimensions (P and D) for 4D inputs.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x shape: (B, C, D) or (B, C, P, D)
        if x.dim() == 4:
            B, C, P, D = x.shape
            x = x.reshape(B, C, -1) # (B, C, P*D)
        
        return self.norm(self.proj(x))

class QueryGenerator(nn.Module):
    """
    Dynamically generates queries based on input data statistics (Mean, Std, Max, Min).
    """
    def __init__(self, n_features, num_queries, d_fusion):
        super().__init__()
        self.num_queries = num_queries
        self.d_fusion = d_fusion
        
        # 4 stats per feature
        input_dim = n_features * 4
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, d_fusion),
            nn.GELU(),
            nn.Linear(d_fusion, num_queries * d_fusion),
            nn.LayerNorm(num_queries * d_fusion)
        )
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # 1. Global statistics extraction
        mean = x.mean(dim=1) # (B, C)
        std = x.std(dim=1)   # (B, C)
        max_val, _ = x.max(dim=1) # (B, C)
        min_val, _ = x.min(dim=1) # (B, C)
        
        stats = torch.cat([mean, std, max_val, min_val], dim=-1) # (B, 4C)
        
        # 2. Dynamic generation
        queries = self.generator(stats) # (B, num_queries * d_fusion)
        return queries.view(B, self.num_queries, self.d_fusion)

class FusionModel(nn.Module):
    """
    Channel-Centric Fusion Architecture:
    1. Experts output (B, C, D) -> Time is embedded into D.
    2. TokenPacker uses C queries (Static or Dynamic) to aggregate features.
    3. Soft MoE performs differentiable expert fusion per channel token.
    4. Prediction head outputs final values in target space.
    """
    def __init__(self, models_dict, seq_len, pred_len, n_features, 
                 num_queries=None, d_fusion=256, num_experts=4, device='cpu', 
                 query_init_type='orthogonal', use_dynamic_queries=True):
        super().__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.device = device
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_dynamic_queries = use_dynamic_queries

        # Use n_features as default num_queries for channel-specific fusion
        self.num_queries = num_queries if num_queries is not None else n_features

        # --- Stage 1: TokenPacker (Injection) ---
        if self.use_dynamic_queries:
            self.query_gen = QueryGenerator(n_features, self.num_queries, d_fusion)
            self.queries = None
        else:
            self.queries = nn.Parameter(torch.empty(1, self.num_queries, d_fusion))
            self._init_queries(query_init_type)

        self.cross_attn = nn.MultiheadAttention(d_fusion, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_fusion)
        
        # Individual projectors to align base models (B, C, D_i) -> (B, C, d_fusion)
        self.projectors = nn.ModuleDict()
        for name, model in self.models_dict.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            
            # Infer hidden dim from dummy pass
            dummy_batch = {
                'x': torch.zeros(1, seq_len, n_features).to(device),
                'observe_power': torch.zeros(1, seq_len, n_features).to(device)
            }
            with torch.no_grad():
                h = model.forward_hidden(dummy_batch)
                # Calculate flattened input dimension:
                # If 4D (B, C, P, D), in_dim = P * D
                # If 3D (B, C, D), in_dim = D
                if h.dim() == 4:
                    in_dim = h.shape[2] * h.shape[3]
                else:
                    in_dim = h.shape[-1]
                self.projectors[name] = FlattenMapper(in_dim, d_fusion)
        
        # --- Stage 2: Soft MoE (Fusing) ---
        self.soft_moe = SoftMoELayer(d_fusion, num_experts=num_experts, slots_per_expert=4)
        self.norm2 = nn.LayerNorm(d_fusion)
        
        # --- Stage 3: Prediction Head ---
        self.output_head = nn.Linear(d_fusion, pred_len)
        self.dropout = nn.Dropout(0.1) # Prevent overfitting in latent space

        # Aggregation layer: Map num_queries back to actual n_features
        self.aggregate = nn.Conv1d(self.num_queries, self.n_features, 1)
        
        self.to(device)

    def _init_queries(self, init_type):
        """
        Experimental initialization strategies for static queries.
        """
        import math
        if self.queries is None: return
        q = self.queries.data
        d_fusion = q.size(-1)

        if init_type == 'normal':
            nn.init.normal_(q, std=0.02)
        elif init_type == 'orthogonal':
            flat_q = torch.empty(self.num_queries, d_fusion)
            nn.init.orthogonal_(flat_q)
            q.copy_(flat_q.unsqueeze(0))
        elif init_type == 'fourier':
            for i in range(self.num_queries):
                for j in range(d_fusion // 2):
                    val = i / math.pow(10000, 2 * j / d_fusion)
                    q[0, i, 2 * j] = math.sin(val)
                    q[0, i, 2 * j + 1] = math.cos(val)
            q.mul_(0.02)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(q, mode='fan_in', nonlinearity='leaky_relu')
        elif init_type == 'xavier':
            flat_q = torch.empty(self.num_queries, d_fusion)
            nn.init.xavier_normal_(flat_q)
            q.copy_(flat_q.unsqueeze(0))
        elif init_type == 'uniform':
            limit = math.sqrt(3.0 / d_fusion)
            nn.init.uniform_(q, -limit, limit)
        elif init_type == 'constant':
            nn.init.constant_(q, 1.0)
            q.add_(torch.randn_like(q) * 0.001)
        else:
            raise ValueError(f"Unknown query_init_type: {init_type}")

    def forward(self, batch):
        # 1. Extract embeddings using the RAW batch
        all_tokens = []
        with torch.no_grad():
            for name, model in self.models_dict.items():
                h = model.forward_hidden(batch) 
                proj_h = self.projectors[name](h)
                all_tokens.append(proj_h)
        
        # Concatenate tokens from all experts: (B, Num_Models * C, d_fusion)
        kv = torch.cat(all_tokens, dim=1)
        B = kv.shape[0]
        
        # 2. Query Generation (Dynamic or Static)
        if self.use_dynamic_queries:
            q = self.query_gen(batch['x']) # (B, num_queries, d_fusion)
        else:
            q = self.queries.expand(B, -1, -1)
            
        # 3. TokenPacker: Aggregate info from all models into queries
        attn_out, _ = self.cross_attn(q, kv, kv)
        packed_tokens = self.norm1(q + attn_out) # (B, num_queries, d_fusion)
        
        # 4. Soft MoE: Fuse features in the latent space
        fused_tokens = self.soft_moe(packed_tokens)
        fused_tokens = self.norm2(packed_tokens + fused_tokens)
        
        # 5. Channel-wise prediction: (B, num_queries, pred_len)
        out = self.output_head(self.dropout(fused_tokens)) 
        
        # 6. Information Aggregation: Map queries to actual feature count
        output = self.aggregate(out)
        
        # 7. Final output formatting
        return output.transpose(1, 2)
