import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.revin import RevIN


class ExpertTokenAdapter(nn.Module):
    """
    Maps an expert hidden tensor with arbitrary token shape to aligned latent
    tokens shaped (B, output_tokens, d_model).
    """

    def __init__(self, input_dim, output_tokens, d_model=128, dropout=0.0, num_heads=4):
        super().__init__()
        self.output_tokens = output_tokens
        self.d_model = d_model

        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.queries = nn.Parameter(torch.randn(1, output_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _as_tokens(self, hidden):
        if hidden.dim() == 2:
            return hidden.unsqueeze(1)
        if hidden.dim() == 3:
            return hidden

        batch_size = hidden.shape[0]
        return hidden.reshape(batch_size, -1, hidden.shape[-1])

    def forward(self, hidden):
        tokens = self._as_tokens(hidden)
        tokens = self.projector(tokens)
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        aligned, _ = self.attn(queries, tokens, tokens)
        return self.norm(queries + self.dropout(aligned))


class TokenForecastHead(nn.Module):
    """
    Forecast head shared in structure across experts after latent alignment.
    Each aligned token emits a forecast; a token gate aggregates them.
    """

    def __init__(self, d_model=128, pred_len=192, n_features=1, dropout=0.0):
        super().__init__()
        hidden_dim = max(d_model // 2, 1)
        self.pred_len = pred_len
        self.n_features = n_features

        self.token_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features * pred_len),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        batch_size, token_count, _ = tokens.shape
        pred = self.token_head(tokens).view(
            batch_size, token_count, self.n_features, self.pred_len
        )
        weight = F.softmax(self.token_gate(tokens), dim=1).view(
            batch_size, token_count, 1, 1
        )
        return (pred * weight).sum(dim=1)


class AlignedExpertHeadFusion(nn.Module):
    """
    V2 expert-head fusion.

    Each expert uses a private adapter to map its hidden state into a common
    latent dimension, then a private forecast head predicts power from those
    aligned tokens. Final fusion is a fixed mean over expert predictions.
    """

    DEFAULT_EXPERT_DIMS = {"m1": 128, "m2": 512, "m3": 384, "m4": 256}
    DEFAULT_ALIGNED_TOKENS = {"m1": 9, "m2": 2, "m3": 16, "m4": 9}
    SUPPORTED_LOSSES = {"mse", "mae", "huber"}

    def __init__(
        self,
        models_dict=None,
        seq_len=None,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        expert_names=None,
        aligned_tokens=None,
        d_fusion=None,
        dropout=0.0,
        target_key="observe_power_future",
        loss_type="mse",
        aux_loss_weight=1.0,
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = 128 if d_fusion is None else d_fusion
        self.target_key = target_key
        self.loss_type = loss_type
        self.aux_loss_weight = aux_loss_weight
        self.expert_names = self._resolve_expert_names(models_dict, expert_names)

        self._validate_loss_type(loss_type)
        resolved_dims = self._resolve_expert_dims(expert_dims)
        resolved_tokens = self._resolve_aligned_tokens(aligned_tokens)
        num_heads = self._choose_num_heads(self.d_fusion)

        self.pv_revin_layer = RevIN(1, affine=1, subtract_last=0)
        self.adapters = nn.ModuleDict()
        self.prediction_heads = nn.ModuleDict()
        for name in self.expert_names:
            self.adapters[name] = ExpertTokenAdapter(
                input_dim=resolved_dims[name],
                output_tokens=resolved_tokens[name],
                d_model=self.d_fusion,
                dropout=dropout,
                num_heads=num_heads,
            )
            self.prediction_heads[name] = TokenForecastHead(
                d_model=self.d_fusion,
                pred_len=pred_len,
                n_features=n_features,
                dropout=dropout,
            )

        self.to(device)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    @staticmethod
    def _choose_num_heads(d_model):
        for num_heads in (8, 4, 2, 1):
            if d_model % num_heads == 0:
                return num_heads
        return 1

    def _resolve_expert_names(self, models_dict, expert_names):
        if expert_names is not None:
            missing = (
                []
                if models_dict is None
                else [name for name in expert_names if name not in models_dict]
            )
            if missing:
                raise ValueError(
                    "fusion_expert_names contains experts not in models_dict: "
                    + ", ".join(missing)
                )
            return list(expert_names)

        if models_dict is None:
            return list(self.DEFAULT_EXPERT_DIMS.keys())
        return list(models_dict.keys())

    def _resolve_expert_dims(self, expert_dims):
        resolved = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved.update(expert_dims)

        missing = [name for name in self.expert_names if name not in resolved]
        if missing:
            raise ValueError(
                "Missing expert_dims for: "
                + ", ".join(missing)
                + ". AlignedExpertHeadFusion needs each expert hidden dimension."
            )
        return resolved

    def _resolve_aligned_tokens(self, aligned_tokens):
        resolved = dict(self.DEFAULT_ALIGNED_TOKENS)
        if aligned_tokens is not None:
            resolved.update(aligned_tokens)

        missing = [name for name in self.expert_names if name not in resolved]
        if missing:
            raise ValueError(
                "Missing aligned token counts for: "
                + ", ".join(missing)
                + ". Use fusion_aligned_tokens like 'm1:9,m2:2,m4:9'."
            )
        return resolved

    def _format_output(self, output):
        if output.dim() == 2:
            output = output.unsqueeze(1)
        elif output.dim() == 3 and output.shape[1] == self.pred_len:
            output = output.transpose(1, 2)

        expected_shape = (output.shape[0], self.n_features, self.pred_len)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"Prediction head output must be {expected_shape}, "
                f"got {tuple(output.shape)}."
            )
        return output

    def _set_revin_statistics(self, batch):
        if batch is None:
            raise ValueError("batch is required for RevIN normalization.")

        pv_his = batch["observe_power"].unsqueeze(1)
        tsfm = batch["chronos"].unsqueeze(1)
        pv = torch.cat([pv_his, tsfm], dim=1)
        pv = pv.permute(0, 2, 1)
        self.pv_revin_layer(pv, "norm")

    def _denorm_output(self, output):
        output = output.permute(0, 2, 1)
        output = self.pv_revin_layer(output, "denorm")
        if output.shape[-1] != self.n_features:
            output = output[..., : self.n_features]
        return output.permute(0, 2, 1)

    def _get_target(self, batch):
        if batch is None:
            raise ValueError("batch is required when flag is not 'test'.")

        if self.target_key in batch:
            target = batch[self.target_key]
        elif "target_power" in batch:
            target = batch["target_power"]
        else:
            raise KeyError(
                f"Cannot find target key '{self.target_key}' or 'target_power' in batch."
            )

        if target.dim() == 2:
            target = target.unsqueeze(1)
        elif target.dim() == 3 and target.shape[1] == self.pred_len:
            target = target.transpose(1, 2)

        expected_shape = (target.shape[0], self.n_features, self.pred_len)
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"Target shape must be {expected_shape}, got {tuple(target.shape)}."
            )
        return target

    def loss_func(self, pred, target):
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=1.0)
        raise ValueError(f"Unknown loss_type={self.loss_type!r}")

    def forward(self, batch_tensor, batch=None, flag="test", return_info=False):
        missing = [name for name in self.expert_names if name not in batch_tensor]
        if missing:
            raise KeyError("Missing hidden tensors for experts: " + ", ".join(missing))

        self._set_revin_statistics(batch)

        aligned_by_expert = {}
        pred_by_expert = {}
        preds = []
        for name in self.expert_names:
            aligned = self.adapters[name](batch_tensor[name])
            pred = self._format_output(self.prediction_heads[name](aligned))
            pred = self._denorm_output(pred)
            aligned_by_expert[name] = aligned
            pred_by_expert[name] = pred
            preds.append(pred)

        pred_stack = torch.stack(preds, dim=1)
        output = pred_stack.mean(dim=1)

        info = {
            "expert_names": self.expert_names,
            "aligned_by_expert": aligned_by_expert,
            "pred_by_expert": pred_by_expert,
            "pred_stack": pred_stack,
        }

        if flag == "test":
            if return_info:
                return output, info
            return output.squeeze(1)

        if flag != "train":
            raise ValueError("flag must be either 'train' or 'test'.")

        target = self._get_target(batch)
        main_loss = self.loss_func(output, target)
        aux_losses = torch.stack(
            [self.loss_func(pred_by_expert[name], target) for name in self.expert_names]
        )
        aux_loss = aux_losses.mean()
        loss = main_loss + self.aux_loss_weight * aux_loss

        info.update(
            {
                "main_loss": main_loss.detach(),
                "aux_loss": aux_loss.detach(),
                "aux_losses": {
                    name: aux_losses[i].detach()
                    for i, name in enumerate(self.expert_names)
                },
            }
        )

        if return_info:
            return output, loss, info
        return output, loss


FusionModel = AlignedExpertHeadFusion
