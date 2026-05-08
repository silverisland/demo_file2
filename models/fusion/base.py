import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBase(nn.Module):
    """
    Minimal hidden-feature fusion baseline.

    Expected input:
        batch_tensor: dict[str, Tensor]
            Expert hidden features, e.g. {"m1": h1, "m2": h2}.
            Each tensor may be shaped (B, D), (B, T, D), or (B, ..., D).

        batch: dict[str, Tensor], optional during inference
            Used only to read the target during training.

    Output shape:
        (B, n_features, pred_len)

    This model intentionally avoids attention, MoE, dynamic gates, auxiliary
    heads, and expert-specific logic. It is meant to be the clean baseline for
    incremental fusion experiments.
    """

    DEFAULT_EXPERT_DIMS = {"m1": 512, "m2": 256, "m3": 384, "m4": 512}
    SUPPORTED_LOSSES = {"mse", "mae", "huber"}

    def __init__(
        self,
        models_dict=None,
        seq_len=None,
        pred_len=192,
        n_features=1,
        expert_dims=None,
        expert_names=None,
        d_fusion=128,
        dropout=0.0,
        target_key="observe_power_future",
        loss_type="mse",
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_fusion = d_fusion
        self.target_key = target_key
        self.loss_type = loss_type

        self._validate_loss_type(loss_type)
        self.expert_names = self._resolve_expert_names(
            expert_names=expert_names,
            expert_dims=expert_dims,
            models_dict=models_dict,
        )
        resolved_expert_dims = self._resolve_expert_dims(expert_dims)

        self.projectors = nn.ModuleDict(
            {
                name: nn.Linear(resolved_expert_dims[name], d_fusion)
                for name in self.expert_names
            }
        )
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(
            d_fusion * len(self.expert_names),
            n_features * pred_len,
        )

        self.to(device)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    @classmethod
    def _resolve_expert_names(cls, expert_names, expert_dims, models_dict):
        if expert_names is not None:
            return list(expert_names)
        if models_dict is not None:
            return list(models_dict.keys())
        if expert_dims is not None:
            return list(expert_dims.keys())
        return list(cls.DEFAULT_EXPERT_DIMS.keys())

    def _resolve_expert_dims(self, expert_dims):
        resolved_dims = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved_dims.update(expert_dims)

        missing_dims = [
            name for name in self.expert_names if name not in resolved_dims
        ]
        if missing_dims:
            raise ValueError(
                "Missing expert_dims for: "
                + ", ".join(missing_dims)
                + ". FusionBase needs explicit hidden dimensions for unknown experts."
            )
        return resolved_dims

    def _pool_hidden(self, hidden):
        if hidden.dim() == 2:
            return hidden

        batch_size = hidden.shape[0]
        hidden = hidden.reshape(batch_size, -1, hidden.shape[-1])
        return hidden.mean(dim=1)

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
            raise KeyError(
                "Missing hidden tensors for experts: " + ", ".join(missing)
            )

        projected = []
        for name in self.expert_names:
            hidden = self._pool_hidden(batch_tensor[name])
            projected.append(self.projectors[name](hidden))

        fused = torch.cat(projected, dim=-1)
        fused = self.dropout(fused)
        output = self.output_head(fused).view(
            fused.shape[0], self.n_features, self.pred_len
        )

        if flag == "test":
            if return_info:
                return output, {"fused": fused}
            return output

        if flag != "train":
            raise ValueError("flag must be either 'train' or 'test'.")

        target = self._get_target(batch)
        loss = self.loss_func(output, target)
        if return_info:
            return output, loss, {"fused": fused}
        return output, loss


FusionModel = FusionBase
