import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePoolLinearHead(nn.Module):
    """
    Temporary runnable head used until the real expert head is filled in.

    Replace this implementation inside M1PredictionHead/M2PredictionHead/etc.
    with the original expert model's prediction head structure.
    """

    def __init__(self, hidden_dim, pred_len, n_features, **_):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.proj = nn.Linear(hidden_dim, n_features * pred_len)

    def _pool_hidden(self, hidden):
        if hidden.dim() == 2:
            return hidden

        batch_size = hidden.shape[0]
        hidden = hidden.reshape(batch_size, -1, hidden.shape[-1])
        return hidden.mean(dim=1)

    def forward(self, hidden):
        pooled = self._pool_hidden(hidden)
        return self.proj(pooled).view(pooled.shape[0], self.n_features, self.pred_len)


class M1PredictionHead(SimplePoolLinearHead):
    """TODO: replace with model1's original prediction head."""


class M2PredictionHead(SimplePoolLinearHead):
    """TODO: replace with model2's original prediction head."""


class M3PredictionHead(SimplePoolLinearHead):
    """TODO: replace with model3's original prediction head."""


class M4PredictionHead(SimplePoolLinearHead):
    """TODO: replace with model4's original prediction head."""


EXPERT_HEAD_REGISTRY = {
    "m1": M1PredictionHead,
    "m2": M2PredictionHead,
    "m3": M3PredictionHead,
    "m4": M4PredictionHead,
}


class ExpertHeadReconstruction(nn.Module):
    """
    Single-expert head reconstruction experiment.

    This module does not fuse multiple experts. It selects one frozen expert's
    hidden state from batch_tensor, feeds it into a newly initialized prediction
    head with the same intended structure as that expert's original head, and
    trains only this new head.

    Expected input:
        batch_tensor[expert_name]: hidden state from the selected expert.
        batch[target_key]: forecast target during training.

    Output shape:
        (B, n_features, pred_len)
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
        expert_name="m1",
        target_key="observe_power_future",
        loss_type="mse",
        device="cuda",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.expert_name = expert_name
        self.target_key = target_key
        self.loss_type = loss_type

        self._validate_loss_type(loss_type)
        self._validate_expert_name(models_dict)
        hidden_dim = self._resolve_hidden_dim(expert_dims)

        head_cls = EXPERT_HEAD_REGISTRY[expert_name]
        self.prediction_head = head_cls(
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=n_features,
        )

        self.to(device)

    @classmethod
    def _validate_loss_type(cls, loss_type):
        if loss_type not in cls.SUPPORTED_LOSSES:
            valid = ", ".join(sorted(cls.SUPPORTED_LOSSES))
            raise ValueError(f"Unknown loss_type={loss_type!r}. Valid: {valid}.")

    def _validate_expert_name(self, models_dict):
        if self.expert_name not in EXPERT_HEAD_REGISTRY:
            valid = ", ".join(sorted(EXPERT_HEAD_REGISTRY))
            raise ValueError(
                f"Unknown expert_name={self.expert_name!r}. Valid: {valid}."
            )

        if models_dict is not None and self.expert_name not in models_dict:
            available = ", ".join(models_dict.keys())
            raise ValueError(
                f"expert_name={self.expert_name!r} is not in models_dict. "
                f"Available experts: {available}."
            )

    def _resolve_hidden_dim(self, expert_dims):
        resolved_dims = dict(self.DEFAULT_EXPERT_DIMS)
        if expert_dims is not None:
            resolved_dims.update(expert_dims)

        if self.expert_name not in resolved_dims:
            raise ValueError(
                f"Missing expert_dims for {self.expert_name!r}. "
                "ExpertHeadReconstruction needs the selected expert hidden dimension."
            )
        return resolved_dims[self.expert_name]

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
        if self.expert_name not in batch_tensor:
            available = ", ".join(batch_tensor.keys())
            raise KeyError(
                f"Missing hidden tensor for expert {self.expert_name!r}. "
                f"Available hidden tensors: {available}."
            )

        hidden = batch_tensor[self.expert_name]
        output = self._format_output(self.prediction_head(hidden))

        if flag == "test":
            if return_info:
                return output, {"expert_name": self.expert_name, "hidden": hidden}
            return output

        if flag != "train":
            raise ValueError("flag must be either 'train' or 'test'.")

        target = self._get_target(batch)
        loss = self.loss_func(output, target)
        if return_info:
            return output, loss, {"expert_name": self.expert_name, "hidden": hidden}
        return output, loss


FusionModel = ExpertHeadReconstruction

