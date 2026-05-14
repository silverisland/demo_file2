import torch
from torch.optim import Optimizer


def _zeropower_via_newton_schulz(matrix, steps):
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization expects a 2D tensor.")

    dtype = matrix.dtype
    matrix = matrix.float()
    if matrix.size(0) > matrix.size(1):
        matrix = matrix.t()
        transposed = True
    else:
        transposed = False

    matrix = matrix / (matrix.norm() + 1e-7)
    for _ in range(steps):
        a = matrix @ matrix.t()
        b = 3.4445 * a - 4.7750 * (a @ a) + 2.0315 * (a @ a @ a)
        matrix = b @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix.to(dtype)


class Muon(Optimizer):
    """
    Muon optimizer with AdamW fallback groups.

    Parameters in groups with use_muon=True are updated with momentum plus
    Newton-Schulz orthogonalized updates. Other groups use AdamW-style updates.
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "use_muon": True,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", True):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)

            update = buf
            original_shape = update.shape
            if update.ndim > 2:
                update = update.view(update.shape[0], -1)

            update = _zeropower_via_newton_schulz(update, ns_steps)
            update = update.view(original_shape)
            scale = max(1.0, original_shape[0] / max(original_shape[-1], 1)) ** 0.5
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)
