import torch
from torch import optim


class SharedAdam(optim.Adam):
    """Adam variant with state pre-allocated so it can be moved to shared memory."""

    def __init__(self, params, lr=1e-3, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                if param is None:
                    continue
                state = self.state[param]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(param.data)
                state["exp_avg_sq"] = torch.zeros_like(param.data)

    def share_memory(self):
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if not state:
                    continue

                for key in ("step", "exp_avg", "exp_avg_sq"):
                    value = state.get(key, None)

                    if value is None or not isinstance(value, torch.Tensor):
                        continue

                    if value.is_cuda:
                        continue
                    value.share_memory_()

        return self