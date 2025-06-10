import torch
import torch.nn.functional as F
from torch import Tensor


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature: float, b_cos: bool = True):
        super().__init__()
        self.temperature = temperature
        self.b_cos = b_cos

    def forward(self, view1: Tensor, view2: Tensor) -> Tensor:
        if self.b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / self.temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
