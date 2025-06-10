import torch
import torch.nn.functional as F
from torch import Tensor


class BPRLoss(torch.nn.Module):
    def __init__(self, lambda_reg: float = 0):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        log_prob = F.logsigmoid(positives - negatives).mean()
        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / positives.size(0)
        return -log_prob + regularization
