import torch
import torch.nn as nn
import torch.nn.functional as F


class PGPointConvLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_exp=0.5, lambda_risk=0.1):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_exp = lambda_exp
        self.lambda_risk = lambda_risk

    def forward(self, seg_logits, labels, risk_pred, expert_scores, risk_phys):
        L_cls = F.cross_entropy(seg_logits.view(-1, 5), labels.view(-1))
        L_exp = F.mse_loss(risk_pred.view(-1), expert_scores.view(-1))
        L_risk = F.mse_loss(risk_pred.view(-1), risk_phys.view(-1))

        L_total = self.lambda_cls * L_cls + self.lambda_exp * L_exp + self.lambda_risk * L_risk
        return L_total, L_cls, L_exp, L_risk