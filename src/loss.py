
import torch
import torch.nn as nn
import torch.nn.functional as F


class VaeLoss(nn.Module):
    def __init__(self, rec_weight, kld_weight, **kwargs):
        super(VaeLoss, self).__init__()
        self.rec_weight = rec_weight
        self.kld_weight = kld_weight

    def forward(self, x_true, x_pred, mean, log_var, z, label_true=None):
        rec = F.mse_loss(x_pred, x_true, reduction='mean')
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
        loss = rec * self.rec_weight + kld * self.kld_weight
        return loss, rec, kld


class VaeLossAuxiliary(VaeLoss):
    def __init__(self, rec_weight, kld_weight, mse_weight, lspace_size, n_outputs, **kwargs):
        super(VaeLossAuxiliary, self).__init__(rec_weight, kld_weight)
        self.mse_weight = mse_weight
        self.fc_outputs = nn.Linear(lspace_size, n_outputs)

    def forward(self, x_true, x_pred, mean, log_var, z, outputs_true):
        _loss, rec, kld = super(VaeLossAuxiliary, self).forward(x_true, x_pred, mean, log_var, z, label_true)
        if outputs_true is not None and len(outputs_true) == x_true.shape[0]:
            logits = self.fc_outputs(z)
            mse = F.mse_loss(logits, outputs_true, reduction='mean')
        else:
            mse = torch.Tensor([0.0]).to(x_true.device)
        loss = _loss + mse * self.mse_weight
        return loss, rec, kld, mse

