import torch.nn as nn
import torch


class VaeLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, output, x):
        recon_x = output[0]
        mu = output[1]
        logvar = output[2]
        mse = self.criterion(recon_x, x)  # mse loss
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return mse + KLD
