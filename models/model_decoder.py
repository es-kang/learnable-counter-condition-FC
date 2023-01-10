import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
import torch.nn.functional as F




class CLS_Decoder(nn.Module):

    def __init__(self, args, dropout=0.3):
        super().__init__()

        self.args = args
        output_size = int((args.input_size * (args.input_size-1)) / 2)
        self.dec = nn.Sequential(nn.Linear(int(args.input_size), int(output_size/2)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(int(output_size/2), output_size))
        self.inds = torch.triu_indices(args.input_size, args.input_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)

    def set_require_grad(self, isgrad):
        self.dec.requires_grad = isgrad

    def forward(self, x):
        output = self.dec(x)
        return output


