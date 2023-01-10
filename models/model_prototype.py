import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
import torch.nn.functional as F



class ProtoClassifier(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.asd_prototypes = nn.Parameter(torch.randn(args.input_size, args.num_proto_asd), requires_grad=True)
        self.td_prototypes = nn.Parameter(torch.randn(args.input_size, args.num_proto_td), requires_grad=True)

        self.pdist = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)

    def set_require_grad(self, isgrad):
        self.asd_prototypes.requires_grad = isgrad
        self.td_prototypes.requires_grad = isgrad

    def forward(self, cls_token, avg_roi_token=None):
        asd_protos = self.asd_prototypes.expand(cls_token.shape[0], -1, -1)
        td_protos = self.td_prototypes.expand(cls_token.shape[0], -1, -1)
        asd_sim = self.pdist(cls_token.transpose(2, 1), asd_protos)
        td_sim = self.pdist(cls_token.transpose(2, 1), td_protos)
        output = self.args.h_sca * torch.cat([td_sim, asd_sim], dim=1)

        if avg_roi_token is not None:
            asdr_sim = self.pdist(avg_roi_token.transpose(2, 1), asd_protos)
            tdr_sim = self.pdist(avg_roi_token.transpose(2, 1), td_protos)
            reg = torch.cat([tdr_sim, asdr_sim], dim=1)
            return output, reg
        else:
            return output


