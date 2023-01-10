import torch
from torch import nn

# Edit the Dynamic spatial filter module from https://github.com/hubertjb/dynamic-spatial-filtering
class AdaptiveMask(nn.Module):

    def __init__(self, args, return_att=False):
        super().__init__()
        self.args = args
        self.n_channels = args.input_size
        self.gumbel_type = args.df
        self.gumbel_tau = args.df_tau
        self.gumbel_soft = args.df_soft
        self.inds = torch.triu_indices(self.n_channels, self.n_channels, 1)
        self.return_att = return_att

        self.gumbel = GumbelSigmoid(args.df_tau)

        n_inputs = int(self.n_channels * (self.n_channels-1) / 2)
        self.mlp = nn.Sequential(nn.Linear(n_inputs, int(n_inputs/ 2)),
                                 nn.ReLU(),
                                 nn.Linear(int(n_inputs / 2), n_inputs)
                                 )
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            nn.init.xavier_normal_(weight)

    def set_require_grad(self, isgrad):
        for param in self.parameters():
            param.requires_grad = isgrad

    def forward(self, x):
        b, _, _ = x.shape # b = the number of batches
        x_flatten = x[:, self.inds[0], self.inds[1]] # b x 5995 (upper triangle of 110x110 matrix)
        mlp_out = self.mlp(x_flatten) # b x 5995
        if self.gumbel_type == 'gumbel':
            mlp_out = self.gumbel(mlp_out, self.gumbel_soft)
        elif self.gumbel_type == 'sigmoid':
            mlp_out = F.sigmoid(mlp_out)

        W_upper = torch.zeros_like(x)
        W_upper[:, self.inds[0], self.inds[1]] = mlp_out
        W = W_upper + W_upper.transpose(1, 2)
        out = torch.mul(W, x)

        if self.return_att:
            return out, W
        else:
            return out, mlp_out

import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    # y = logits + sample_gumbel(logits.size())
    y = F.log_softmax(logits, dim=-1) + sample_gumbel(logits.size())
    return F.softmax(y / temperature)

class GumbelSoftmax():
    def __init__(self, t=0.1):
        assert t != 0
        self.temperature = t
        self.eps = 1e-20

    def __call__(self, logits, soft):
        """computes a gumbel softmax sample
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = gumbel_softmax_sample(logits, self.temperature)
        ind = torch.argmax(y, dim=-1)
        ind_0 = torch.where(ind == 0)
        ind_1 = torch.where(ind == 1)
        y_hard = torch.zeros_like(y)
        y_hard[ind_0[0], ind_0[1], 0] = 1
        y_hard[ind_1[0], ind_1[1], 0] = 1

        # shape = y.size()
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # draw a sample from the Gumbel-Sigmoid distribution
        # return (y_hard - y).detach() + y
        # return y
        if soft:
            print("Dynamic Soft")
            return y
        else:
            print("Dynamic Hard")
            return (y_hard - y).detach() + y


class GumbelSigmoid():
    """
    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit

    Explaination and motivation: https://arxiv.org/abs/1611.01144

    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)

    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)

    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1))) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )

    :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
    :param eps: a small number used for numerical stability
    :returns: a callable that can (and should) be used as a nonlinearity

    Reference:
    https://github.com/yandexdataschool/gumbel_lstm/blob/763a2f01dc3686d7269003e6eeb1c80fdc4cddde/gumbel_sigmoid.py
    """

    def __init__(self, t, eps=1e-20):
        assert t != 0
        self.temperature = t
        self.eps = eps

    def __call__(self, logits, soft):
        """computes a gumbel sigmoid """

        # sample from Gumbel(0, 1)
        uniform1 = torch.rand(logits.size()).cuda()
        uniform2 = torch.rand(logits.size()).cuda()

        noise = -Variable(torch.log(torch.log(uniform2 + self.eps) / torch.log(uniform1 + self.eps) + self.eps))
        y = torch.sigmoid((logits + noise) / self.temperature)

        # draw a sample from the Gumbel-Sigmoid distribution
        if soft:
            return y
        else:

            ind = torch.where(y > 0.5)
            y_hard = torch.zeros_like(y)
            y_hard[ind[0], ind[1]] = 1
            return y_hard - y.detach() + y


