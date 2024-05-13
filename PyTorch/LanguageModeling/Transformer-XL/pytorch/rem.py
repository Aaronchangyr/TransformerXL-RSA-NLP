import numpy as np
import torch
from scipy.linalg import toeplitz
import torch.nn as nn


class REM(nn.Module):
    # initialise the object
    def __init__(self, k1, k2, k3, k4, k5, k6, d, truncation, T, device):
        super(REM, self).__init__()

        self.k1 = k1  # (reg)
        self.k2 = k2  # (c1)
        self.k3 = k3  # (c2)
        # dilated versions:
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6

        self.d = d
        self.truncation = truncation

        self.T = T
        self.device = device

    def get_sinusoid(self, L, theta):
        # ? k4->k5, k4:->k6
        # ? these are the cyclical rems
        M = L * theta
        s1 = torch.cos(M[:self.k2, ]).to(self.device)  # P(c1)s if exists
        s2 = torch.sin(M[self.k2:(self.k2 + self.k3), ]).to(self.device)  # P(c2)s if exists

        s3, s4 = torch.empty(0).to(self.device), torch.empty(0).to(self.device)
        if self.k5:
            s3 = torch.stack([torch.cos(self.get_dilated_L(L[0]) * theta).to(self.device) for _ in range(self.k5)]).to(
                self.device)  # P(c1 & dilated)s if exists
        if self.k6:
            s4 = torch.stack([torch.sin(self.get_dilated_L(L[0]) * theta).to(self.device) for _ in range(self.k6)]).to(
                self.device)  # P(c2 & dilated)s if exists

        s = torch.concat([s1, s2, s3, s4]).to(self.device)
        return s

    def get_regulars(self, lambda_, L):
        powered_lambda = pow(lambda_, L[0])
        r1, r2 = torch.empty(0).to(self.device), torch.empty(0).to(self.device)
        if self.k1:
            r1 = torch.stack([powered_lambda] * self.k1, 0).to(self.device)
        if self.k4:
            r2 = torch.stack([pow(lambda_, self.get_dilated_L(L[0])) for _ in range(self.k4)]).to(self.device)
        r = torch.concat([r1, r2]).to(self.device).to(self.device)
        return r

    def forward(self, eta, nu, theta):
        print(f'k1-6: {self.k1, self.k2, self.k3, self.k4, self.k5, self.k6}')
        lambda_ = torch.tanh(eta).to(self.device)
        gamma = torch.sigmoid(nu).to(self.device)
        print(f'lambda_: {lambda_}, gamma: {gamma}')
        L = self.create_Toeplitz_3D(self.d, self.truncation)  # L is of shape (n_heads x query_len x key_len)

        # must do regular dilated before cyclic dilated
        r = self.get_regulars(lambda_, L)  # regulars

        powered_gamma = pow(gamma, L[0])  # ([0] - to get just a single L toeplitz)

        s = self.get_sinusoid(L, theta)  # cyclics
        print(f's.shape: {s.shape}, r.shape: {r.shape}')

        # ? not doing the -identity currently...
        # ? not doing mask currently... (lower triangular..)

        REM = torch.concat([r, (powered_gamma * s)]).to(self.device)
        # ! does the order matter here? order of rems?
        # # ! with -identity:
        # REM = torch.concat([r, (powered_gamma * s)]).to(self.device) - torch.eye(n = L[0].shape[0], m = L[0].shape[1]).to(self.device)

        # # ! mask? why does it need -I also... thought tril does this but evidently not getting that...
        REM = torch.tril(REM).to(self.device) - torch.eye(n=REM.shape[1], m=REM.shape[2]).to(self.device)

        # print(REM[0])
        return REM

    def create_Toeplitz_3D(self, d, truncation):
        T = np.arange(self.T)  # TODO: define T - token number #!
        A = toeplitz(c=T)
        A[A > 200] = 0
        L = torch.from_numpy(A).to(self.device)
        L = L[:][:truncation]  # !

        # ! adding mask
        # L = torch.tril(L)

        # ! stack?
        L = torch.stack([L] * 8, 0).to(self.device)  # n_heads - 8
        return L

    # ! feels wrong... but also can't see how else to interpret this?
    def get_dilated_L(self, L):
        d = self.d.pop()
        L_d = L.detach().clone()
        L_d = L_d.to(torch.float)
        L_d[L_d % d != 0] = 0.0
        L_d /= d
        return L_d