import numpy as np
import torch
from scipy.linalg import toeplitz
import torch.nn as nn


class REM(nn.Module):
    # initialise the object
    def __init__(self, k1, k2, k3, k4, k5, k6, d, device):
        super(REM, self).__init__()

        self.k1 = k1  # (reg)
        self.k2 = k2  # (c1)
        self.k3 = k3  # (c2)
        # dilated versions
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6

        self.d = d
        self.device = device

    def get_sinusoid(self, L, theta):
        M = L * theta
        s1 = torch.cos(M[:self.k2, ]).to(self.device)
        s2 = torch.sin(M[self.k2:(self.k2 + self.k3), ]).to(self.device)
        s3 = torch.cos(M[(self.k2 + self.k3):(self.k2 + self.k3 + self.k5), ])  # !
        s4 = torch.sin(M[(self.k2 + self.k3 + self.k5):(self.k2 + self.k3 + self.k5 + self.k6), ])  # !
        s = torch.concat([s1, s2, s3, s4]).to(self.device)
        return s

    def forward(self, eta, nu, theta, query_length, key_length):
        # print(f'k1: {self.k1}, k2: {self.k2}, k3: {self.k3}, k4: {self.k4}, k5: {self.k5}, k6: {self.k6} ')
        lambda_ = torch.tanh(eta).to(self.device)
        gamma = torch.sigmoid(nu).to(self.device)
        L = self.create_Toeplitz_3D(query_length, key_length)  # L is of shape (n_heads x query_len x key_len)

        powered_lambda = pow(lambda_, L)
        powered_gamma = pow(gamma, L)
        s = self.get_sinusoid(L, theta)  # cyclics

        powered_gamma = powered_gamma[:self.k2 + self.k3 + self.k5 + self.k6]  # !
        powered_lambda = powered_lambda[:self.k1 + self.k4]  # !

        # initialize with just the regulars and then add the dilated
        regular_rems = powered_lambda[:self.k1]
        cyclic_rems = powered_gamma[:self.k2 + self.k3]

        # print(self.d)
git
        d_copy = self.d.copy()

        # dilate regular rems: (k4)
        n_dilated_regs = self.k4
        for i in range(n_dilated_regs):
            dilated_reg_rem = torch.kron(powered_lambda[self.k1 + i], torch.eye(n=d_copy.pop()).to(self.device)).to(
                self.device)
            dilated_reg_rem = dilated_reg_rem[:L.shape[1], :L.shape[2]]
            regular_rems = torch.concat([regular_rems, torch.unsqueeze(dilated_reg_rem, 0).to(self.device)]).to(
                self.device)

        # dilate cyclic rems: (k5, k6)
        n_dilated_cyclics = self.k5 + self.k6

        # print(f'n_dilated_regs: {n_dilated_regs}, n_dilated_cyclics: {n_dilated_cyclics}')
        for j in range(n_dilated_cyclics):
            # print(f'd before pop: {self.d}')
            dilated_cyclic_rem = torch.kron(s[self.k2 + self.k3 + j], torch.eye(n=d_copy.pop()).to(self.device)).to(
                self.device)
            dilated_cyclic_rem = dilated_cyclic_rem[:L.shape[1], :L.shape[2]]
            cyclic_rems = torch.concat([cyclic_rems, torch.unsqueeze(dilated_cyclic_rem, 0).to(self.device)]).to(
                self.device)

        # mask & -I
        REM = torch.concat([regular_rems, (cyclic_rems * s)]).to(self.device)
        REM = torch.tril(REM).to(self.device) - torch.eye(n=REM.shape[1], m=REM.shape[2]).to(self.device)
        return REM

    def create_Toeplitz_3D(self, query_len, key_len):
        x = np.arange(0, key_len)
        y = np.arange(0, query_len)

        A = toeplitz(y, x)
        A[A > 200] = 0
        L = torch.from_numpy(A).to(self.device)
        L = torch.stack([L] * 4, 0).to(self.device)
        return L

