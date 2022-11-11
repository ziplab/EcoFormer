import sys

import torch
import torch.nn as nn
import torch.optim as optim

from hashing.utils import BinaryQuantizer, gaussian_kernel, square_dist


class KernelSH(nn.Module):
    """
    Kernel-Based Supervised Hashing (KSH)
    """

    def __init__(
        self,
        head_num=3,
        head_dim=64,
        nbits=16,
        m=20,
        kernelfunc=gaussian_kernel,
        learning_rate=0.0005,
        momentum=0.9,
        iteration_n=100,
    ):
        super().__init__()
        self.nbits = nbits
        self.m = m
        self.kernelfunc = kernelfunc
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.iteration_n = iteration_n
        self.head_dim = head_dim
        self.register_buffer("A", torch.randn(head_num, self.nbits, self.m))
        self.register_buffer("mvec", torch.zeros(1, head_num, 1, self.m))
        self.register_buffer("anchor", torch.zeros(1, head_num, self.m, self.head_dim))
        nn.init.normal_(self.A)

    def compute_loss(self, y, S):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            B, H, N = y.shape
            ys = -torch.einsum("...n,...nl->...l", y, S) / (N * self.nbits)
            loss = torch.einsum("bhl,bhl->bh", ys, y).mean()
        return loss

    def compute_loss_fast(self, y, S):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            B, H, N, K = y.shape
            ys = -torch.einsum("...nk,...nl->...lk", y, S) / (N * self.nbits)
            loss = torch.einsum("bhlk,bhlk->bhk", ys, y).sum(2).mean()
        return loss

    def optimization(self, ktrain, S, a, iteration_n):
        a.requires_grad_()
        optimizer = optim.SGD(
            [a], lr=self.learning_rate, momentum=self.momentum, nesterov=True
        )

        B, H, N, m = ktrain.shape
        loss_list = []
        for iteration in range(iteration_n):
            a_reshape = a.reshape(H, m)

            y = torch.einsum("bhnm,hm->bhn", ktrain, a_reshape)
            # use sigmoid to replace sign function
            y = 2 / (1 + torch.exp(-y)) - 1
            loss = self.compute_loss(y, S)

            # print("Loss: {}".format(loss))
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return a, loss_list

    def optimization_fast(self, ktrain, S, A, iteration_n):
        A.requires_grad_()
        optimizer = optim.SGD(
            [A], lr=self.learning_rate, momentum=self.momentum, nesterov=True
        )

        B, H, N, m = ktrain.shape
        loss_list = []
        for iteration in range(iteration_n):
            a_reshape = A.reshape(H, self.nbits, m)

            y = torch.einsum("bhnm,hkm->bhnk", ktrain, a_reshape)
            # use sigmoid to replace sign function
            y = 2 / (1 + torch.exp(-y)) - 1
            loss = self.compute_loss_fast(y, S)

            # print("Loss: {}".format(loss))
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return A, loss_list

    def train_hashing_weight_woeig(self, X, anchor, S, train_sample_index=None):
        with torch.no_grad():
            b, h, n, _ = X.shape

            self.anchor.data.copy_(anchor.data)

            # kernel computing
            ktrain = square_dist(X, self.anchor)
            sigma = ktrain.mean(-1).mean(-1)
            ktrain = torch.exp(-ktrain / (2 * sigma.unsqueeze(-1).unsqueeze(-1)))
            self.mvec.data.copy_(
                ktrain.mean(-2, keepdim=True).mean(0, keepdim=True).data
            )
            ktrain = ktrain - self.mvec

            S = S * self.nbits
            if train_sample_index is not None:
                KK = torch.index_select(ktrain, 2, train_sample_index)
            else:
                KK = ktrain
            # RM = torch.einsum("...nm,...nl->...ml", KK, KK)
        # Without init
        for rr in range(self.nbits):
            a = self.A[:, rr, :]
            a_optim = a.new_tensor(a, requires_grad=True)
            a_optim, loss_list = self.optimization(
                KK.detach(), S.detach(), a_optim, self.iteration_n
            )

            if a_optim.isnan().any():
                print("Nan occur!")
                sys.exit(1)

            with torch.no_grad():
                y_previous = torch.einsum("bhnm,hm->bhn", KK, a)
                y_previous = torch.where(y_previous > 0.0, 1.0, -1.0)
                loss_previous = self.compute_loss(y_previous, S)

                y_current = torch.einsum("bhnm,hm->bhn", KK, a_optim)
                y_current = torch.where(y_current > 0.0, 1.0, -1.0)
                loss_current = self.compute_loss(y_current, S)

                if loss_current < loss_previous:
                    self.A[:, rr, :].data.copy_(a_optim.data)
                    y = y_current
                else:
                    self.A[:, rr, :].data.copy_(a.data)
                    y = y_previous

                S = S - torch.einsum("...n,...m->...nm", y, y)

    def train_hashing_weight_woeig_fast(self, X, anchor, S, train_sample_index=None):
        with torch.no_grad():
            b, h, n, _ = X.shape

            self.anchor.data.copy_(anchor.data)

            # kernel computing
            ktrain = square_dist(X, self.anchor)
            sigma = ktrain.mean(-1).mean(-1)
            ktrain = torch.exp(-ktrain / (2 * sigma.unsqueeze(-1).unsqueeze(-1)))
            self.mvec.data.copy_(
                ktrain.mean(-2, keepdim=True).mean(0, keepdim=True).data
            )
            ktrain = ktrain - self.mvec

            S = S * self.nbits
            if train_sample_index is not None:
                KK = torch.index_select(ktrain, 2, train_sample_index)
            else:
                KK = ktrain
            # RM = torch.einsum("...nm,...nl->...ml", KK, KK)
        # nn.init.normal_(self.A)
        A_optim = self.A.new_tensor(self.A, requires_grad=True)
        A_optim, loss_list = self.optimization_fast(
            KK.detach(), S.detach(), A_optim, self.iteration_n
        )

        if A_optim.isnan().any():
            print("Nan occur!")
            sys.exit(1)

        with torch.no_grad():
            y_previous = torch.einsum("bhnm,hmk->bhnk", KK, self.A)
            y_previous = torch.where(y_previous > 0.0, 1.0, -1.0)
            loss_previous = self.compute_loss_fast(y_previous, S)

            y_current = torch.einsum("bhnm,hmk->bhnk", KK, A_optim)
            y_current = torch.where(y_current > 0.0, 1.0, -1.0)
            loss_current = self.compute_loss_fast(y_current, S)

            if loss_current < loss_previous:
                self.A[:, :, :].data.copy_(A_optim.data)
                y = y_current
            else:
                y = y_previous

    def train_hashing_weight(self, X, anchor, S, train_sample_index=None):
        b, h, n, _ = X.shape

        # Choose random subsample of m elements
        # perm = torch.randperm(n, device=X.device)
        # selected_idx = perm[:self.m]
        # shape: b, h, m, _
        # self.anchor = torch.index_select(X, 2, selected_idx)
        self.anchor.data.copy_(anchor.data)

        # kernel computing
        ktrain = square_dist(X, self.anchor)
        sigma = ktrain.mean(-1).mean(-1)
        ktrain = torch.exp(-ktrain / (2 * sigma.unsqueeze(-1).unsqueeze(-1)))
        self.mvec.data = ktrain.mean(-2, keepdim=True)
        ktrain = ktrain - self.mvec

        S = S * self.nbits
        if train_sample_index is not None:
            KK = torch.index_select(ktrain, 2, train_sample_index)
        else:
            KK = ktrain
        _, _, train_n, _ = KK.shape
        RM = torch.einsum("...nm,...nl->...ml", KK, KK)
        for rr in range(self.nbits):
            KKS = torch.einsum("...nm,...nl->...ml", KK, S)
            LM = torch.einsum("...ml,...lk->...mk", KKS, KK)
            v, U = torch.lobpcg(
                A=LM.reshape(-1, self.m, self.m), B=RM.reshape(-1, self.m, self.m), k=1
            )
            U = U.reshape(b, h, self.m)
            a = U

            ARM = torch.einsum("...m,...mn->...n", a, RM)
            tep = torch.einsum("...n,...n->...", ARM, a).squeeze()
            a = torch.sqrt(n / tep) * a

            # with torch.no_grad():
            #     self.A[rr, :].copy_(a.squeeze().data)
            a_optim = a.squeeze().data.clone()
            a_optim.requires_grad = True
            a_optim, loss_list = self.optimization(KK, S, a_optim, self.iteration_n)

            y_previous = torch.einsum("...nm,...m->...n", KK, a)
            y_previous = torch.where(y_previous > 0.0, 1.0, -1.0)
            y_previous_s = -torch.einsum("...n,...nl->...l", y_previous, S)
            loss_previous = (
                torch.einsum("...l,...l->", y_previous_s, y_previous) / train_n
            )

            y_current = torch.einsum("...nm,...m->...n", KK, a_optim)
            y_current = torch.where(y_current > 0.0, 1.0, -1.0)
            y_current_s = -torch.einsum("...n,...nl->...l", y_current, S)
            loss_current = torch.einsum("...l,...l->", y_current_s, y_current) / train_n

            if loss_current < loss_previous:
                self.A[rr, :].data.copy_(a_optim.data)
                y = y_current
            else:
                self.A[rr, :].data.copy_(a.data)
                y = y_previous

            S = S - torch.einsum("...n,...m->...nm", y, y)

    def forward(self, X):
        # kernel computing
        ktest = square_dist(X, self.anchor)
        sigma = ktest.mean(-1).mean(-1)
        ktest = torch.exp(-ktest / (2 * sigma.unsqueeze(-1).unsqueeze(-1)))
        ktest = ktest - self.mvec
        y = torch.einsum("bhnm,hkm->bhnk", ktest, self.A.detach())
        # y = torch.where(y > 0.0, 1.0, -1.0)
        y = BinaryQuantizer.apply(y)
        # binary_y = torch.sign(y)
        # cliped_input = torch.clamp(y, -1.0, 1.0)
        # y = binary_y.detach() - cliped_input.detach() + cliped_input
        return y

