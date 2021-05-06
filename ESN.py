import torch


def select_indexes(states, indexes):
    sub_states = {}
    for layer in states:
        sub_states[layer] = states[layer][:, indexes]
    return sub_states


class DeepESN:

    def __init__(self, n_layer, dim_in, reservoir_size, ae_size, input_scale, inter_scale,
                 density, leaky_rate, spectral_radius, norm):

        self.n_layer = n_layer

        self.reservoir_size = reservoir_size
        self.ae_size = ae_size

        self.leaky_rate = leaky_rate
        self.norm = norm
        self.actf = torch.tanh

        self.input_scale = input_scale
        self.inter_scale = inter_scale

        self.Winputs = {}
        self.Wrec = {}
        self.Win_ae = {}
        self.Wout_ae = {}

        self.W_readout = torch.zeros(1)

        for layer in range(n_layer):

            # Input weights
            self.Winputs[layer] = 2 * torch.rand(reservoir_size[layer], dim_in[layer]) - 1

            # Reccurent weights
            wtmp = 2 * torch.rand(reservoir_size[layer], reservoir_size[layer]) - 1
            if density[layer] < 1.0:
                n_zeros = int(torch.round(reservoir_size[layer] * (1 - density[layer])))
                for row in range(reservoir_size[layer]):
                    row_zeros = torch.randperm(reservoir_size[layer])[:n_zeros]
                    wtmp[row, row_zeros] = 0.0

            ws = (1 - leaky_rate[layer]) * torch.eye(reservoir_size[layer]) + leaky_rate[layer] * wtmp
            actual_rho = torch.max(torch.absolute(torch.eig(ws)[0]))
            ws = ws * spectral_radius[layer] / actual_rho
            self.Wrec[layer] = (1 / leaky_rate[layer]) * (
                    ws - (1 - leaky_rate[layer]) * torch.eye(reservoir_size[layer]))

    def forward(self, x, train_ae=False, initial_states=None):

        states0 = {}
        if initial_states is None:
            for layer in range(self.n_layer):
                states0[layer] = torch.zeros(self.Wrec[layer].shape[0], 1)
        else:
            states0 = initial_states

        states = {}
        henc = torch.zeros(1)
        for layer in range(self.n_layer):

            states[layer] = torch.zeros(self.Wrec[layer].shape[0], x[0].shape[1])

            if layer == 0:
                x_in = self.input_scale[0] * torch.mm(self.Winputs[0], x[0])
            else:
                x_in = self.inter_scale[layer] * torch.mm(henc, self.Wout_ae[layer - 1]).T
                x_in += self.input_scale[layer] * torch.mm(self.Winputs[layer], x[layer])

            states[layer][:, 0:1] = (1 - self.leaky_rate[layer]) * states0[layer] + self.leaky_rate[layer] * self.actf(
                torch.mm(self.Wrec[layer], states0[layer]) + x_in[:, 0:1])

            if self.norm[layer] > 0:
                states[layer][:, 0:1] *= self.norm[layer] / torch.linalg.norm(states[layer][:, 0:1])

            for t in range(1, x_in.shape[1]):
                states[layer][:, t:t + 1] = (1 - self.leaky_rate[layer]) * states[layer][:, t - 1:t] + self.leaky_rate[
                    layer] * self.actf(torch.mm(self.Wrec[layer], states[layer][:, t - 1:t]) + x_in[:, t:t + 1])
                if self.norm[layer] > 0:
                    states[layer][:, t:t + 1] *= self.norm[layer] / torch.linalg.norm(states[layer][:, t:t + 1])

            if layer < self.n_layer - 1:
                if train_ae:
                    wtmp = (2 * torch.rand(self.ae_size[layer], self.reservoir_size[layer]) - 1)
                    h = self.actf(torch.mm(wtmp, states[layer]))
                    b = torch.mm(torch.linalg.pinv(h.T), states[layer].T)
                    self.Win_ae[layer] = b.T
                    self.Wout_ae[layer] = torch.linalg.pinv(b.T)

                henc = torch.mm(states[layer].T, self.Win_ae[layer])

        return states

    def train_readout(self, train_states, train_targets, lb=0.01):

        readout = train_states[0]
        for layer in range(1, self.n_layer):
            readout = torch.cat((readout, train_states[layer]), 0)
        n_s = readout.shape[1]

        # add bias
        x = torch.cat((readout, torch.ones(1, n_s)), 0)
        y = train_targets

        b = torch.mm(y, x.T)
        a = torch.mm(x, x.T)

        self.W_readout = torch.linalg.solve((a + torch.eye(a.shape[0]) * lb), b.T).T

        return self.W_readout

    def compute_readout(self, states):

        readout = states[0]
        for layer in range(1, self.n_layer):
            readout = torch.cat((readout, states[layer]), 0)

        return torch.mm(self.W_readout[:, 0:-1], readout) + self.W_readout[:, -1:]

    def train_readout_EO(self, train_states, train_targets, bounds=1, npop=30, max_iter=100, verbose=False):

        Pool_Size = 4
        a1, a2, GP = 2, 1, 0.5

        readout = train_states[0]
        for layer in range(1, self.n_layer):
            readout = torch.cat((readout, train_states[layer]), 0)

        dim = train_targets.shape[0] * (readout.shape[0] + 1)
        C = 2 * bounds * (torch.rand(npop, dim) - 0.5)
        Ceq_val = torch.zeros((Pool_Size, dim))
        Ceq_fit = torch.ones(Pool_Size) * float('Inf')

        # EO main loop
        for iter in range(max_iter):

            fitness = torch.zeros(npop)
            for i in range(npop):
                W_readout = C[i].view(train_targets.shape[0], readout.shape[0]+1)
                outp = torch.mm(W_readout[:, 0:-1], readout) + W_readout[:, -1:]
                fitness[i] = torch.mean((outp - train_targets) ** 2)

                if fitness[i] < Ceq_fit[0]:
                    Ceq_val[0], Ceq_fit[0] = C[i], fitness[i]
                elif fitness[i] > Ceq_fit[0] and fitness[i] < Ceq_fit[1]:
                    Ceq_val[1], Ceq_fit[1] = C[i], fitness[i]
                elif fitness[i] > Ceq_fit[0] and fitness[i] > Ceq_fit[1] and fitness[i] < Ceq_fit[2]:
                    Ceq_val[2], Ceq_fit[2] = C[i], fitness[i]
                elif fitness[i] > Ceq_fit[0] and fitness[i] > Ceq_fit[1] and fitness[i] > Ceq_fit[2] \
                        and fitness[i] < Ceq_fit[3]:
                    Ceq_val[3], Ceq_fit[3] = C[i], fitness[i]

            if iter == 0:
                F_old = torch.clone(fitness)
                C_old = torch.clone(C)
            else:
                losers = F_old < fitness
                C[losers], fitness[losers] = C_old[losers], F_old[losers]
                C_old, F_old = C, fitness

            t = (1 - iter / max_iter) ** (a2 * iter / (max_iter - iter))

            Ceq_ave = torch.mean(Ceq_val, dim=0).view(1, dim)
            C_pool = torch.cat((Ceq_val, Ceq_ave))

            for i in range(npop):
                lmb = torch.rand(dim)
                r = torch.rand(dim)

                Ceq = C_pool[torch.randint(Pool_Size + 1, (1, 1))]

                F = a1 * torch.sign(r-0.5) * (torch.exp(-lmb*t)-1)
                GCP = 0.5 * torch.rand(1) * torch.ones(1, dim) * (torch.rand(1) <= GP)
                G0 = GCP * (Ceq - lmb * C[i])
                G = G0 * F

                C[i] = Ceq + (C[i] - Ceq) * F + (G / lmb) * (1 - F)
                C = torch.clamp(C, min=-bounds, max=bounds)

            if verbose and iter % (max_iter/10) == 0:
                print(["At iteration " + str(iter) + " the best fitness is " + str(Ceq_fit[0])])

        self.W_readout = Ceq_val[0].view(train_targets.shape[0], readout.shape[0]+1)
        return True
