import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import TensorDataset


import imageio
import os
# Utility functions


def compute_schedule(T, beta_min, beta_max):
    betas = torch.linspace(beta_min, beta_max, steps=T)
    alphas = 1 - betas

    var_t = torch.sqrt(betas)
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    hparams = {
        "var_t": var_t,
        "alphas": alphas,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar
    }

    return hparams

# Courtesty of chatgpt


def sample_circle_points(x0, y0, r, n):
    theta = torch.rand(n) * 2 * torch.pi
    x = x0 + r * torch.cos(theta)
    y = y0 + r * torch.sin(theta)
    points = torch.stack((x, y), dim=1)
    return points


class Block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, data):
        pre_act = self.fc(data)
        post_act = nn.GELU()(pre_act)
        return data + post_act


class DiffusionModel(nn.Module):
    def __init__(self, freq_num, embed_dim, hidden_layers):
        super().__init__()
        self.input_project = nn.Linear(2, embed_dim)

        self.freq_num = freq_num

        self.hiddens = nn.Sequential(
            *[Block(embed_dim + 2*freq_num + 1) for i in range(hidden_layers)])
        self.output_project = nn.Linear(embed_dim + 2*freq_num + 1, 2)

    def forward(self, data, pos):

        data = self.input_project(data)
        data = nn.GELU()(data)

        posenc = self.positional_encoding(pos)
        # print(posenc.shape, data.shape)
        data = torch.cat((data, posenc), dim=-1)
        data = self.hiddens(data)
        data = self.output_project(data)

        return data

    def positional_encoding(self, position):
        terms = [position]
        for i in range(self.freq_num):
            sin_encoding = torch.sin(2 ** i * torch.pi * position)
            cos_encoding = torch.cos(2 ** i * torch.pi * position)
            terms.append(sin_encoding)
            terms.append(cos_encoding)

        return torch.concat(terms, dim=1)


# Create the dataset

# Courtesty of chatgpt
def sample_circle_points(x0, y0, r, n):
    # Generate random angles between 0 and 2*pi
    theta = torch.rand(n) * 2 * torch.pi
    # Calculate the x and y coordinates of the points on the circle
    x = x0 + r * torch.cos(theta)
    y = y0 + r * torch.sin(theta)
    # Combine the x and y coordinates into a tensor of points
    points = torch.stack((x, y), dim=1)
    return points


def gen_blobs():
    n = 8000
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3

    points = X.astype(np.float32)
    return points


def train(net, points, epochs, batch_size, T, beta_min, beta_max):
    hparams = compute_schedule(T, beta_min, beta_max)
    unif = torch.arange(1, T+1)

    dataloader = torch.utils.data.DataLoader(TensorDataset(
        torch.from_numpy(points)), shuffle=True, batch_size=32)

    loss = nn.MSELoss()
    optimizer = Adam(params=net.parameters(), lr=1e-4)

    for i in range(1, epochs + 1):
        for sample in dataloader:
            sample = sample[0]
            idxs = torch.randint(len(unif), size=(batch_size, ))
            ts = unif[idxs]

            eps = torch.randn(batch_size, 2)

            # Forward pass through model
            x_pass = hparams["sqrt_alpha_bar"][idxs][..., None] * sample + \
                hparams['sqrt_one_minus_alpha_bar'][idxs][..., None] * eps

            pred = net(x_pass, ts.unsqueeze(1).float())

            # print(torch.norm(pred))
            train_loss = loss(pred, eps)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        print(f'epoch: {i}, loss = {train_loss.item()}')
    return net


def sample(net, T, beta_min, beta_max):
    hparams = compute_schedule(T, beta_min, beta_max)
    seed = torch.randn(1000, 2)
    catalysm = [seed]
    for i in range(T, 0, -1):
        z = torch.randn(1000, 2)
        ts = torch.tensor(i, dtype=torch.int64).repeat(1000)

        term1 = (1/torch.sqrt(hparams["alphas"]))[ts-1][..., None]
        term2 = seed - (((1-hparams["alphas"][ts-1]) * (1/hparams["sqrt_one_minus_alpha_bar"])[
                        ts-1])[..., None] * net(seed, ts.unsqueeze(1).float()))
        term3 = z * hparams["var_t"][ts-1][..., None]

        seed = term1 * term2 + term3
        catalysm.append(seed)
    return seed, catalysm



if __name__ == "__main__":
    # T = 128
    # beta_min, beta_max = 0.0001, 0.02

    # points = gen_blobs()
    # net = DiffusionModel(10, 128, 5)
    # train(net, points, 400, 32, T, beta_min, beta_max)
    # _, ret = sample(net, T, beta_min, beta_max)
    # for j, i in enumerate(ret):
    #     f, ax = plt.subplots(1, 1)

    #     ax.scatter(i.detach().numpy()[:, 0], i.detach().numpy()[:, 1])
    #     ax.set_axis_off()
    #     f.savefig(f'folder/{j}.png')
    #     f.clear()
    #     f.clf()

    images = []
    for filename in sorted(os.listdir("./folder/"), key = lambda i: int(i.split(".")[0])):
        images.append(imageio.imread("./folder/" + filename))
    imageio.mimsave("movie.gif", images, fps= 3)
