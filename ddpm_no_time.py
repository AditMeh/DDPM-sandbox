import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import TensorDataset
from cnn_generator import CNNUnet
from mnist import create_dataloaders


import torchvision.transforms.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


import imageio
import os
import tqdm
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
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "oneover_sqrta": 1/torch.sqrt(alphas),
        "mab_over_sqrtmab": (1-alphas)/sqrt_one_minus_alpha_bar
    }

    return hparams

def generate_img(net, T, img_shape, hparams):
    net.eval()
    with torch.no_grad():
        seed = torch.randn(1, *img_shape).to(device=device)
        for i in range(T, 0, -1):
            z = torch.randn(1, *img_shape).to(device=device)

            term1 = hparams["oneover_sqrta"][i-1]
            term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * net(seed, None))
            term3 = z * hparams["var_t"][i-1]

            seed = term1 * term2 + term3 if i > 1 else term1 * term2

        return seed

blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class HardCodeUNetMNIST(nn.Module):
    def __init__(self, n_channel) -> None:
        super().__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = HardCodeUNetMNIST(1)

    def forward(self, data, pos):
        data = self.unet(data)
        return data

def train(net, train_loader, epochs, T, beta_min, beta_max, img_shape, device):
    hparams = compute_schedule(T, beta_min, beta_max)

    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)

    loss = nn.MSELoss()
    optimizer = Adam(params=net.parameters(), lr=2e-4)

    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        for x, _ in tqdm.tqdm(train_loader):
            net.train()
            x = x.to(device=device)
            ts = torch.randint(1, T + 1, (x.shape[0],)).to(device)

            eps = torch.randn(*x.shape).to(device=device)

            # Forward pass through model
            x_pass = hparams["sqrt_alpha_bar"][ts - 1][..., None, None, None] * x + \
                hparams['sqrt_one_minus_alpha_bar'][ts - 1][..., None, None, None] * eps

            pred = net(x_pass, ((ts/T).unsqueeze(1).float()))

            # print(torch.norm(pred))
            train_loss = loss(pred, eps)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss = {train_loss.item()}')

        sampled_img = generate_img(net, T, img_shape, hparams)
        
        f, ax = plt.subplots(1, 1)

        ax.imshow((torch.squeeze(sampled_img, dim= 0 ).permute(1,2,0)).detach().cpu().numpy(), cmap="gray")
        ax.set_axis_off()
        f.savefig(f'samples/{epoch}.png')
        f.clear()
        f.clf()

    return net


def sample_chain(net, T, beta_min, beta_max, img_shape, device, ):
    hparams = compute_schedule(T, beta_min, beta_max)
    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)
    seed = torch.randn(1, *img_shape).to(device=device)
    chain_samples = [seed]
    for i in range(T, 0, -1):
        z = torch.randn(1, *img_shape).to(device=device)

        term1 = hparams["oneover_sqrta"][i-1]
        term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * net(seed, None))
        term3 = z * hparams["var_t"][i-1]

        seed = term1 * term2 + term3 if i > 1 else term1 * term2
        chain_samples.append(seed)
    return seed, chain_samples



if __name__ == "__main__":
    if not os.path.exists("./folder"):
        os.mkdir("folder/")
    if not os.path.exists("./samples"):
        os.mkdir("./samples/")
    T = 1000
    beta_min, beta_max = 1e-4, 0.02

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    tf = transforms.Compose(
        [transforms.ToTensor()]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    train_ = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    sample = next(iter(train_))[0]    


    # Build a static graph
    net = DiffusionModel().to(device=device)

    train(net, train_, 3, T, beta_min, beta_max,  sample.shape[1:], device)
    _, ret = sample_chain(net, T, beta_min, beta_max, sample.shape[1:], device)
    for j, i in tqdm.tqdm(enumerate(ret)):
        f, ax = plt.subplots(1, 1)

        ax.imshow((torch.squeeze(i, dim = 0).permute(1,2,0)).detach().cpu().numpy(), cmap="gray")
        ax.set_axis_off()
        f.savefig(f'folder/{j}.png')
        f.clear()
        f.clf()

    images = []
    for filename in sorted(os.listdir("./folder/"), key = lambda i: int(i.split(".")[0])):
        images.append(imageio.imread("./folder/" + filename))
    
    # take last 300 images of the markov chain. Append the final generated image 20 times for visual clarity. 
    imageio.mimsave("movie.gif", images[700:] + [images[-1] for _ in range(20)], fps= 40)
    torch.save(net.state_dict(), "model.pt")
