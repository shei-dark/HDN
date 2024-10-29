import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, kl_divergence


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_components):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=4, stride=2, padding=1
        )  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 8x8

        # Fully connected layers for mu and logvar for each mixture component
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim * num_components)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim * num_components)
        self.num_components = num_components
        self.latent_dim = latent_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the feature map

        # Output mean and log variance for each component
        mu = self.fc_mu(x).view(-1, self.num_components, self.latent_dim)
        logvar = self.fc_logvar(x).view(-1, self.num_components, self.latent_dim)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)

        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )  # 32x32 -> 64x64

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 128, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Sigmoid activation for binary output
        return x


class ConvMVAE(nn.Module):
    def __init__(self, latent_dim, num_components):
        super(ConvMVAE, self).__init__()
        self.encoder = ConvEncoder(latent_dim, num_components)
        self.decoder = ConvDecoder(latent_dim)
        self.num_components = num_components
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        mu, logvar = self.encoder(x)

        # Mixture Components
        component_weights = torch.ones(self.num_components) / self.num_components
        component_dist = Categorical(component_weights)
        component_idx = component_dist.sample((x.size(0),))

        # Select mu and logvar for the chosen component
        mu = mu[torch.arange(x.size(0)), component_idx]
        logvar = logvar[torch.arange(x.size(0)), component_idx]

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar, component_idx


def loss_function(x, x_reconstructed, mu, logvar, component_idx, num_components):
    # Reconstruction loss (Binary Cross-Entropy)
    recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction="sum")

    # KL Divergence between posterior and mixture of Gaussians prior
    kl_loss = 0
    for i in range(num_components):
        prior = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        posterior = Normal(mu, torch.exp(0.5 * logvar))
        kl_loss += kl_divergence(posterior, prior).sum()

    return recon_loss + kl_loss
