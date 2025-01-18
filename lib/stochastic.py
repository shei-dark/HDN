import torch
from torch import nn
from torch.distributions import kl_divergence, Categorical, MultivariateNormal
from torch.distributions.normal import Normal
from typing import Type, Union
import torch.nn.functional as F


class NormalStochasticConvBlock(nn.Module):
    """
    Transform input parameters to q(z) with a convolution, optionally do the
    same for p(z), then sample z ~ q(z) and return conv(z).

    If q's parameters are not given, do the same but sample from p(z).
    """

    def __init__(
        self, c_in, c_vars, c_out, conv_mult, kernel=3, transform_p_params=True
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        if transform_p_params:
            self.conv_in_p = conv_type(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = conv_type(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = conv_type(c_vars, c_out, kernel, padding=pad)

    def forward(
        self,
        label,
        p_params,
        q_params=None,
        forced_latent=None,
        use_mode=False,
        force_constant_output=False,
        analytical_kl=False,
        mode_pred=False,
        use_uncond_mode=False,
        epoch=0,
    ):

        # assert (forced_latent is None) or (not use_mode)

        # if self.transform_p_params:
        #     p_params = self.conv_in_p(p_params)
        # else:
        #     # TODO better assertion logic
        #     assert max(p_params.shape) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        p_mu = torch.clamp(p_mu, min=-10.0, max=10.0)  # Clamp p_mu
        p_lv = torch.clamp(p_lv, min=-10.0, max=10.0)  # Clamp p_lv
        p = Normal(p_mu, (p_lv / 2).exp())

        if q_params is not None:
            # Define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_lv = q_params.chunk(2, dim=1)
            q_mu = torch.clamp(q_mu, min=-10.0, max=10.0)  # Clamp q_mu
            q_lv = torch.clamp(q_lv, min=-10.0, max=10.0)  # Clamp q_lv
            q = Normal(q_mu, (q_lv / 2).exp())
            # Sample from q(z)
            sampling_distrib = q
        else:
            # Sample from p(z)
            sampling_distrib = p

        # Generate latent variable (typically by sampling)
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                if mode_pred:
                    if use_uncond_mode:
                        z = sampling_distrib.mean
                    #                         z = sampling_distrib.rsample()
                    else:
                        z = sampling_distrib.rsample()
                else:
                    z = sampling_distrib.rsample()
        else:
            z = forced_latent

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = p_params[0:1].expand_as(p_params).clone()

        # Output of stochastic layer
        out = self.conv_out(z)

        logprob_p = None
        logprob_q = None
        kl_analytical = None

        # Compute log p(z)
        if mode_pred is False:
            # Summing over all dims but batch
            logprob_p = p.log_prob(z).sum(list(range(1, z.dim())))

        if q_params is not None:
            # Compute log q(z)
            logprob_q = q.log_prob(z).sum(list(range(1, z.dim())))

            if mode_pred is False:  # if not predicting
                # Compute KL (analytical or MC estimate)
                kl_analytical = kl_divergence(q, p)
                kl_analytical = kl_analytical.sum(
                    list(range(1, kl_analytical.dim()))
                ).mean()

        data = {
            "z": z,  # sampled variable at this layer (batch, ch, h, w)
            "p_params": p_params,  # (b, ch, h, w) where b is 1 or batch size
            "q_params": q_params,  # (batch, ch, h, w)
            "logprob_p": logprob_p,  # (batch, )
            "logprob_q": logprob_q,  # (batch, )
            "kl": kl_analytical,  # (batch, )
            "repulsive": None,
            "mu": q_mu,
            "logvar": q_lv,
            "pi": None,
            "cross_entropy": None,
            "temperature": 0,
            "entropy": 0,
        }
        return out, data


class MixtureStochasticConvBlock(nn.Module):
    """
    Stochastic block with GMM for p(z) and q(z), handling both p(z) and q(z) parameters.
    Each component in the mixture has its own set of mu and log-variance.
    Gumbel-Softmax is used to approximate the categorical distribution.
    """

    def __init__(
        self,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        n_components=4,
        labeled_ratio=1,
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.n_components = n_components
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self.temperature = 1.0
        self.labeled_ratio = labeled_ratio
        self.prior_probs = torch.tensor([0.58, 0.13, 0.22, 0.07]).cuda()
        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        # q(y|x): Outputs logits for the categorical distribution
        self.qy_x = nn.Sequential(
            conv_type(c_in, c_vars, kernel, padding=pad),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c_vars * 8 * 8, n_components),
        )

        # q(z|x, y): Outputs parameters (mu, logvar) for the Gaussian distribution
        self.qz_xy = nn.Sequential(
            conv_type(c_in, 2 * c_vars, kernel, padding=pad),
            nn.ReLU(),
            conv_type(2 * c_vars, 2 * c_vars, kernel, padding=pad),
        )
        # Feature Modulation (FiLM Layer)
        # learning parameters to scale and shift the feature map based on the component mode vector.
        # Linear layers to compute gamma and beta from the component mode vector
        self.gamma_layer = nn.Linear(4, c_in)
        self.beta_layer = nn.Linear(4, c_in)

        self.conv_out = conv_type(c_vars, c_out, kernel, padding=pad)

    def forward(
        self,
        label,
        p_params,
        q_params=None,
        forced_latent=None,
        use_mode=False,
        force_constant_output=False,
        analytical_kl=False,
        mode_pred=False,
        use_uncond_mode=False,
        hard=True,  # Use hard Gumbel-Softmax
    ):

        assert (forced_latent is None) or (not use_mode)

        # Separate mu and logvar for each component of the gmm prior
        p_mu, p_lv = torch.chunk(p_params, 2, dim=1)
        p_mu = torch.clamp(p_mu, min=-10.0, max=10.0)  # Clamp p_mu
        p_lv = torch.clamp(p_lv, min=-10.0, max=10.0)  # Clamp p_lv
        p_std = torch.where(p_lv < 0, (p_lv / 2).exp(), 1 + p_lv)

        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)

        p_components = []

        for mu_chunk, std_chunk in zip(p_mu_chunks, p_std_chunks):
            p_components.append(
                Normal(mu_chunk, std_chunk)
            )  # Create Gaussian components for p

        batch_size = q_params.size(0)
        small_batch_size = int(batch_size * self.labeled_ratio)
        qy_logits = self.qy_x(q_params)

        if label is not None:
            label = label.long()
            label = label[:small_batch_size]
            supervised_loss = torch.nn.functional.cross_entropy(
                qy_logits[:small_batch_size], label
            )
        else:
            supervised_loss = 0
        # y = F.softmax(qy_logits, dim=-1)

        # Step 2: Compute q(z|x, y)
        gamma = self.gamma_layer(qy_logits)  # Shape: (batch_size, c_in)
        beta = self.beta_layer(qy_logits)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, c_in, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x_modulated = gamma * q_params + beta
        qz_params = self.qz_xy(x_modulated)
        q_mu, q_lv = torch.chunk(qz_params, 2, dim=1)
        q_mu = torch.clamp(q_mu, min=-10.0, max=10.0)  # Clamp q_mu
        q_lv = torch.clamp(q_lv, min=-10.0, max=10.0)  # Clamp q_lv
        q_std = torch.where(q_lv < 0, (q_lv / 2).exp(), 1 + q_lv)

        y = torch.nn.functional.gumbel_softmax(
            qy_logits, tau=self.temperature, hard=False
        )

        m = 0.5 * (y + self.prior_probs)
        js_div = 0.5 * torch.sum(
            y * torch.log(y / (m + 1e-10)), dim=-1
        ) + 0.5 * torch.sum(self.prior_probs * torch.log(self.prior_probs / (m + 1e-10)), dim=-1)

        self.temperature = max(0.5, self.temperature * 0.999)

        y_pred = y.argmax(dim=-1)
        if small_batch_size < batch_size:
            entropy = -torch.mean(
                torch.sum(
                    y[small_batch_size:] * torch.log(y[small_batch_size:] + 1e-10),
                    dim=-1,
                )
            )
        else:
            entropy = 0
        z = q_mu + q_std * torch.randn_like(q_std)

        out = self.conv_out(z)

        logprob_p = None
        logprob_q = None
        kl_divergences = []

        # Compute logprob_p
        log_probs_p = torch.stack(
            [comp.log_prob(z) for comp in p_components], dim=-1
        )  # Shape: [batch_size, 32, 8, 8, 4]
        logprob_p = torch.sum(
            log_probs_p * y.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=-1
        )  # Weighted sum

        # Compute logprob_q
        q_distribution = Normal(q_mu, q_std)
        logprob_q = q_distribution.log_prob(z)  # Shape: [512, 32, 8, 8]

        for i in range(len(p_components)):
            # Compute KL divergence between q and each component in p_components
            kl_i = kl_divergence(Normal(q_mu, q_std), p_components[i]).mean(
                dim=(1, 2, 3)
            )
            kl_divergences.append(kl_i)
        # Stack KL divergences for all components (Shape: [batch_size, n_components])
        kl_divergences = torch.stack(kl_divergences, dim=-1)

        # # Separate cases where label matches y_pred and where it doesn't
        # matching_mask = (y_pred == label).unsqueeze(-1)  # Shape: [batch_size, 1]

        if label is None:
            kl_loss = 0
        else:
            if small_batch_size < batch_size:
                kl = torch.cat(
                    [
                        kl_divergences[range(small_batch_size), label],
                        kl_divergences[
                            range(small_batch_size, batch_size),
                            y_pred[small_batch_size:],
                        ],
                    ],
                    dim=0,
                )
            else:
                kl = kl_divergences[range(batch_size), label]

            kl_loss = kl.mean() + js_div.mean()

        data = {
            "z": z,  # sampled latent variable
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": logprob_p,
            "logprob_q": logprob_q,
            "kl": kl_loss,
            "repulsive": 0,
            "mu": q_mu,
            "logvar": q_lv,
            "pi": y,  # mixture coefficients
            "cross_entropy": (
                supervised_loss * (1 / self.labeled_ratio)
                if self.labeled_ratio > 0
                else 0
            ),
            "entropy": entropy,
        }

        return out, data


def kl_normal_mc(z, p_mulv, q_mulv):
    """
    One-sample estimation of element-wise KL between two diagonal
    multivariate normal distributions. Any number of dimensions,
    broadcasting supported (be careful).

    :param z:
    :param p_mulv:
    :param q_mulv:
    :return:
    """
    p_mu, p_lv = torch.chunk(p_mulv, 2, dim=1)
    q_mu, q_lv = torch.chunk(q_mulv, 2, dim=1)
    p_std = (p_lv / 2).exp()
    q_std = (q_lv / 2).exp()
    p_distrib = Normal(p_mu, p_std)
    q_distrib = Normal(q_mu, q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)
