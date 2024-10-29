import torch
from torch import nn
from torch.distributions import kl_divergence, Categorical
from torch.distributions.normal import Normal
from typing import Type, Union


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
        p_params,
        q_params=None,
        forced_latent=None,
        use_mode=False,
        force_constant_output=False,
        analytical_kl=False,
        mode_pred=False,
        use_uncond_mode=False,
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

        # Compute log p(z)
        if mode_pred is False:
            # Summing over all dims but batch
            logprob_p = p.log_prob(z).sum(list(range(1, z.dim())))
        else:
            logprob_p = None

        if q_params is not None:

            # Compute log q(z)
            logprob_q = q.log_prob(z).sum(list(range(1, z.dim())))

            if mode_pred is False:  # if not predicting
                # Compute KL (analytical or MC estimate)
                kl_analytical = kl_divergence(q, p)
                if analytical_kl:
                    kl_elementwise = kl_analytical
                else:
                    kl_elementwise = kl_normal_mc(z, p_params, q_params)
                kl_samplewise = kl_elementwise.sum(list(range(1, z.dim())))

                # Compute spatial KL analytically (but conditioned on samples from
                # previous layers)
                kl_spatial_analytical = kl_analytical.sum(1)
            else:  # if predicting, no need to compute KL
                kl_analytical = None
                kl_elementwise = None
                kl_samplewise = None
                kl_spatial_analytical = None

        else:
            kl_elementwise = kl_samplewise = kl_spatial_analytical = None
            logprob_q = None

        data = {
            "z": z,  # sampled variable at this layer (batch, ch, h, w)
            "p_params": p_params,  # (b, ch, h, w) where b is 1 or batch size
            "q_params": q_params,  # (batch, ch, h, w)
            "logprob_p": logprob_p,  # (batch, )
            "logprob_q": logprob_q,  # (batch, )
            "kl_elementwise": kl_elementwise,  # (batch, ch, h, w)
            "kl_samplewise": kl_samplewise,  # (batch, )
            "kl_spatial": kl_spatial_analytical,  # (batch, h, w)
            "mu": q_mu,
            "logvar": q_lv,
            "pi": None,
        }
        return out, data


class MixtureStochasticConvBlock(nn.Module):
    """
    Stochastic block with GMM for p(z) and q(z), handling both p(z) and q(z) parameters.
    Each component in the mixture has its own set of mu and log-variance.
    """

    def __init__(
        self,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        n_components=4,
        transform_p_params=False,
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.n_components = n_components
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        self.conv_in_q = conv_type(c_in, 2 * c_vars * n_components, kernel, padding=pad)
        self.conv_out = conv_type(c_vars, c_out, kernel, padding=pad)

        # Define mixture coefficients for p and q as learnable 1D tensors
        self.p_pi = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.q_pi = nn.Parameter(torch.randn(n_components), requires_grad=True)

    def forward(
        self,
        p_params,
        q_params=None,
        forced_latent=None,
        use_mode=False,
        force_constant_output=False,
        analytical_kl=False,
        mode_pred=False,
        use_uncond_mode=False,
    ):

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)

        p_pi = torch.softmax(
            self.p_pi, dim=1
        )  # Get the mixture probabilities for each component

        # Separate mu and logvar for each component
        p_mu, p_lv = torch.chunk(p_params, 2, dim=1)

        p_std = (p_lv / 2).exp()

        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)

        p_components = []

        for mu_chunk, std_chunk in zip(p_mu_chunks, p_std_chunks):
            p_components.append(
                Normal(mu_chunk, std_chunk)
            )  # Create Gaussian components for p

        if q_params is not None:
            q_params = self.conv_in_q(q_params)
            q_pi, q_mu_lv = torch.split(
                q_params,
                [self.c_vars * self.n_components, 2 * self.c_vars * self.n_components],
                dim=1,
            )
            q_pi = torch.softmax(q_pi, dim=1)  # Mixture probabilities for q
            q_pi = q_pi.view(
                q_pi.size(0), self.n_components, self.c_vars, *q_pi.shape[2:]
            )
            q_pi = q_pi.permute(0, *range(2, q_pi.ndim), 1)

            q_mu, q_lv = torch.chunk(q_mu_lv, 2, dim=1)

            q_std = (q_lv / 2).exp()

            q_mu_chunks = q_mu.chunk(self.n_components, dim=1)
            q_std_chunks = q_std.chunk(self.n_components, dim=1)

            q_components = []

            for mu_chunk, std_chunk in zip(q_mu_chunks, q_std_chunks):
                q_components.append(
                    Normal(mu_chunk, std_chunk)
                )  # Create Gaussian components for q

            sampling_distrib = q_components
        else:
            sampling_distrib = p_components

        # Sample the mixture component
        component_distribution = (
            Categorical(p_pi) if q_params is None else Categorical(q_pi)
        )
        selected_component = component_distribution.sample()

        z_samples = []
        for i, component in enumerate(sampling_distrib):
            mask = (selected_component == i).float().unsqueeze(1)
            z_samples.append(component.sample() * mask)

        z = torch.sum(torch.stack(z_samples), dim=0)

        # Get the output from the latent variable
        out = self.conv_out(z)

        # Compute log p(z) and log q(z)
        log_probs_p = torch.stack([component.log_prob(z) for component in p_components])
        weighted_log_probs = log_probs_p + torch.log(p_pi.unsqueeze(-1))
        log_prob_p_z = torch.logsumexp(weighted_log_probs, dim=0)

        log_probs_q = torch.stack([component.log_prob(z) for component in q_components])
        weighted_log_probs = log_probs_q + torch.log(q_pi.unsqueeze(-1))
        log_prob_q_z = torch.logsumexp(weighted_log_probs, dim=0)

        # Compute the Wasserstein distance
        wasserstein_dist = wasserstein_distance_gmm(
            p_components, q_components, p_pi, q_pi
        )

        data = {
            "z": z,  # sampled latent variable
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": log_prob_p_z,
            "logprob_q": log_prob_q_z,
            "wasserstein_distance": wasserstein_dist,
            "mu": q_mu if q_params is not None else p_mu,
            "logvar": q_lv if q_params is not None else p_lv,
            "pi": q_pi if q_params is not None else p_pi,  # mixture coefficients
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


def wasserstein_distance_gmm(p_components, q_components, p_pi, q_pi):
    # Compute pairwise Wasserstein distances between components
    num_components_p = len(p_components)
    num_components_q = len(q_components)
    pairwise_distances = torch.zeros(num_components_p, num_components_q)

    for i, p_comp in enumerate(p_components):
        for j, q_comp in enumerate(q_components):
            mean_diff = p_comp.mean - q_comp.mean
            cov_p = p_comp.covariance_matrix
            cov_q = q_comp.covariance_matrix
            cov_mean = 0.5 * (cov_p + cov_q)
            mean_term = torch.dot(mean_diff, mean_diff)
            cov_term = torch.trace(cov_p + cov_q - 2 * torch.sqrt(cov_mean))
            pairwise_distances[i, j] = mean_term + cov_term

    # Compute the Wasserstein distance between the GMMs
    wasserstein_distance = torch.sum(
        p_pi.unsqueeze(1) * q_pi.unsqueeze(0) * pairwise_distances
    )
    return wasserstein_distance
