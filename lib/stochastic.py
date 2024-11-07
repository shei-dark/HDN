import torch
from torch import nn
from torch.distributions import kl_divergence, Categorical, MultivariateNormal
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
        label,
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
        # transform_p_params=False,
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.n_components = n_components
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        # self.transform_p_params = transform_p_params

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        # if transform_p_params:
            # self.conv_in_p = conv_type(c_in, 2 * c_vars * n_components, kernel, padding=pad)
        self.conv_in_q = conv_type(c_in, 2 * c_vars * n_components, kernel, padding=pad)
        self.conv_out = conv_type(c_vars, c_out, kernel, padding=pad)

        # Define mixture coefficients for p and q as learnable 1D tensors
        self.p_pi = nn.Parameter(torch.rand(n_components) * 0.5, requires_grad=True)


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
    ):

        assert (forced_latent is None) or (not use_mode)

        p_pi = torch.softmax(torch.clamp(self.p_pi, min=-10, max=10), dim=0)

        # Separate mu and logvar for each component
        # if self.transform_p_params:
        #     p_params = self.conv_in_p(p_params)
        p_mu, p_lv = torch.chunk(p_params, 2, dim=1)
        p_mu = torch.clamp(p_mu, min=-10.0, max=10.0)  # Clamp p_mu
        p_lv = torch.clamp(p_lv, min=-10.0, max=10.0)  # Clamp p_lv
        p_std = (p_lv / 2).exp()

        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)

        p_components = []

        for mu_chunk, std_chunk in zip(p_mu_chunks, p_std_chunks):
            p_components.append(
                Normal(mu_chunk, std_chunk)
            )  # Create Gaussian components for p
        repulsive = 0
        if q_params is not None:
            q_params = self.conv_in_q(q_params)

            q_mu, q_lv = torch.chunk(q_params, 2, dim=1)
            q_mu = torch.clamp(q_mu, min=-10.0, max=10.0)  # Clamp q_mu
            q_lv = torch.clamp(q_lv, min=-10.0, max=10.0)  # Clamp q_lv
            q_std = (q_lv / 2).exp()

            q_mu_chunks = q_mu.chunk(self.n_components, dim=1)
            q_std_chunks = q_std.chunk(self.n_components, dim=1)

            mus_avg = torch.stack([mu.mean(dim=(1, 2, 3)) for mu in q_mu_chunks])
            dist_matrix = torch.cdist(mus_avg, mus_avg, p=2)
            mask = torch.ones_like(dist_matrix) - torch.eye(dist_matrix.size(0)).to(dist_matrix.device)
            repulsive = (1 / (dist_matrix + 1e-5)) * mask

            q_components = []

            for mu_chunk, std_chunk in zip(q_mu_chunks, q_std_chunks):
                q_components.append(
                    Normal(mu_chunk, std_chunk)
                )  # Create Gaussian components for q

            sampling_distrib = q_components
        else:
            sampling_distrib = p_components

        batch_size = q_params.size(0) if q_params is not None else 1
        
        if label is not None:
            z_samples = []
            for i, component in enumerate(q_components):
                # Create a mask based on the label to select the correct component
                mask = (label == i).float().view(batch_size, *[1] * (q_mu.ndim - 1))
                mask = mask.to(q_mu.device)
                z_samples.append(component.sample() * mask)
            z = torch.sum(torch.stack(z_samples), dim=0)
        else:
            print("Label is None")
            # Sample the mixture component
            component_distribution = Categorical(p_pi)
            # Adjust the sampling based on q_params or p_params
            selected_component = component_distribution.sample(
                (batch_size,)
            )  # Sample a component for each batch entry

            # Create z samples based on selected components
            z_samples = []
            for i, component in enumerate(sampling_distrib):
                # Reshape mask to match component's dimensions
                mask = (
                    (selected_component == i)
                    .float()
                    .view(batch_size, *[1] * (p_mu.ndim - 1))
                )
                z_samples.append(component.sample() * mask)

            # Combine samples from all components based on selection
            z = torch.sum(torch.stack(z_samples), dim=0)

        # Get the output from the latent variable
        out = self.conv_out(z)

        # Compute log p(z) and log q(z)
        log_probs_p = torch.stack([component.log_prob(z) for component in p_components])
        weighted_log_probs_p = log_probs_p + torch.log(p_pi).view(
            -1, *[1] * (log_probs_p.dim() - 1)
        )
        log_prob_p_z = torch.logsumexp(weighted_log_probs_p, dim=0)

        if q_params is not None:
            log_probs_q = torch.stack(
                [component.log_prob(z) for component in q_components]
            )
            weighted_log_probs_q = log_probs_q + torch.log(p_pi).view(
                -1, *[1] * (log_probs_p.dim() - 1)
            )
            log_prob_q_z = torch.logsumexp(weighted_log_probs_q, dim=0)
        else:
            log_prob_q_z = None

        kl_analytical = None

        # Compute KL divergence
        if q_params is not None and mode_pred is False:
            for i, (p_component, q_component) in enumerate(
                zip(p_components, q_components)
            ):
                current_kl = kl_divergence(q_component, p_component) * p_pi[i]
                if kl_analytical is None:
                    kl_analytical = torch.zeros_like(current_kl)
                kl_analytical += current_kl
        if kl_analytical is not None:
            kl_analytical = kl_analytical.sum(
                dim=tuple(range(1, kl_analytical.dim()))
            ).mean()

        data = {
            "z": z,  # sampled latent variable
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": log_prob_p_z,
            "logprob_q": log_prob_q_z,
            "kl": kl_analytical,
            "repulsive": repulsive.sum(),
            "mu": q_mu if q_params is not None else p_mu,
            "logvar": q_lv if q_params is not None else p_lv,
            "pi": p_pi, # mixture coefficients
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
