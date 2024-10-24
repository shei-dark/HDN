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

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            # TODO better assertion logic
            assert max(p_params.shape) == 2 * self.c_vars

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
        num_components=4,
        transform_p_params=True,
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.num_components = num_components
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self.transform_p_params = transform_p_params

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        # Transform p_params to get pi logits, mu, and logvar for each component
        if transform_p_params:
            self.conv_in_p = conv_type(
                c_in, 2 * c_vars * num_components, kernel, padding=pad
            )
        self.conv_in_q = conv_type(
            c_in, 2 * c_vars * num_components, kernel, padding=pad
        )
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

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)

        # Split p_params and q_params into pi, mu, and logvar for each component
        p_pi, p_mu_lv = torch.chunk(
            p_params, self.num_components + 2 * self.c_vars * self.num_components, dim=1
        )
        p_pi = torch.softmax(
            p_pi, dim=1
        )  # Get the mixture probabilities for each component

        # Separate mu and logvar for each component
        p_mu, p_lv = torch.chunk(p_mu_lv, 2, dim=1)
        p_mu = p_mu.view(
            p_mu.size(0), self.num_components, self.c_vars, *p_mu.shape[2:]
        )
        p_lv = p_lv.view(
            p_lv.size(0), self.num_components, self.c_vars, *p_lv.shape[2:]
        )
        p_std = (p_lv / 2).exp()

        p_components = Normal(p_mu, p_std)  # Create Gaussian components

        if q_params is not None:
            q_params = self.conv_in_q(q_params)
            q_pi, q_mu_lv = torch.chunk(
                q_params,
                self.num_components + 2 * self.c_vars * self.num_components,
                dim=1,
            )
            q_pi = torch.softmax(q_pi, dim=1)  # Mixture probabilities for q

            q_mu, q_lv = torch.chunk(q_mu_lv, 2, dim=1)
            q_mu = q_mu.view(
                q_mu.size(0), self.num_components, self.c_vars, *q_mu.shape[2:]
            )
            q_lv = q_lv.view(
                q_lv.size(0), self.num_components, self.c_vars, *q_lv.shape[2:]
            )
            q_std = (q_lv / 2).exp()

            q_components = Normal(q_mu, q_std)  # Gaussian components for q
            sampling_distrib = q_components
        else:
            sampling_distrib = p_components

        # Sample from the mixture
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean  # If using mode, take the mean
            else:
                z = sampling_distrib.sample()  # Sample from q(z) or p(z)
        else:
            z = forced_latent

        # Sample the mixture component
        component_distribution = (
            Categorical(p_pi) if q_params is None else Categorical(q_pi)
        )
        selected_component = component_distribution.sample()
        z_selected = z.gather(1, selected_component.unsqueeze(-1).expand_as(z))

        # Get the output from the latent variable
        out = self.conv_out(z_selected)

        # Compute log p(z) and log q(z)
        logprob_p = (
            p_components.log_prob(z_selected).sum(list(range(1, z_selected.dim())))
            if mode_pred is False
            else None
        )
        logprob_q = (
            q_components.log_prob(z_selected).sum(list(range(1, z_selected.dim())))
            if q_params is not None
            else None
        )

        # Compute KL divergence
        if q_params is not None and mode_pred is False:
            kl_analytical = kl_divergence(q_components, p_components)
            kl_samplewise = kl_analytical.sum(list(range(1, z.dim())))
            kl_spatial_analytical = kl_analytical.sum(1)
        else:
            kl_samplewise = kl_spatial_analytical = None

        data = {
            "z": z_selected,  # sampled latent variable
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": logprob_p,
            "logprob_q": logprob_q,
            "kl_elementwise": kl_samplewise,
            "kl_samplewise": kl_samplewise,
            "kl_spatial": kl_spatial_analytical,
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
