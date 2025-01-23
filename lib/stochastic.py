import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from typing import Type, Union
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StochasticConvBlock(nn.Module):
    """
    Stochastic Conv Block to handle both normal and mixture models,
    for both conditional and unconditional cases.
    This also can replace transformer blocks in the model (only in the topmost level).
    """

    def __init__(
        self,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        block_type="normal",
        n_components=1,
        top_layer=False,
        conditional=False,
        condition_type="mlp",
        labeled_ratio=1.0,
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self.block_type = block_type
        self.n_components = n_components
        self.top_layer = top_layer
        self.conditional = conditional
        self.condition_type = condition_type
        self.labeled_ratio = labeled_ratio
        self.temperature = 1.0
        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        if not top_layer or (block_type == "normal" and not conditional):
            self.conv_in_q = conv_type(c_in, 2 * c_vars, kernel, padding=pad)
        elif conditional:
            if condition_type == "mlp":
                self.qy_x = nn.Sequential(
                    conv_type(c_in, c_vars, kernel, padding=pad),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(c_vars * 8 * 8, n_components),
                )
                self.qz_xy = nn.Sequential(
                    conv_type(c_in, 2 * c_vars, kernel, padding=pad),
                    nn.ReLU(),
                    conv_type(2 * c_vars, 2 * c_vars, kernel, padding=pad),
                )
            elif condition_type == "transformer":
                self.qy_x = TransformerQ(
                    c_in=c_in,
                    embed_dim=128,
                    n_components=n_components,
                    num_heads=4,
                    num_layers=3,
                    mode="mlp",
                )
                self.qz_xy = TransformerQ(
                    c_in=c_in, embed_dim=128, num_heads=4, num_layers=6, mode="conv"
                )
        else:  # Top layer, mixture, unconditional
            self.conv_in_q = conv_type(
                c_in, 2 * c_vars * n_components, kernel, padding=pad
            )
        self.conv_out = conv_type(c_vars, c_out, kernel, padding=pad)

    def forward(self, label, p_params, q_params):

        p_mu, p_lv = torch.chunk(p_params, 2, dim=1)
        p_mu = torch.clamp(p_mu, min=-10.0, max=10.0)  # Clamp p_mu
        p_lv = torch.clamp(p_lv, min=-10.0, max=10.0)  # Clamp p_lv
        p_std = torch.where(p_lv < 0, (p_lv / 2).exp(), 1 + p_lv)

        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)

        p_components = []

        for mu_chunk, std_chunk in zip(p_mu_chunks, p_std_chunks):
            p_components.append(Normal(mu_chunk, std_chunk))

        if not self.top_layer or (self.block_type == "normal" and not self.conditional):
            # Define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_lv = q_params.chunk(2, dim=1)
            q_mu = torch.clamp(q_mu, min=-10.0, max=10.0)
            q_lv = torch.clamp(q_lv, min=-10.0, max=10.0)
            q_std = torch.where(q_lv < 0, (q_lv / 2).exp(), 1 + q_lv)
            q = Normal(q_mu, q_std)

            z = q.rsample()
            out = self.conv_out(z)
            kl = self._compute_kl(q, p_components)
            logprob_p = self._compute_logprob(p_components, z)
            logprob_q = self._compute_logprob(q, z)
        else:  # Top layer
            if self.conditional:
                qy_logits = self.qy_x(q_params)
                # FiLM layer
                gamma = self.gamma_layer(qy_logits)
                beta = self.beta_layer(qy_logits)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                x_modulated = gamma * q_params + beta
                qz_params = self.qz_xy(x_modulated)
                q_mu, q_lv = torch.chunk(qz_params, 2, dim=1)
                q_mu = torch.clamp(q_mu, min=-10.0, max=10.0)
                q_lv = torch.clamp(q_lv, min=-10.0, max=10.0)
                q_std = torch.where(q_lv < 0, (q_lv / 2).exp(), 1 + q_lv)
                q = Normal(q_mu, q_std)
                z = q.rsample()
                
                y = F.gumbel_softmax(qy_logits, tau=self.temperature, hard=False)
                self._update_temperature(js_div)
                y_pred = y.argmax(dim=1)
                
                js_div = self._compute_js_div(y)
                kl = self._compute_kl(q, p_components, label, y_pred)
                entropy = self._compute_entropy(y)
                cross_entropy = self._compute_cross_entropy(qy_logits, label)
                logprob_p = self._compute_logprob(p_components, z)
                logprob_q = self._compute_logprob(q, z)

        data = {
            "z": z,
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": logprob_p,
            "logprob_q": logprob_q,
            "kl": kl + js_div,
            "mu": q_mu,
            "lv": q_lv,
            "pi": y,
            "cross_entropy": cross_entropy,
            "entropy": entropy,
        }

        return out, data


class TransformerQ(nn.Module):
    def __init__(
        self, c_in, embed_dim, n_components=1, num_heads=4, num_layers=2, mode="mlp"
    ):
        super().__init__()
        # Embedding layer for channel tokens
        self.embedding = nn.Linear(c_in, embed_dim)

        # Transformer Encoder
        # embed_dim = num_heads * head_dim
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        if mode == "mlp":
            # Output layer to predict logits for each token
            self.output = nn.Linear(embed_dim, n_components)
        else:  # mode == "conv"
            self.output = nn.Conv2d(embed_dim, c_in, kernel_size=1)

        self.mode = mode

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.flatten(2)  # Combine H and W into one dimension
        x = x.permute(2, 0, 1)
        x = self.embedding(x)
        x = self.transformer(x)
        # [seq_len, B, embed_dim] -> [B, embed_dim]

        if self.mode == "mlp":
            x = x.mean(dim=0)
        else:
            x = x.permute(1, 2, 0).view(B, -1, H, W)
        return self.output(x)


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
