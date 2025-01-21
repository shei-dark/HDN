import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Type, Union


def no_cp(func, inp):
    return func(inp)


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.

    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(
        self,
        channels,
        conv_mult,
        nonlin,
        kernel=None,
        groups=1,
        batchnorm=True,
        block_type=None,
        dropout=None,
        gated=None,
        grad_checkpoint=False,
        r2_act=None,
    ):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        pad = [k // 2 for k in kernel]
        dropout = dropout if not grad_checkpoint else None
        self.cp = checkpoint if grad_checkpoint else no_cp
        # TODO Might need to update batchnorm stats calculation for grad checkpointing

        if conv_mult == 0:
            # Orientation-invariant case
            assert r2_act is not None, "r2_act must be provided when conv_mult == 0"
            from e2cnn.nn import R2Conv, FieldType

            # Define field types for equivariant convolutions
            self.input_type = FieldType(r2_act, [r2_act.regular_repr] * channels)
            self.output_type = FieldType(r2_act, [r2_act.regular_repr] * channels)

            conv_layer = lambda c_in, c_out, kernel_size, padding: R2Conv(
                FieldType(r2_act, [r2_act.regular_repr] * c_in),
                FieldType(r2_act, [r2_act.regular_repr] * c_out),
                kernel_size=kernel_size,
                padding=padding
            )
            batchnorm_layer = None  # No batchnorm for equivariant convolutions
            dropout_layer = None if dropout is None else nn.Dropout(dropout)
        else:
            conv_layer: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")
            batchnorm_layer_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = getattr(
                nn, f"BatchNorm{conv_mult}d"
            )
            dropout_layer_type: Type[Union[nn.Dropout2d, nn.Dropout3d]] = getattr(
                nn, f"Dropout{conv_mult}d"
            )
        modules = []

        # Define block based on block_type
        if block_type == "cabdcabd":
            for i in range(2):
                if conv_mult == 0:
                    conv = conv_layer(
                        channels, channels, kernel[i], padding=pad[i]
                    )
                else:
                    conv = conv_layer(
                        channels, channels, kernel[i], padding=pad[i], groups=groups
                    )
                modules.append(conv)
                modules.append(nonlin())
                if batchnorm and conv_mult != 0:
                    modules.append(batchnorm_layer(channels))
                if dropout and conv_mult != 0:
                    modules.append(dropout_layer(dropout))

        elif block_type == "bacdbac":
            for i in range(2):
                if batchnorm and conv_mult != 0:
                    modules.append(batchnorm_layer(channels))
                modules.append(nonlin())
                if conv_mult == 0:
                    conv = conv_layer(
                        channels, channels, kernel[i], padding=pad[i]
                    )
                else:
                    conv = conv_layer(
                        channels, channels, kernel[i], padding=pad[i], groups=groups
                    )
                modules.append(conv)
                if dropout is not None and i == 0 and conv_mult != 0:
                    modules.append(dropout_layer(dropout))

        elif block_type == "bacdbacd":
            for i in range(2):
                if batchnorm and conv_mult != 0:
                    modules.append(batchnorm_layer(channels))
                modules.append(nonlin())
                if conv_mult == 0:
                    conv = conv_layer(
                        channels, channels, kernel[i], padding=pad[i]
                    )
                else:
                    conv = conv_layer(
                        channels, channels, kernel[i], padding=pad[i], groups=groups
                    )
                modules.append(conv)
                if dropout is not None and conv_mult != 0:
                    modules.append(dropout_layer(dropout))

        else:
            raise ValueError(f"Unrecognized block type '{block_type}'")

        if gated:
            modules.append(GateLayer(channels, 1, conv_layer, nonlin))
        self.block = nn.Sequential(*modules)

    def forward(self, inp):
        return self.cp(self.block, inp) + inp


class ResidualGatedBlock(ResidualBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)


class GateLayer(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size, conv_type, nonlin=nn.LeakyReLU):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = conv_type(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate
