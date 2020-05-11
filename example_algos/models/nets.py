import warnings

import numpy as np
import torch
import torch.nn as nn


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        """NoOp Pytorch Module.
        Forwards the given input as is.
        """
        super(NoOp, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        """Basic Conv Pytorch Conv Module
        Has can have a Conv Op, a Normlization Op and a Non Linearity:
        x = conv(x)
        x = some_norm(x)
        x = nonlin(x)

        Args:
            in_channels ([int]): [Number on input channels/ feature maps]
            out_channels ([int]): [Number of ouput channels/ feature maps]
            conv_op ([torch.nn.Module], optional): [Conv operation]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to None.
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...)]. Defaults to None.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...)]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
        """

        super(ConvModule, self).__init__()

        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}
        self.activation_params = activation_params
        if self.activation_params is None:
            self.activation_params = {}
        self.normalization_params = normalization_params
        if self.normalization_params is None:
            self.normalization_params = {}

        self.conv = None
        if conv_op is not None and not isinstance(conv_op, str):
            self.conv = conv_op(in_channels, out_channels, **self.conv_params)

        self.normalization = None
        if normalization_op is not None and not isinstance(normalization_op, str):
            self.normalization = normalization_op(out_channels, **self.normalization_params)

        self.activation = None
        if activation_op is not None and not isinstance(activation_op, str):
            self.activation = activation_op(**self.activation_params)

    def forward(self, input, conv_add_input=None, normalization_add_input=None, activation_add_input=None):

        x = input

        if self.conv is not None:
            if conv_add_input is None:
                x = self.conv(x)
            else:
                x = self.conv(x, **conv_add_input)

        if self.normalization is not None:
            if normalization_add_input is None:
                x = self.normalization(x)
            else:
                x = self.normalization(x, **normalization_add_input)

        if self.activation is not None:
            if activation_add_input is None:
                x = self.activation(x)
            else:
                x = self.activation(x, **activation_add_input)

        # nn.functional.dropout(x, p=0.95, training=True)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_convs: int,
        n_featmaps: int,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=nn.BatchNorm2d,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        """Basic Conv block with repeated conv, build up from repeated @ConvModules (with same/fixed feature map size)

        Args:
            n_convs ([type]): [Number of convolutions]
            n_featmaps ([type]): [Feature map size of the conv]
            conv_op ([torch.nn.Module], optional): [Convulioton operation -> see ConvModule ]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to None.
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
        """

        super(ConvBlock, self).__init__()

        self.n_featmaps = n_featmaps
        self.n_convs = n_convs
        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}

        self.conv_list = nn.ModuleList()

        for i in range(self.n_convs):
            conv_layer = ConvModule(
                n_featmaps,
                n_featmaps,
                conv_op=conv_op,
                conv_params=conv_params,
                normalization_op=normalization_op,
                normalization_params=normalization_params,
                activation_op=activation_op,
                activation_params=activation_params,
            )
            self.conv_list.append(conv_layer)

    def forward(self, input, **frwd_params):
        x = input
        for conv_layer in self.conv_list:
            x = conv_layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        n_convs,
        n_featmaps,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=nn.BatchNorm2d,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        """Basic Conv block with repeated conv, build up from repeated @ConvModules (with same/fixed feature map size) and a skip/ residual connection:
        x = input
        x = conv_block(x)
        out = x + input

        Args:
            n_convs ([type]): [Number of convolutions in the conv block]
            n_featmaps ([type]): [Feature map size of the conv block]
            conv_op ([torch.nn.Module], optional): [Convulioton operation -> see ConvModule ]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to None.
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
        """
        super(ResBlock, self).__init__()

        self.n_featmaps = n_featmaps
        self.n_convs = n_convs
        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}

        self.conv_block = ConvBlock(
            n_featmaps,
            n_convs,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
        )

    def forward(self, input, **frwd_params):
        x = input
        x = self.conv_block(x)

        out = x + input

        return out


# Basic Generator
class BasicGenerator(nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(256, 128, 64),
        upsample_op=nn.ConvTranspose2d,
        conv_params=None,
        normalization_op=NoOp,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
        block_op=NoOp,
        block_params=None,
        to_1x1=True,
    ):
        """Basic configureable Generator/ Decoder.
        Allows for mutilple "feature-map" levels defined by the feature map size, where for each feature map size a conv operation + optional conv block is used.

        Args:
            input_size ((int, int, int): Size of the input in format CxHxW): 
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim).
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (256, 128, 64).
            upsample_op ([torch.nn.Module], optional): [Upsampling operation used, to upsample to a new level/ featuremap size]. Defaults to nn.ConvTranspose2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
            to_1x1 (bool, optional): [If Latent dimesion is a z_dim x 1 x 1 vector (True) or if allows spatial resolution not to be 1x1 (z_dim x H x W) (False) ]. Defaults to True.
        """

        super(BasicGenerator, self).__init__()

        if conv_params is None:
            conv_params = dict(kernel_size=4, stride=2, padding=1, bias=False)
        if block_op is None:
            block_op = NoOp
        if block_params is None:
            block_params = {}

        n_channels = input_size[0]
        input_size_ = np.array(input_size[1:])

        if not isinstance(fmap_sizes, list) and not isinstance(fmap_sizes, tuple):
            raise AttributeError("fmap_sizes has to be either a list or tuple or an int")
        elif len(fmap_sizes) < 2:
            raise AttributeError("fmap_sizes has to contain at least three elements")
        else:
            h_size_bot = fmap_sizes[0]

        # We need to know how many layers we will use at the beginning
        input_size_new = input_size_ // (2 ** len(fmap_sizes))
        if np.min(input_size_new) < 2 and z_dim is not None:
            raise AttributeError("fmap_sizes to long, one image dimension has already perished")

        ### Start block
        start_block = []

        if not to_1x1:
            kernel_size_start = [min(conv_params["kernel_size"], i) for i in input_size_new]
        else:
            kernel_size_start = input_size_new.tolist()

        if z_dim is not None:
            self.start = ConvModule(
                z_dim,
                h_size_bot,
                conv_op=upsample_op,
                conv_params=dict(kernel_size=kernel_size_start, stride=1, padding=0, bias=False),
                normalization_op=normalization_op,
                normalization_params=normalization_params,
                activation_op=activation_op,
                activation_params=activation_params,
            )

            input_size_new = input_size_new * 2
        else:
            self.start = NoOp()

        ### Middle block (Done until we reach ? x input_size/2 x input_size/2)
        self.middle_blocks = nn.ModuleList()

        for h_size_top in fmap_sizes[1:]:

            self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvModule(
                    h_size_bot,
                    h_size_top,
                    conv_op=upsample_op,
                    conv_params=conv_params,
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            input_size_new = input_size_new * 2

        ### End block
        self.end = ConvModule(
            h_size_bot,
            n_channels,
            conv_op=upsample_op,
            conv_params=conv_params,
            normalization_op=None,
            activation_op=None,
        )

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


# Basic Encoder
class BasicEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(64, 128, 256),
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=NoOp,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
        block_op=NoOp,
        block_params=None,
        to_1x1=True,
    ):
        """Basic configureable Encoder.
        Allows for mutilple "feature-map" levels defined by the feature map size, where for each feature map size a conv operation + optional conv block is used. 

        Args:
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim).
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (64, 128, 256).
            conv_op ([torch.nn.Module], optional): [Convolutioon operation used to downsample to a new level/ featuremap size]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
            to_1x1 (bool, optional): [If True, then the last conv layer goes to a latent dimesion is a z_dim x 1 x 1 vector (similar to fully connected) or if False allows spatial resolution not to be 1x1 (z_dim x H x W, uses the in the conv_params given conv-kernel-size) ]. Defaults to True.
        """
        super(BasicEncoder, self).__init__()

        if conv_params is None:
            conv_params = dict(kernel_size=3, stride=2, padding=1, bias=False)
        if block_op is None:
            block_op = NoOp
        if block_params is None:
            block_params = {}

        n_channels = input_size[0]
        input_size_new = np.array(input_size[1:])

        if not isinstance(fmap_sizes, list) and not isinstance(fmap_sizes, tuple):
            raise AttributeError("fmap_sizes has to be either a list or tuple or an int")
        # elif len(fmap_sizes) < 2:
        #     raise AttributeError("fmap_sizes has to contain at least three elements")
        else:
            h_size_bot = fmap_sizes[0]

        ### Start block
        self.start = ConvModule(
            n_channels,
            h_size_bot,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params={},
            activation_op=activation_op,
            activation_params=activation_params,
        )
        input_size_new = input_size_new // 2

        ### Middle block (Done until we reach ? x 4 x 4)
        self.middle_blocks = nn.ModuleList()

        for h_size_top in fmap_sizes[1:]:

            self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvModule(
                    h_size_bot,
                    h_size_top,
                    conv_op=conv_op,
                    conv_params=conv_params,
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            input_size_new = input_size_new // 2

            if np.min(input_size_new) < 2 and z_dim is not None:
                raise ("fmap_sizes to long, one image dimension has already perished")

        ### End block
        if not to_1x1:
            kernel_size_end = [min(conv_params["kernel_size"], i) for i in input_size_new]
        else:
            kernel_size_end = input_size_new.tolist()

        if z_dim is not None:
            self.end = ConvModule(
                h_size_bot,
                z_dim,
                conv_op=conv_op,
                conv_params=dict(kernel_size=kernel_size_end, stride=1, padding=0, bias=False),
                normalization_op=None,
                activation_op=None,
            )

            if to_1x1:
                self.output_size = (z_dim, 1, 1)
            else:
                self.output_size = (z_dim, *[i - (j - 1) for i, j in zip(input_size_new, kernel_size_end)])
        else:
            self.end = NoOp()
            self.output_size = input_size_new

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output
