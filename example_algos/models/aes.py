import numpy as np
import torch
import torch.distributions as dist

from example_algos.models.nets import BasicEncoder, BasicGenerator


class VAE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(16, 64, 256, 1024),
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        conv_params=None,
        tconv_op=torch.nn.ConvTranspose2d,
        tconv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=torch.nn.LeakyReLU,
        activation_params=None,
        block_op=None,
        block_params=None,
        *args,
        **kwargs
    ):
        """Basic VAE build up of a symetric BasicEncoder (Encoder) and BasicGenerator (Decoder)

        Args:
            input_size ((int, int, int): Size of the input in format CxHxW): 
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim). Defaults to 256
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (16, 64, 256, 1024).
            to_1x1 (bool, optional): [If True, then the last conv layer goes to a latent dimesion is a z_dim x 1 x 1 vector (similar to fully connected) 
                                        or if False allows spatial resolution not to be 1x1 (z_dim x H x W, uses the in the conv_params given conv-kernel-size) ].
                                        Defaults to True.
            conv_op ([torch.nn.Module], optional): [Convolutioon operation used in the encoder to downsample to a new level/ featuremap size]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            tconv_op ([torch.nn.Module], optional): [Upsampling/ Transposed Conv operation used in the decoder to upsample to a new level/ featuremap size]. Defaults to nn.ConvTranspose2d.
            tconv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
        """

        super(VAE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)

        self.enc = BasicEncoder(
            input_size=input_size_enc,
            fmap_sizes=fmap_sizes,
            z_dim=z_dim * 2,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )
        self.dec = BasicGenerator(
            input_size=input_size_dec,
            fmap_sizes=fmap_sizes[::-1],
            z_dim=z_dim,
            upsample_op=tconv_op,
            conv_params=tconv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )

        self.hidden_size = self.enc.output_size

    def forward(self, inpt, sample=True, no_dist=False, **kwargs):
        y1 = self.enc(inpt, **kwargs)

        mu, log_std = torch.chunk(y1, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)
        if sample:
            z_sample = z_dist.rsample()
        else:
            z_sample = mu

        x_rec = self.dec(z_sample)

        if no_dist:
            return x_rec
        else:
            return x_rec, z_dist

    def encode(self, inpt, **kwargs):
        """Encodes a sample and returns the paramters for the approx inference dist. (Normal)

        Args:
            inpt ([tensor]): The input to encode

        Returns:
            mu : The mean used to parameterized a Normal distribution
            std: The standard deviation used to parameterized a Normal distribution
        """
        enc = self.enc(inpt, **kwargs)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        std = torch.exp(log_std)
        return mu, std

    def decode(self, inpt, **kwargs):
        """Decodes a latent space sample, used the generative model (decode = mu_{gen}(z) as used in p(x|z) = N(x | mu_{gen}(z), 1) ).

        Args:
            inpt ([type]): A sample from the latent space to decode

        Returns:
            [type]: [description]
        """
        x_rec = self.dec(inpt, **kwargs)
        return x_rec


class AE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=1024,
        fmap_sizes=(16, 64, 256, 1024),
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        conv_params=None,
        tconv_op=torch.nn.ConvTranspose2d,
        tconv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=torch.nn.LeakyReLU,
        activation_params=None,
        block_op=None,
        block_params=None,
        *args,
        **kwargs
    ):
        """Basic AE build up of a symetric BasicEncoder (Encoder) and BasicGenerator (Decoder)

        Args:
            input_size ((int, int, int): Size of the input in format CxHxW): 
            z_dim (int, optional): [description]. Dimension of the latent / Input dimension (C channel-dim). Defaults to 256
            fmap_sizes (tuple, optional): [Defines the Upsampling-Levels of the generator, list/ tuple of ints, where each 
                                            int defines the number of feature maps in the layer]. Defaults to (16, 64, 256, 1024).
            to_1x1 (bool, optional): [If True, then the last conv layer goes to a latent dimesion is a z_dim x 1 x 1 vector (similar to fully connected) 
                                        or if False allows spatial resolution not to be 1x1 (z_dim x H x W, uses the in the conv_params given conv-kernel-size) ].
                                        Defaults to True.
            conv_op ([torch.nn.Module], optional): [Convolutioon operation used in the encoder to downsample to a new level/ featuremap size]. Defaults to nn.Conv2d.
            conv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            tconv_op ([torch.nn.Module], optional): [Upsampling/ Transposed Conv operation used in the decoder to upsample to a new level/ featuremap size]. Defaults to nn.ConvTranspose2d.
            tconv_params ([dict], optional): [Init parameters for the conv operation]. Defaults to dict(kernel_size=3, stride=2, padding=1, bias=False).
            normalization_op ([torch.nn.Module], optional): [Normalization Operation (e.g. BatchNorm, InstanceNorm,...) -> see ConvModule]. Defaults to nn.BatchNorm2d.
            normalization_params ([dict], optional): [Init parameters for the normalization operation]. Defaults to None.
            activation_op ([torch.nn.Module], optional): [Actiovation Operation/ Non-linearity (e.g. ReLU, Sigmoid,...) -> see ConvModule]. Defaults to nn.LeakyReLU.
            activation_params ([dict], optional): [Init parameters for the activation operation]. Defaults to None.
            block_op ([torch.nn.Module], optional): [Block operation used for each feature map size after each upsample op of e.g. ConvBlock/ ResidualBlock]. Defaults to NoOp.
            block_params ([dict], optional): [Init parameters for the block operation]. Defaults to None.
        """
        super(AE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)

        self.enc = BasicEncoder(
            input_size=input_size_enc,
            fmap_sizes=fmap_sizes,
            z_dim=z_dim,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )
        self.dec = BasicGenerator(
            input_size=input_size_dec,
            fmap_sizes=fmap_sizes[::-1],
            z_dim=z_dim,
            upsample_op=tconv_op,
            conv_params=tconv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )

        self.hidden_size = self.enc.output_size

    def forward(self, inpt, **kwargs):

        y1 = self.enc(inpt, **kwargs)

        x_rec = self.dec(y1)

        return x_rec

    def encode(self, inpt, **kwargs):
        """Encodes a input sample to a latent space sample

        Args:
            inpt ([tensor]): Input sample

        Returns:
            enc: Encoded input sample in the latent space
        """
        enc = self.enc(inpt, **kwargs)
        return enc

    def decode(self, inpt, **kwargs):
        """Decodes a latent space sample back to the input space

        Args:
            inpt ([tensor]): [Latent space sample]

        Returns:
            [rec]: [Encoded latent sample back in the input space]
        """
        rec = self.dec(inpt, **kwargs)
        return rec
