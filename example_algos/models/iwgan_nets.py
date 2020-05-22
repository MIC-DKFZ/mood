from torch import nn
import torch


def weights_init(m):
    if isinstance(m, IWConv2d):
        if m.conv.weight is not None:
            if m.he_init:
                nn.init.kaiming_uniform_(m.conv.weight)
            else:
                nn.init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            nn.init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class IWConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, stride=1, bias=True):
        super(IWConv2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = IWConv2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (
            output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]
        ) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = IWConv2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = input
        output = (
            output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]
        ) / 4
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = (
            torch.stack(stacks, 0)
            .transpose(0, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, output_height, output_width, output_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = IWConv2d(input_dim, output_dim, kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=64):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == "down":
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == "up":
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception("invalid resample value")

        if resample == "down":
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = IWConv2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)
        elif resample == "up":
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = IWConv2d(output_dim, output_dim, kernel_size=kernel_size)
        elif resample is None:
            self.conv_shortcut = IWConv2d(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = IWConv2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = IWConv2d(input_dim, output_dim, kernel_size=kernel_size)
        else:
            raise Exception("invalid resample value")

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class IWGenerator(nn.Module):
    def __init__(self, input_size=64, z_dim=128, n_image_channels=3):
        super(IWGenerator, self).__init__()

        self.size = input_size
        self.n_image_channels = n_image_channels

        self.ssize = self.size // 16
        self.ln1 = nn.Linear(z_dim, self.ssize * self.ssize * 8 * self.size)
        self.rb1 = ResidualBlock(8 * self.size, 8 * self.size, 3, resample="up")
        self.rb2 = ResidualBlock(8 * self.size, 4 * self.size, 3, resample="up")
        self.rb3 = ResidualBlock(4 * self.size, 2 * self.size, 3, resample="up")
        self.rb4 = ResidualBlock(2 * self.size, 1 * self.size, 3, resample="up")
        self.bn = nn.BatchNorm2d(self.size)

        self.conv1 = IWConv2d(1 * self.size, self.n_image_channels, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8 * self.size, self.ssize, self.ssize)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = output.view(-1, self.n_image_channels * self.size * self.size)
        return output


class IWDiscriminator(nn.Module):
    def __init__(self, input_size=64, n_image_channels=3):
        super(IWDiscriminator, self).__init__()

        self.size = input_size
        self.n_image_channels = n_image_channels

        self.ssize = self.size // 16
        self.conv1 = IWConv2d(n_image_channels, self.size, 3, he_init=False)
        self.rb1 = ResidualBlock(self.size, 2 * self.size, 3, resample="down", hw=self.size)
        self.rb2 = ResidualBlock(2 * self.size, 4 * self.size, 3, resample="down", hw=int(self.size / 2))
        self.rb3 = ResidualBlock(4 * self.size, 8 * self.size, 3, resample="down", hw=int(self.size / 4))
        self.rb4 = ResidualBlock(8 * self.size, 8 * self.size, 3, resample="down", hw=int(self.size / 8))
        self.ln1 = nn.Linear(self.ssize * self.ssize * 8 * self.size, 1)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, self.n_image_channels, self.size, self.size)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.size)
        output = self.ln1(output)
        output = output.view(-1)
        return output

    def forward_last_feature(self, input):
        output = input.contiguous()
        output = output.view(-1, self.n_image_channels, self.size, self.size)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.size)
        out_features = output
        output = self.ln1(output)
        output = output.view(-1)
        return output, out_features

    def forward_all_feature(self, input):
        out_features_list = []

        output = input.contiguous()
        output = output.view(-1, self.n_image_channels, self.size, self.size)
        output = self.conv1(output)
        out_features_list.append(output)
        output = self.rb1(output)
        out_features_list.append(output)
        output = self.rb2(output)
        out_features_list.append(output)
        output = self.rb3(output)
        out_features_list.append(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.size)
        out_features_list.append(output)
        output = self.ln1(output)
        out_features_list.append(output)
        output = output.view(-1)
        return output, out_features_list


class IWEncoder(nn.Module):
    def __init__(self, input_size=64, z_dim=128, n_image_channels=3):
        super(IWEncoder, self).__init__()

        self.size = input_size
        self.n_image_channels = n_image_channels

        self.ssize = self.size // 16
        self.conv1 = IWConv2d(n_image_channels, self.size, 3, he_init=False)
        self.rb1 = ResidualBlock(self.size, 2 * self.size, 3, resample="down", hw=self.size)
        self.rb2 = ResidualBlock(2 * self.size, 4 * self.size, 3, resample="down", hw=int(self.size / 2))
        self.rb3 = ResidualBlock(4 * self.size, 8 * self.size, 3, resample="down", hw=int(self.size / 4))
        self.rb4 = ResidualBlock(8 * self.size, 8 * self.size, 3, resample="down", hw=int(self.size / 8))
        self.ln1 = nn.Linear(self.ssize * self.ssize * 8 * self.size, z_dim)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, self.n_image_channels, self.size, self.size)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.size)
        output = self.ln1(output)
        output = torch.tanh(output)
        return output
