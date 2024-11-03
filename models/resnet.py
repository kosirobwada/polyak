import torch.nn as nn


class ResidualBlock(nn.Module):
    """Base class for different types of residual blocks used in ResNet architectures."""

    def __init__(self, input_channels, output_channels, stride, expansion_factor):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.residual_path = self._make_residual_path(input_channels, output_channels, stride)
        self.shortcut_path = self._make_shortcut_path(input_channels, output_channels, stride)

    def _make_shortcut_path(self, input_channels, output_channels, stride):
        if stride != 1 or input_channels != output_channels * self.expansion_factor:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels * self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels * self.expansion_factor)
            )
        else:
            return nn.Sequential()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_path(x) + self.shortcut_path(x))


class BasicResidualBlock(ResidualBlock):
    expansion_factor = 1

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__(input_channels, output_channels, stride, BasicResidualBlock.expansion_factor)

    def _make_residual_path(self, input_channels, output_channels, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels * self.expansion_factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels * self.expansion_factor)
        )


class BottleNeckResidualBlock(ResidualBlock):
    expansion_factor = 4

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__(input_channels, output_channels, stride, BottleNeckResidualBlock.expansion_factor)

    def _make_residual_path(self, input_channels, output_channels, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels * self.expansion_factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels * self.expansion_factor),
        )


class ResNetModel(nn.Module):
    """ResNet model class to define ResNet architecture."""

    def __init__(self, residual_block, num_blocks_per_layer, dataset_name='CIFAR100'):
        super().__init__()
        self.input_channels = 64
        num_classes = self._get_num_classes(dataset_name)
        self.initial_conv = self._make_initial_conv()
        self.layers = self._make_resnet_layers(residual_block, num_blocks_per_layer)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_output = nn.Linear(512 * residual_block.expansion_factor, num_classes)

    def _get_num_classes(self, dataset_name):
        if dataset_name == 'TinyImageNet':
            return 200
        elif dataset_name == 'CIFAR10':
            return 10
        else:
            return 100

    def _make_initial_conv(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def _make_resnet_layers(self, residual_block, num_blocks_per_layer):
        layers = []
        output_channels = [64, 128, 256, 512]
        for i, num_blocks in enumerate(num_blocks_per_layer):
            stride = 1 if i == 0 else 2
            layers.append(self._make_resnet_layer(residual_block, output_channels[i], num_blocks, stride))
        return nn.Sequential(*layers)

    def _make_resnet_layer(self, residual_block, output_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(residual_block(self.input_channels, output_channels, stride))
            self.input_channels = output_channels * residual_block.expansion_factor
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc_output(x)


def build_resnet(model_type='resnet18', dataset_name='CIFAR100'):
    model_map = {
        'resnet18': (BasicResidualBlock, [2, 2, 2, 2]),
        'resnet34': (BasicResidualBlock, [3, 4, 6, 3]),
        'resnet50': (BottleNeckResidualBlock, [3, 4, 6, 3]),
        'resnet101': (BottleNeckResidualBlock, [3, 4, 23, 3]),
        'resnet152': (BottleNeckResidualBlock, [3, 8, 36, 3]),
    }
    block, layers = model_map[model_type]
    return ResNetModel(block, layers, dataset_name=dataset_name)
