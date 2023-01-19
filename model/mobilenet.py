from torch import nn
from torchvision import ops


def depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResBlock(nn.Module):
    def __init__(self, expansion, in_channels, out_channels, kernel_size, se_ratio, activation, stride):
        super().__init__()

        self.activation = activation

        if expansion > 1:
            self.expand = nn.Conv2d(in_channels=in_channels, out_channels=depth(in_channels * expansion),
                                    kernel_size=1, padding='same', bias=False)
            self.expand_bn = nn.BatchNorm2d(num_features=depth(in_channels * expansion), eps=1e-3, momentum=1e-3)
            self.expand_activation = activation()

        self.depthwise = nn.Conv2d(in_channels=depth(in_channels * expansion), stride=stride,
                                   out_channels=depth(in_channels * expansion), kernel_size=kernel_size,
                                   groups=depth(in_channels * expansion), padding=(kernel_size - 1) // 2, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(num_features=depth(in_channels * expansion), eps=1e-3, momentum=1e-3)
        self.depthwise_activation = activation()

        if se_ratio:
            self.se = ops.SqueezeExcitation(input_channels=depth(in_channels * expansion),
                                            squeeze_channels=depth(in_channels * expansion * se_ratio),
                                            scale_activation=activation)

        self.project = nn.Conv2d(in_channels=depth(in_channels * expansion), out_channels=out_channels,
                                 kernel_size=1, padding='same', bias=False)
        self.project_bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3, momentum=1e-3)
        self.project_activation = activation()

        self.expansion = expansion
        self.se_ratio = se_ratio
        self.residual_connection = in_channels == out_channels and stride == 1

    def forward(self, x):
        shortcut = x

        if self.expansion > 1:
            x = self.expand(x)
            x = self.expand_bn(x)
            x = self.expand_activation(x)

        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activation(x)

        if self.se_ratio:
            x = self.se(x)

        x = self.project(x)
        x = self.project_bn(x)
        x = self.project_activation(x)

        if self.residual_connection:
            x = x + shortcut

        return x


def MobilenetV3(arch='large', in_channels=3, activation=nn.Hardswish, kernel=5, se_ratio=0.25, include_top=True,
                dropout=0.2, num_classes=1000, classifier_activation=nn.Softmax, finetune=False):

    layers = [nn.Conv2d(in_channels=in_channels, out_channels=16,
                        kernel_size=3, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(num_features=16, eps=1e-3, momentum=1e-3), activation()]

    if finetune:
        layers[0].requires_grad_(False)
        layers[1].requires_grad_(False)

    if arch == 'large':
        stack_cfg = [
            [1, 16, 16, 3, None, nn.ReLU, 1],
            [4, 16, 24, 3, None, nn.ReLU, 2],
            [3, 24, 24, 3, None, nn.ReLU, 1],
            [3, 24, 40, kernel, se_ratio, nn.ReLU, 2],
            [3, 40, 40, kernel, se_ratio, nn.ReLU, 1],
            [3, 40, 40, kernel, se_ratio, nn.ReLU, 1],
            [6, 40, 80, 3, None, activation, 2],
            [2.5, 80, 80, 3, None, activation, 1],
            [2.3, 80, 80, 3, None, activation, 1],
            [2.3, 80, 80, 3, None, activation, 1],
            [6, 80, 112, 3, se_ratio, activation, 1],
            [6, 112, 112, 3, se_ratio, activation, 1],
            [6, 112, 160, kernel, se_ratio, activation, 2],
            [6, 160, 160, kernel, se_ratio, activation, 1],
            [6, 160, 160, kernel, se_ratio, activation, 1]
        ]
        out_ch = 160
    elif arch == 'small':
        stack_cfg = [
            [1, 16, 16, 3, None, nn.ReLU, 2],
            [72. / 16, 16, 24, 3, None, nn.ReLU, 2],
            [88. / 24, 24, 24, 3, None, nn.ReLU, 1],
            [4, 24, 40, kernel, se_ratio, activation, 2],
            [6, 40, 40, kernel, se_ratio, activation, 1],
            [6, 40, 40, kernel, se_ratio, activation, 1],
            [3, 40, 48, kernel, se_ratio, activation, 1],
            [3, 48, 48, kernel, se_ratio, activation, 1],
            [6, 48, 96, kernel, se_ratio, activation, 2],
            [6, 96, 96, kernel, se_ratio, activation, 1],
            [6, 96, 96, kernel, se_ratio, activation, 1]
        ]
        out_ch = 96
    else:
        raise ValueError(f"Not recognized {arch}")

    for i in stack_cfg:
        layers.append(InvertedResBlock(*i))
        if finetune:
            layers[-1].requires_grad_(False)

    if include_top:
        layers.append(nn.Conv2d(in_channels=out_ch, out_channels=out_ch * 6, kernel_size=1, padding='same',
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_ch * 6, eps=1e-3, momentum=1e-3))
        layers.append(activation())

        layers.append(nn.AdaptiveAvgPool2d(1))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=out_ch * 6, out_features=out_ch * 6))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features=out_ch * 6, out_features=num_classes))
        layers.append(classifier_activation())

    return nn.Sequential(*layers)
