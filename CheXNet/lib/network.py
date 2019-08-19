import torch
from torch import nn
import torch.nn.functional as F
try:
    from lib.dense_net import densenet121
except ModuleNotFoundError:
    from dense_net import densenet121


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        print('Network Self.')
        net = densenet121(True)

        features = net.features

        self.head = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0,
        )

        self.denseblock1 = features.denseblock1
        self.transition1 = features.transition1

        self.denseblock2 = features.denseblock2
        self.transition2 = features.transition2

        self.denseblock3 = features.denseblock3
        self.transition3 = features.transition3

        self.denseblock4 = features.denseblock4
        self.norm5 = features.norm5

        self.classifier = nn.Sequential(
            nn.Linear(1024, 14, bias=True),
            nn.Sigmoid(),
        )

        self.weight_init(self.classifier)


    @staticmethod
    def weight_init(layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        output = self.head(x)

        output = self.denseblock1(output)
        output = self.transition1(output)

        output = self.denseblock2(output)
        output = self.transition2(output)

        output = self.denseblock3(output)
        output = self.transition3(output)

        output = self.denseblock4(output)
        features = self.norm5(output)
        out = F.relu(features, inplace=True)

        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        prediction = self.classifier(out)

        if self.training:
            return prediction
        else:
            return prediction



if __name__ == '__main__':
    net = Network()


    x = torch.randn(1, 3, 224, 224)
    a = net(x)
    print(a.size())
