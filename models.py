# vae model
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        base = models.resnet50(pretrained=pretrain)
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *list(base.children())[4:-1]
        )

    def forward(self, x):
        x = self.extractor(x)
        feature = x.view(x.size(0), -1)
        return feature


class Projector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.project_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.project_layer(x)
        z = x / torch.norm(x, p=2, dim=1).view(-1, 1)
        return z


class Linear_Classifier(nn.Module):
    def __init__(self, input_size, classNum):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, classNum)
        )

    def forward(self, x):
        logit = self.fc(x)
        return logit


if __name__ == "__main__":
    model = ResNet(pretrain=False)
    print(model)
