import torch.nn
from torch import nn

dropout_rate = 0.5

class Conv1D_Module_2(nn.Module):
    def __init__(self, channels_block) -> None:
        super().__init__()
        self.group_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[0],
                      out_channels=channels_block[0],
                      groups=channels_block[0],
                      kernel_size=3,
                      bias=True),
            nn.BatchNorm1d(channels_block[0]),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[0],
                      out_channels=channels_block[1],
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.group_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[1],
                      out_channels=channels_block[1],
                      groups=channels_block[1],
                      kernel_size=3,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[1]),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[1],
                      out_channels=channels_block[2],
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(2 * channels_block[2], 32, bias=True)
        self.fc2 = nn.Linear(32, 1, bias=True)

    def forward(self, input):
        output = self.group_conv1(input)
        output = self.point_conv1(output)
        output = self.group_conv2(output)
        output = self.point_conv2(output)
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output

class Conv2D_Module_5(nn.Module):
    def __init__(self, channels_block) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels_block[0],
                      out_channels=channels_block[1],
                      kernel_size=3,
                      padding='same',
                      bias=True),
            nn.BatchNorm2d(channels_block[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.group_conv1 = nn.Sequential(
            nn.Conv2d(channels_block[1], channels_block[1], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[1], bias=True),
            nn.BatchNorm2d(channels_block[1]),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channels_block[1], channels_block[2], kernel_size=1, bias=True),
            nn.BatchNorm2d(channels_block[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.group_conv2 = nn.Sequential(
            nn.Conv2d(channels_block[2], channels_block[2], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[2], bias=True),
            nn.BatchNorm2d(channels_block[2]),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(channels_block[2], channels_block[3], 1, bias=True),
            nn.BatchNorm2d(channels_block[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.group_conv3 = nn.Sequential(
            nn.Conv2d(channels_block[3], channels_block[3], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[3], bias=True),
            nn.BatchNorm2d(channels_block[3]),
        )
        self.point_conv3 = nn.Sequential(
            nn.Conv2d(channels_block[3], channels_block[4], 1, bias=True),
            nn.BatchNorm2d(channels_block[4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.group_conv4 = nn.Sequential(
            nn.Conv2d(channels_block[4], channels_block[4], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[4], bias=True),
            nn.BatchNorm2d(channels_block[4]),
        )
        self.point_conv4 = nn.Sequential(
            nn.Conv2d(channels_block[4], channels_block[5], 1, bias=True),
            nn.BatchNorm2d(channels_block[5]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()

    def forward(self, input):
        output = self.conv1(input)
        output = self.group_conv1(output)
        output = self.point_conv1(output)
        output = self.group_conv2(output)
        output = self.point_conv2(output)
        output = self.group_conv3(output)
        output = self.point_conv3(output)
        output = self.group_conv4(output)
        output = self.point_conv4(output)
        output = output.mean(dim=-1)
        output = output.mean(dim=-1)
        return output

class RDTNet(nn.Module):
    def __init__(self, channels_block) -> None:
        super().__init__()

        self.Conv2D_Net = Conv2D_Module_5(channels_block[0])
        self.Conv1D_Net = Conv1D_Module_2(channels_block[1])
        self.feature_num = channels_block[1][0]

    def forward(self, inputs):
        data = self.Conv2D_Net(inputs)
        data = torch.reshape(data, (-1, self.feature_num, 16))
        output = self.Conv1D_Net(data)
        output = torch.sigmoid(output)

        return output
