"""
Split architecture for CIFAR-10 matching reference paper §VI-A scale.

Client-side: 4 conv blocks + auxiliary classifier  (small, ~10% of params)
Server-side: 1 conv + 3 FC  (large, ~90% of params)

Forward returns (client_logits, representation_at_cut_layer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARClient(nn.Module):
    """Client-side block: 4 conv layers + auxiliary classifier."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        c = base_channels
        # 4 conv layers, with downsampling at 2nd and 4th
        self.conv1 = nn.Conv2d(3, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        # after pool: 16x16
        self.conv3 = nn.Conv2d(c * 2, c * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 2)
        self.conv4 = nn.Conv2d(c * 2, c * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(c * 4)
        # after pool: 8x8, channels = 4c

        # Auxiliary classifier h: simple conv pool + FC
        self.aux_conv = nn.Conv2d(c * 4, c * 4, 3, padding=1)
        self.aux_bn = nn.BatchNorm2d(c * 4)
        self.aux_fc = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        # block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        # block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        rep = F.max_pool2d(x, 2)  # representation at cut layer: [B, 4c, 8, 8]

        # auxiliary classifier from rep
        a = F.relu(self.aux_bn(self.aux_conv(rep)))
        a = F.adaptive_avg_pool2d(a, 1).flatten(1)
        client_logits = self.aux_fc(a)
        return client_logits, rep


class CIFARServer(nn.Module):
    """Server-side block: 1 conv + 3 FC, takes [B, 4c, 8, 8] -> num_classes."""

    def __init__(self, num_classes=10, base_channels=32, in_spatial=8):
        super().__init__()
        c = base_channels
        in_ch = c * 4
        self.conv = nn.Conv2d(in_ch, c * 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(c * 8)
        # after pool: spatial / 2
        spatial = in_spatial // 2
        flat = c * 8 * spatial * spatial
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, rep):
        x = F.relu(self.bn(self.conv(rep)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, None


class ModelFactory:
    """Convenient constructor; ensures all clients/servers same arch."""

    def __init__(self, dataset='cifar10', num_classes=10, base_channels=32, seed=None):
        self.dataset = dataset
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.seed = seed

    def make_client(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return CIFARClient(num_classes=self.num_classes,
                          base_channels=self.base_channels)

    def make_server(self):
        if self.seed is not None:
            torch.manual_seed(self.seed + 1)
        return CIFARServer(num_classes=self.num_classes,
                          base_channels=self.base_channels)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    f = ModelFactory()
    c = f.make_client()
    s = f.make_server()
    print(f"Client params: {count_params(c):,}")
    print(f"Server params: {count_params(s):,}")
    print(f"Total: {count_params(c) + count_params(s):,}")
    print(f"Client fraction: {count_params(c) / (count_params(c) + count_params(s)) * 100:.1f}%")
    x = torch.randn(4, 3, 32, 32)
    cl, rep = c(x)
    print(f"Client logits: {cl.shape}, Rep: {rep.shape}")
    sl, _ = s(rep)
    print(f"Server logits: {sl.shape}")
