"""Stand-alone copy of the CIFAR-10 split CNN (models/architectures.py, unchanged layers) for the
device benchmarks, so bench_device.py / offload_server.py need only PyTorch and this file.

  client block + client exit : x [B,3,32,32] -> (client_logits [B,10], feature [B,128,8,8])
  server block + server exit : feature [B,128,8,8] -> server_logits [B,10]
The feature is what an offloaded request sends: 128*8*8 fp32 = 32,768 bytes (A in §6.1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_SHAPE = (128, 8, 8)
FEATURE_BYTES = 128 * 8 * 8 * 4
NUM_CLASSES = 10


class CIFARClient(nn.Module):
    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        c = base_channels
        self.conv1 = nn.Conv2d(3, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 2)
        self.conv4 = nn.Conv2d(c * 2, c * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(c * 4)
        self.aux_conv = nn.Conv2d(c * 4, c * 4, 3, padding=1)
        self.aux_bn = nn.BatchNorm2d(c * 4)
        self.aux_fc = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        rep = F.max_pool2d(x, 2)
        a = F.relu(self.aux_bn(self.aux_conv(rep)))
        a = F.adaptive_avg_pool2d(a, 1).flatten(1)
        return self.aux_fc(a), rep


class CIFARServer(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, in_spatial=8):
        super().__init__()
        c = base_channels
        self.conv = nn.Conv2d(c * 4, c * 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(c * 8)
        spatial = in_spatial // 2
        self.fc1 = nn.Linear(c * 8 * spatial * spatial, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, rep):
        x = F.relu(self.bn(self.conv(rep)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)          # the training code returns (logits, None); here logits only


def load(weights_path, device="cpu"):
    """weights_path: the .pth written by export_models.py -> (client, server) in eval mode."""
    sd = torch.load(weights_path, map_location=device)
    c, s = CIFARClient(), CIFARServer()
    c.load_state_dict(sd["client_block_and_exit"])
    s.load_state_dict(sd["server_block_and_exit"])
    return c.to(device).eval(), s.to(device).eval()


def multi_exit_loss(client_logits, server_logits, labels, gamma=0.5):
    ce = nn.CrossEntropyLoss()
    return gamma * ce(client_logits, labels) + (1 - gamma) * ce(server_logits, labels)
