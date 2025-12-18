"""
Standalone OSNet ReID encoder for person re-identification.

OSNet: Omni-Scale Feature Learning for Person Re-Identification (ICCV 2019)
Reference: https://github.com/KaiyangZhou/deep-person-reid

This is a self-contained implementation that only requires PyTorch.
"""

from __future__ import annotations

import os
import errno
import warnings
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["OSNetEncoder", "osnet_x1_0", "osnet_x0_25"]

# Google Drive download URLs for pretrained weights
PRETRAINED_URLS = {
    "osnet_x1_0": "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
    "osnet_x0_75": "https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq",
    "osnet_x0_5": "https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i",
    "osnet_x0_25": "https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
}


# ============================================================================
# OSNet Model Architecture (from torchreid)
# ============================================================================


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups
        )
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 (linear) + dw 3x3 (nonlinear)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Channel-wise attention gate (SE-style)."""

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation="sigmoid", reduction=16):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid() if gate_activation == "sigmoid" else None

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return inp * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network for person re-identification."""

    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss="softmax", IN=False, **kwargs):
        super().__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers) == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # Convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # FC layer
        self.fc = self._construct_fc_layer(feature_dim, channels[3])

        # Classifier (not used for feature extraction)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = [block(in_channels, out_channels, IN=IN)]
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        return nn.Sequential(
            nn.Linear(input_dim, fc_dims),
            nn.BatchNorm1d(fc_dims),
            nn.ReLU(inplace=True),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return (y, v) if self.loss == "triplet" else y


# ============================================================================
# Weight Download & Model Factory
# ============================================================================


def _get_cache_dir() -> Path:
    """Get the cache directory for OSNet weights."""
    torch_home = os.environ.get("TORCH_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "torch"))
    return Path(os.path.expanduser(torch_home)) / "checkpoints"


def download_weights(model_name: str = "osnet_x1_0") -> Path:
    """Download pretrained weights from Google Drive.

    Args:
        model_name: One of 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25'

    Returns:
        Path to the downloaded weights file.
    """
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_imagenet.pth"
    cached_file = cache_dir / filename

    if not cached_file.exists():
        try:
            import gdown

            url = PRETRAINED_URLS[model_name]
            print(f"Downloading {model_name} weights to {cached_file}...")
            gdown.download(url, str(cached_file), quiet=False)
        except ImportError:
            raise RuntimeError(
                f"gdown is required to download weights. Install with: pip install gdown\n"
                f"Or manually download from {PRETRAINED_URLS[model_name]} to {cached_file}"
            )

    return cached_file


def _load_pretrained_weights(model: nn.Module, model_name: str) -> None:
    """Load pretrained weights into model."""
    cached_file = download_weights(model_name)
    state_dict = torch.load(cached_file, map_location="cpu")
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(f"No layers matched from {cached_file}")
    else:
        print(f"Loaded {len(matched_layers)} layers from {cached_file}")
        if discarded_layers:
            print(f"Discarded {len(discarded_layers)} layers: {discarded_layers[:5]}...")


def osnet_x1_0(num_classes: int = 1000, pretrained: bool = True, **kwargs) -> OSNet:
    """OSNet with width multiplier 1.0 (standard size, ~2.2M params)."""
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        **kwargs,
    )
    if pretrained:
        _load_pretrained_weights(model, "osnet_x1_0")
    return model


def osnet_x0_25(num_classes: int = 1000, pretrained: bool = True, **kwargs) -> OSNet:
    """OSNet with width multiplier 0.25 (very small, ~0.2M params)."""
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        **kwargs,
    )
    if pretrained:
        _load_pretrained_weights(model, "osnet_x0_25")
    return model


# ============================================================================
# BOTSORT-Compatible Encoder Wrapper
# ============================================================================


class OSNetEncoder:
    """OSNet encoder compatible with BOTSORT tracker interface.

    This wrapper extracts person crops from full frames and computes
    512-dimensional ReID embeddings using OSNet.

    Args:
        model_name: OSNet variant ('osnet_x1_0' or 'osnet_x0_25')
        device: Torch device ('cuda' or 'cpu')
        pretrained: Whether to load pretrained weights

    Example:
        >>> encoder = OSNetEncoder('osnet_x1_0', device='cuda')
        >>> embeddings = encoder(frame, detections)  # Returns list of 512-dim vectors
    """

    def __init__(self, model_name: str = "osnet_x1_0", device: str = "cuda", pretrained: bool = True):
        self.device = device

        # Load model
        if model_name == "osnet_x1_0":
            self.model = osnet_x1_0(pretrained=pretrained)
        elif model_name == "osnet_x0_25":
            self.model = osnet_x0_25(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'osnet_x1_0' or 'osnet_x0_25'")

        self.model.to(device).eval()

        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected persons.

        Args:
            img: Full frame (H, W, 3) RGB numpy array [0-255]
            dets: (N, 5+) array with [cx, cy, w, h, ...] in xywh format

        Returns:
            List of N L2-normalized 512-dim embeddings (numpy arrays)
        """
        if len(dets) == 0:
            return []

        crops = self._extract_crops(img, dets[:, :4])

        with torch.no_grad():
            # Normalize: [0,255] -> [0,1] -> ImageNet normalized
            crops = crops / 255.0
            crops = (crops - self.mean) / self.std

            # Forward pass
            features = self.model(crops)

            # L2 normalize
            features = F.normalize(features, p=2, dim=1)

        return [f.cpu().numpy() for f in features]

    def _extract_crops(self, img: np.ndarray, boxes_xywh: np.ndarray) -> torch.Tensor:
        """Extract and resize person crops from image.

        Args:
            img: Full frame (H, W, 3) RGB
            boxes_xywh: (N, 4) array with [cx, cy, w, h]

        Returns:
            Tensor of shape (N, 3, 256, 128)
        """
        crops = []
        h, w = img.shape[:2]

        for cx, cy, bw, bh in boxes_xywh:
            # Convert xywh (center) to xyxy (corners)
            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(w, cx + bw / 2))
            y2 = int(min(h, cy + bh / 2))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            else:
                # Resize to OSNet input size (256 height x 128 width)
                crop = cv2.resize(crop, (128, 256))

            crops.append(crop)

        # Stack and convert: (N, H, W, C) -> (N, C, H, W)
        crops = np.stack(crops)
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).float().to(self.device)
        return crops


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    # Test encoder
    print("Testing OSNetEncoder...")
    encoder = OSNetEncoder("osnet_x1_0", device="cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets = np.array([[320, 240, 100, 200, 0], [100, 100, 50, 100, 1]])  # 2 detections

    embeddings = encoder(img, dets)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding shape: {embeddings[0].shape}")
    print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
    print("Done!")
