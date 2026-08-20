"""The compact 6x64 squeeze-excitation residual policy/WDL network."""

from __future__ import annotations

from typing import Any

from .errors import DependencyUnavailable

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # Operations that only inspect manifests must remain usable.
    torch = None
    Tensor = Any
    nn = None


def require_torch() -> Any:
    if torch is None:
        raise DependencyUnavailable("training requires PyTorch; run `uv sync --extra train`")
    return torch


if nn is not None:

    class SqueezeExcitation(nn.Module):
        def __init__(self, channels: int, hidden: int):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels)

        def forward(self, value: Tensor) -> Tensor:
            gates = self.pool(value).flatten(1)
            gates = torch.relu(self.fc1(gates))
            gates = torch.sigmoid(self.fc2(gates)).unsqueeze(-1).unsqueeze(-1)
            return value * gates

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, se_hidden: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.se = SqueezeExcitation(channels, se_hidden)

        def forward(self, value: Tensor) -> Tensor:
            residual = value
            value = torch.relu(self.bn1(self.conv1(value)))
            value = self.se(self.bn2(self.conv2(value)))
            return torch.relu(value + residual)

    class AlphaMiniNet(nn.Module):
        """Position-preserving policy head plus compact spatial WDL head."""

        def __init__(
            self,
            *,
            input_planes: int = 22,
            channels: int = 64,
            residual_blocks: int = 6,
            se_hidden: int = 8,
            action_size: int = 4672,
        ):
            super().__init__()
            if action_size != 64 * 73:
                raise ValueError("AlphaMini v1 action_size must equal 64 * 73")
            self.input_planes = input_planes
            self.action_size = action_size
            self.stem = nn.Sequential(
                nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(
                *(ResidualBlock(channels, se_hidden) for _ in range(residual_blocks))
            )
            # Canonical action index is plane * 64 + origin; NCHW flattening preserves it.
            self.policy = nn.Conv2d(channels, 73, 1, bias=True)
            self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(8)
            self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
            self.value_fc2 = nn.Linear(128, 3)

        def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
            value = self.body(self.stem(inputs))
            policy = self.policy(value).contiguous().view(-1, self.action_size)
            wdl = torch.relu(self.value_bn(self.value_conv(value))).flatten(1)
            wdl = torch.relu(self.value_fc1(wdl))
            return policy, self.value_fc2(wdl)


else:

    class AlphaMiniNet:  # pragma: no cover - only instantiated after require_torch
        def __init__(self, *_: Any, **__: Any):
            require_torch()


def build_model(config: Any) -> Any:
    require_torch()
    model = config.values["model"] if hasattr(config, "values") else config["model"]
    return AlphaMiniNet(
        input_planes=int(model["input_planes"]),
        channels=int(model["channels"]),
        residual_blocks=int(model["residual_blocks"]),
        se_hidden=int(model["se_hidden"]),
        action_size=int(model["action_size"]),
    )


def parameter_count(model: Any) -> int:
    require_torch()
    return sum(parameter.numel() for parameter in model.parameters())
