"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import torch
from torch import nn

_BACKBONE_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "mlp": {"hidden_dim": 512, "depth": 4},
}
VALID_BACKBONES = set(_BACKBONE_DEFAULT_KWARGS)


def resolve_backbone_kwargs(backbone: str, **kwargs: Any) -> dict[str, Any]:
    """Validate and resolve kwargs for a given backbone."""
    if backbone not in _BACKBONE_DEFAULT_KWARGS:
        raise ValueError(
            f"Unknown backbone: {backbone}. Supported backbones: {sorted(VALID_BACKBONES)}"
        )

    defaults = _BACKBONE_DEFAULT_KWARGS[backbone]
    unexpected = set(kwargs) - set(defaults)
    if unexpected:
        raise ValueError(
            f"Unexpected kwargs for backbone '{backbone}': {sorted(unexpected)}. "
            f"Expected subset of {sorted(defaults)}."
        )

    resolved = {**defaults, **kwargs}
    if backbone == "mlp":
        resolved["hidden_dim"] = int(resolved["hidden_dim"])
        resolved["depth"] = int(resolved["depth"])
        if resolved["hidden_dim"] <= 0 or resolved["depth"] <= 0:
            raise ValueError("MLP backbone requires positive 'hidden_dim' and 'depth'.")
    return resolved


def get_policy_checkpoint_config(model: "BasePolicy") -> dict[str, Any]:
    """Return checkpoint model config metadata for reconstructing a policy."""
    backbone = getattr(model, "backbone_name", None)
    backbone_kwargs = getattr(model, "backbone_kwargs", None)
    if backbone is None or backbone_kwargs is None:
        return {}
    return {
        "backbone": backbone,
        "backbone_kwargs": dict(backbone_kwargs),
    }

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        activation: type[nn.Module] = nn.GELU,
        use_layernorm: bool = False
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_layernorm = use_layernorm

        self.layers = nn.ModuleList()

        for i in range(depth):
            d_out = out_dim if i == depth - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, d_out))
            if i != depth - 1:
                if use_layernorm:
                    self.layers.append(nn.LayerNorm(d_out))
                self.layers.append(activation())
            in_dim = d_out

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        chunk_size: int,
        backbone: str = "mlp",
        **kwargs
        ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        assert isinstance(backbone, str) and backbone, (
            "ObstaclePolicy requires a non-empty backbone string."
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()
        self.backbone_name = backbone
        self.backbone_kwargs = resolve_backbone_kwargs(backbone, **kwargs)

        # Select backbone
        if backbone == "mlp":
            self.backbone = MLP(
                state_dim,
                chunk_size * action_dim,
                hidden_dim=self.backbone_kwargs["hidden_dim"],
                depth=self.backbone_kwargs["depth"],
                use_layernorm=True,
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        assert state.ndim == 2 and state.shape[1] == self.state_dim, "State must have shape (B, state_dim)"
            
        B = state.shape[0]
        pred = self.backbone(state)
        pred = pred.reshape(B, self.chunk_size, self.action_dim)
        return pred

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # Forward pass
        pred_action_chunk = self(state)
        assert pred_action_chunk.shape == action_chunk.shape, "Predicted and actual action chunks must have the same shape"
        
        # Compute loss
        loss = self.loss_fn(pred_action_chunk, action_chunk)
        return loss

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self(state)



# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    backbone: str | None = None,
    **kwargs,
    # TODO,
) -> BasePolicy:
    if policy_type == "obstacle":
        if backbone is None:
            backbone = "mlp"
        assert isinstance(backbone, str) and backbone, (
            "build_policy() requires a non-empty 'backbone' for obstacle policy."
        )
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            backbone=backbone,
            **kwargs,
            # TODO: Build with your chosen specifications
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
