"""Robot policy (VLA) inference pipeline.

Extends the VLM pipeline with the robot-specific tail: after the vision +
language stages, an action decoder maps the policy output to an action vector
(e.g. joint angles), a postprocess clamps/normalizes it, and a control step
writes it to the actuator.  The whole chain is the "sensor-to-action" path a
real robot runs every control cycle.

Camera -> capture -> preprocess -> H2D -> vision encoder -> policy (LLM)
       -> action decoder -> postprocess -> control
"""

from __future__ import annotations

import torch
from torch import nn

from inference.vlm.pipeline import VLM, decode_image, make_image_bytes, preprocess


class VLAPolicy(nn.Module):
    """A VLM head plus an action decoder + postprocess."""

    def __init__(self, llm_hidden=512, llm_layers=4, action_dim=7):
        super().__init__()
        self.vlm = VLM(llm_hidden=llm_hidden, llm_layers=llm_layers)
        self.action_decoder = nn.Linear(llm_hidden, action_dim)

    def infer(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vt = self.vlm.vision_encode(img)      # vision + connector
            h = self.vlm.llm_forward(vt)          # policy (LLM)
            h = h.mean(dim=1)                      # pool token sequence
            action = self.action_decoder(h)        # -> action vector
        return action


def postprocess_action(action: torch.Tensor) -> torch.Tensor:
    """Clamp actions to the actuator range and (optionally) normalize."""
    return torch.clamp(action, -1.0, 1.0)


def control_step(action: torch.Tensor) -> torch.Tensor:
    """Simulate writing the action to the actuator (returns the command)."""
    return action.detach().cpu()


def make_camera_frame(seed: int = 0) -> bytes:
    """Simulate a camera capture as JPEG bytes (the real sensor path)."""
    return make_image_bytes(size=224, seed=seed)
