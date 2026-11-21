"""Detector v3 network: dilated CNN over the front-end image, 3-channel head on a (32 x 75) heatmap.

Head channels per (frequency, time) cell: 0 fly-over logit (CPA = local maximum), 1 log(v / 50 m/s) source speed at
CPA, 2 log(d / 1000 m) slant distance at CPA. Same layout as the research model (v0 body, v1 head), so the shipped
state dict loads unchanged.
"""
import os
from importlib import resources


def _block(nn, ci, co, dil=1, pool=None):
    layers = [nn.Conv2d(ci, co, 3, padding=dil, dilation=dil), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
              nn.Conv2d(co, co, 3, padding=dil, dilation=dil), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


def build_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(_block(nn, 1, 32, pool=(2, 2)), _block(nn, 32, 64, pool=(2, 2)),
                                      _block(nn, 64, 96, dil=2), _block(nn, 96, 128, dil=4),
                                      _block(nn, 128, 128, dil=8))
            self.head = nn.Conv2d(128, 3, 1)

        def forward(self, x):
            return self.head(self.body(x))           # (B, 3, HF, HT) logits

    return Net()


def model_dir():
    d = os.environ.get("SKYHUM_V3_MODEL")
    return d if d else str(resources.files("skyhum.models").joinpath("v3"))


def weights_path():
    return os.path.join(model_dir(), "ckpt.pt")


def load_net(path=None, dev="cpu"):
    """Load the v3 checkpoint ({"arch": {...}, "state": state dict}) as plain tensors (weights_only, no pickle
    code) and return the network in eval mode."""
    import torch
    path = path or weights_path()
    ck = torch.load(path, map_location=dev, weights_only=True)
    arch = ck.get("arch", {}) if isinstance(ck, dict) else {}
    state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
    if arch.get("kind", "v1") != "v1" or state["head.weight"].shape[0] != 3:
        raise ValueError(f"{path}: not a detector v3 checkpoint (arch {arch})")
    net = build_net().to(dev)
    net.load_state_dict(state)
    net.eval()
    return net

