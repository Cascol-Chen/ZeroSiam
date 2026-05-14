# ZeroSiam paper: https://arxiv.org/pdf/2509.23183
# ZeroSiam code: https://github.com/Cascol-Chen/ZeroSiam
# Adapted from Tent: https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.byol_wrapper import BYOLWrapper


class ZeroSiam(nn.Module):
    """ZeroSiam test-time adaptation wrapper.

    ZeroSiam adds a lightweight asymmetric branch to test-time entropy
    minimization. A single backbone pass produces the feature, the predictor
    creates an online branch for adaptation, and the original prediction is
    used as a stop-gradient target. The alignment between the two branches
    regularizes shortcut updates and helps avoid collapsed predictions.
    """

    def __init__(self, model: BYOLWrapper, backbone_optimizer, predictor_optimizer, steps=1, episodic=False, alignment_weight=1.0):
        super().__init__()
        assert steps > 0, "ZeroSiam requires >= 1 adaptation step"

        self.model = model
        self.backbone_optimizer = backbone_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.steps = steps
        self.episodic = episodic
        self.alignment_weight = alignment_weight # alpha in Eq. 4 of the paper

        self.model_state, self.backbone_optimizer_state, self.predictor_optimizer_state = copy_model_and_optimizer(self.model, self.backbone_optimizer, self.predictor_optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Run one forward pass, update ZeroSiam parameters, return logits.

        ZeroSiam keeps the backbone pass single-shot: ``self.model(x)`` returns
        both the classifier output and the penultimate feature. The target and
        online branches are then built from that shared feature.
        """
        outputs, features = self.model(x)
        loss = self.cal_loss(features)

        loss.backward()
        self.backbone_optimizer.step()
        self.predictor_optimizer.step()
        self.backbone_optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()

        return outputs

    def cal_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the ZeroSiam loss for a batch of features."""
        # Target branch: original classifier prediction with stop-gradient.
        target_logits = self.model.head(features).detach()

        # Online branch: the same feature passes through the learnable predictor
        # before the frozen classifier head, creating the asymmetric branch.
        online_logits = self.get_online_logits(features)

        entropy_loss = softmax_entropy(online_logits).mean(0)
        alignment_loss = symmetric_kl_divergence(online_logits, target_logits).mean(0)

        return entropy_loss + self.alignment_weight * alignment_loss

    def get_online_logits(self, features: torch.Tensor) -> torch.Tensor:
        predicted_features = self.model.predictor(features)
        return self.model.head(predicted_features)

    def reset(self):
        if self.model_state is None or self.backbone_optimizer_state is None or self.predictor_optimizer_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")

        load_model_and_optimizer(self.model, self.backbone_optimizer, self.predictor_optimizer, self.model_state, self.backbone_optimizer_state, self.predictor_optimizer_state)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the softmax distribution from logits."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def symmetric_kl_divergence(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between online and stop-gradient targets.

    The paper writes the alignment term as a generic divergence. This
    implementation uses the same symmetric KL form as the original code.
    """
    online_to_target = F.kl_div(outputs.log_softmax(dim=1), targets.softmax(dim=1), reduction="none").sum(1)
    target_to_online = F.kl_div(targets.log_softmax(dim=1), outputs.softmax(dim=1), reduction="none").sum(1)
    return online_to_target + target_to_online


def collect_params(model):
    """Collect normalization affine parameters adapted at test time."""
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for param_name, param in module.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def copy_model_and_optimizer(model, optimizer1, optimizer2):
    """Copy model and optimizer states for episodic reset."""
    model_state = deepcopy(model.state_dict())
    optimizer_state1 = deepcopy(optimizer1.state_dict())
    optimizer_state2 = deepcopy(optimizer2.state_dict())
    return model_state, optimizer_state1, optimizer_state2


def load_model_and_optimizer(model, optimizer1, optimizer2, model_state, optimizer_state1, optimizer_state2):
    """Restore model and optimizer states from saved copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer1.load_state_dict(optimizer_state1)
    optimizer2.load_state_dict(optimizer_state2)


def configure_model(model):
    """Configure a wrapped model for ZeroSiam adaptation.

    Only the lightweight predictor and normalization affine parameters are
    trainable. BatchNorm layers use batch statistics during adaptation, matching
    the common Tent/TTA setup.
    """
    model.train()
    model.requires_grad_(False)

    if hasattr(model, "predictor"):
        model.predictor.requires_grad_(True)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            module.requires_grad_(True)

    return model


def check_model(model):
    """Check that the model is configured for test-time adaptation."""
    assert model.training, "ZeroSiam needs train mode: call model.train()"

    param_grads = [param.requires_grad for param in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "ZeroSiam needs params to update"
    assert not has_all_params, "ZeroSiam should not update all params"

    has_norm = any(isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for module in model.modules())
    assert has_norm, "ZeroSiam needs normalization parameters for adaptation"
