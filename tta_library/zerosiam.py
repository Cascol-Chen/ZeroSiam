# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import torch.autograd
import torch.linalg
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import math
from models.byol_wrapper import BYOLWrapper
from collections import defaultdict
# from utils.LogME import LogME

def update_ema(ema, new_data):
    with torch.no_grad():
        if ema is None:
            return new_data
        else:
            return 0.9 * ema + (1 - 0.9) * new_data

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model:BYOLWrapper, backbone_optimizer, predictor_optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.backbone_optimizer = backbone_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.backbone_optimizer_state, self.predictor_optimizer_state = \
            copy_model_and_optimizer(self.model, self.backbone_optimizer, self.predictor_optimizer)
        self.use_ent = True
                
        
    def forward(self, x, y=None):
        if self.episodic:
            self.reset()

        for i in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.backbone_optimizer, y)
        return outputs

    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self,x, model, optimizer, y):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        final_outputs, features = self.model(x)
        loss = self.cal_loss(features)
        loss.backward()
        self.backbone_optimizer.step()
        self.backbone_optimizer.zero_grad()
        self.predictor_optimizer.step()
        self.predictor_optimizer.zero_grad()
        return final_outputs
    
    
    def cal_loss(self, features:torch.Tensor):
        clean_outputs = self.model.head(features).detach()
        loss = 0
        noisy_output = self.get_noisy_outputs(features)
        loss += 1 * plain_entropy(noisy_output).mean(0)
        loss += 1 * get_loss_js(noisy_output, clean_outputs).mean(0)
        return loss
        
    def get_noisy_outputs(self, features):
        features = self.model.predictor(features)
        return self.model.head(features)
    

    def reset(self):
        if self.model_state is None or self.backbone_optimizer_state is None or self.predictor_optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        
        load_predictor_and_optimizer(self.model, self.predictor_optimizer, self.predictor_optimizer_state)


criterion_kl = nn.KLDivLoss(reduction='none').cuda()
def get_loss_js(outputs, targets):
    loss = criterion_kl(outputs.log_softmax(dim=1), targets.softmax(dim=1)).sum(1) + criterion_kl(targets.log_softmax(dim=1), outputs.softmax(dim=1)).sum(1)
    return loss

def plain_entropy(x:torch.Tensor):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer1, optimizer2):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state1 = deepcopy(optimizer1.state_dict())
    optimizer_state2 = deepcopy(optimizer2.state_dict())
    return model_state, optimizer_state1, optimizer_state2


def load_model_and_optimizer(model, optimizer1, optimizer2, model_state, optimizer_state1, optimizer_state2):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer1.load_state_dict(optimizer_state1)
    optimizer2.load_state_dict(optimizer_state2)

def load_predictor_and_optimizer(model, optimizer, optimizer_state):
    """Restore the model and optimizer states from copies."""
    nn.init.eye_(model.predictor.weight)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    if hasattr(model, 'predictor'):
        model.predictor.requires_grad_(True)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"