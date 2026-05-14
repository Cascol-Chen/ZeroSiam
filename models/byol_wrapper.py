import torch.nn as nn


class BYOLWrapper(nn.Module):
    """Wrap a classifier with the lightweight predictor used by ZeroSiam.

    The wrapper exposes two things ZeroSiam needs: a penultimate feature from
    one backbone pass, and a ``head`` method that maps either original or
    predictor-transformed features back to logits.
    """

    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

        # ZeroSiam inserts the predictor in feature space, so its dimension must
        # match the classifier input dimension for each backbone family.
        self.project_dim = self._infer_projector_dim()
        self.predictor = self._build_predictor(self.project_dim)

        device = next(self.model.parameters()).device
        self.to(device)

    def _infer_projector_dim(self):
        if "vit" in self.model_name or "swin" in self.model_name:
            return self.model.head.weight.data.size(1)
        if "convnext" in self.model_name:
            return self.model.head.fc.weight.data.size(1)
        return self.model.fc.weight.data.size(1)

    def _build_predictor(self, dim):
        predictor = nn.Sequential(nn.Linear(dim, dim, bias=False))
        # Identity initialization gives ZeroSiam a warm start for fully TTA
        nn.init.eye_(predictor[0].weight)
        return predictor

    def forward(self, x):
        features, outputs = self._extract_features_and_logits(x)
        return outputs, self._pool_features(features)

    def _extract_features_and_logits(self, x):
        if self.model_name == "resnet50_bn_torch":
            outputs, features = self.model(x, return_feature=True)
            return features, outputs

        features = self.model.forward_features(x)
        outputs = self.model.forward_head(features)
        return features, outputs

    def _pool_features(self, features):
        # Convert feature tensors into the vector expected by head(...).
        if "vit" in self.model_name:
            return features[:, 0]
        if "convnext" in self.model_name:
            features = self.model.head.global_pool(features)
            features = self.model.head.norm(features)
            return self.model.head.flatten(features)
        if "swin" in self.model_name and self.model.global_pool == "avg":
            return features.mean(dim=1)
        if self.model_name == "resnet50_gn_timm":
            return self.model.global_pool(features)
        return features

    def head(self, x):
        if "vit" in self.model_name or "swin" in self.model_name:
            return self.model.head(x)
        if "convnext" in self.model_name:
            return self.model.head.fc(x)
        return self.model.fc(x)
