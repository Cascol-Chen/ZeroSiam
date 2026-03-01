import torch
import torch.nn as nn

class DiagPredictor(nn.Module):
    def __init__(self, dim, init_identity=True):
        super().__init__()
        # 可学习缩放向量，相当于对角元素
        init = torch.ones(dim) if init_identity else torch.randn(dim)*0.01
        self.scale = nn.Parameter(init)          # shape: [D]

    def forward(self, z):                        # z: [B,D]
        return z * self.scale                   # 按维度逐元素缩放


class BYOLWrapper(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name
        if 'vit' in self.model_name or 'swin' in self.model_name:
            projector_dim = self.model.head.weight.data.size(1)
        elif 'convnext' in self.model_name:
            projector_dim = self.model.head.fc.weight.data.size(1)
        else:
            projector_dim = self.model.fc.weight.data.size(1)

        # self.predictor = nn.Sequential(
        #     *[nn.Linear(projector_dim, projector_dim, bias=False) for _ in range(1)],
        # ).cuda()
        # self.predictor = nn.Sequential(
        #     nn.Linear(projector_dim, projector_dim, bias=False),
        # ).cuda()

        # self.scale = nn.Parameter(torch.zeros(1))
        self.predictor = nn.Sequential(
            nn.Linear(projector_dim, projector_dim, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(projector_dim // 4, projector_dim, bias=False),
        ).cuda()
        # self.predictor = nn.Linear(projector_dim, projector_dim, bias=False).cuda()
        for i, m in enumerate(self.predictor.modules()):
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.eye_(m.weight)
                # m.weight.data = torch.eye(m.weight.data.shape[0], m.weight.data.shape[1]).cuda()
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # nn.init.zeros_(self.predictor[0].weight)

        # self.predictor.weight.data += 0.01 * torch.randn_like(self.predictor.weight.data)

        self.project_dim = projector_dim
        self.cuda()

    def forward(self, x):
        if self.model_name == 'resnet50_bn_torch':
            outputs, prev_features = self.model(x, return_feature=True)
        else:
            prev_features = self.model.forward_features(x)
            outputs = self.model.forward_head(prev_features)
        
        if 'vit' in self.model_name:
            prev_features = prev_features[:, 0]
        elif 'convnext' in self.model_name:
            prev_features = self.model.head.global_pool(prev_features)
            prev_features = self.model.head.norm(prev_features)
            prev_features = self.model.head.flatten(prev_features)
        elif 'swin' in self.model_name:
            if self.model.global_pool == 'avg':
                prev_features = prev_features.mean(dim=1)
        elif self.model_name == 'resnet50_bn_torch': # ResNet_bn
            prev_features = prev_features
        elif self.model_name == 'resnet50_gn_timm': # ResNet_gn
            prev_features = self.model.global_pool(prev_features)
        return outputs, prev_features
    
    def head(self, x):
        if 'vit' in self.model_name:
            return self.model.head(x)
        elif 'convnext' in self.model_name:
            return self.model.head.fc(x)
        elif 'swin' in self.model_name:
            return self.model.head(x)
        else:
            # print(x.shape, self.model.fc.weight.shape)
            return self.model.fc(x)