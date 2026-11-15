import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from transformers import BertModel, ViTModel

N_TRAIN = 4


class ImageEncoder(nn.Module):
    def __init__(self, kind, n_train=N_TRAIN):
        super().__init__()
        self.kind = kind
        if kind == "vit":
            self.net = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
            self.dim = 768
            for p in self.net.parameters():
                p.requires_grad = False
            for blk in self.net.encoder.layer[-n_train:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.net.layernorm.parameters():
                p.requires_grad = True
        else:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            m.fc = nn.Identity()
            self.net, self.dim = m, 2048
            for p in m.parameters():
                p.requires_grad = False
            for layer in (m.layer3, m.layer4):
                for mod in layer.modules():
                    if isinstance(mod, nn.Conv2d):
                        mod.weight.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.kind == "resnet":
            for mod in self.net.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        return self

    def forward(self, x):
        if self.kind == "vit":
            return self.net(pixel_values=x).last_hidden_state[:, 0]
        return self.net(x)


class MultiModalDetector(nn.Module):
    def __init__(self, kind, n_train=N_TRAIN, hidden=512, p=0.3):
        super().__init__()
        self.img = ImageEncoder(kind, n_train)
        self.txt = BertModel.from_pretrained("google-bert/bert-base-uncased")
        for q in self.txt.parameters():
            q.requires_grad = False
        for blk in self.txt.encoder.layer[-n_train:]:
            for q in blk.parameters():
                q.requires_grad = True
        for q in self.txt.pooler.parameters():
            q.requires_grad = True

        d_i, d_t = self.img.dim, 768
        self.head_img = nn.Sequential(nn.Dropout(p), nn.Linear(d_i, 2))
        self.head_txt = nn.Sequential(nn.Dropout(p), nn.Linear(d_t, 2))
        self.head_fuse = nn.Sequential(nn.Linear(d_i + d_t, hidden), nn.ReLU(),
                                       nn.Dropout(p), nn.Linear(hidden, 2))

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        out, fi, ft = {}, None, None
        if pixel_values is not None:
            fi = self.img(pixel_values)
            out["img"] = self.head_img(fi)
        if input_ids is not None:
            ft = self.txt(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            out["txt"] = self.head_txt(ft)
        if fi is not None and ft is not None:
            out["fuse"] = self.head_fuse(torch.cat([fi, ft], dim=-1))
        return out
