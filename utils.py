import csv
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoTokenizer

FAKE_IDX = 1
MAX_LEN = 128
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

_STATS = {"vit": ([0.5] * 3, [0.5] * 3),
          "resnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}


def get_transform(kind, train=False):
    mean, std = _STATS[kind]
    ops = [T.Resize((224, 224))]
    if train:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def _detect_columns(ds):
    from datasets import Image as HFImage, Value
    feats = ds.features
    img = next(k for k, v in feats.items() if isinstance(v, HFImage))
    prefer = [k for k in feats if k.lower() in ("text", "caption", "title", "article")]
    strs = [k for k, v in feats.items() if isinstance(v, Value) and v.dtype == "string"]
    txt = (prefer or strs)[0]
    lab = next(k for k in feats if k.lower() == "label")
    return img, txt, lab


class HFData(Dataset):
    def __init__(self, ds, kind, train=False):
        self.ds, self.tf = ds, get_transform(kind, train)
        self.ci, self.ct, self.cl = _detect_columns(ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return self.tf(r[self.ci].convert("RGB")), r[self.ct] or "", int(r[self.cl])


class LocalData(Dataset):
    def __init__(self, folder, split, kind, train=False):
        self.folder, self.tf = folder, get_transform(kind, train)
        with open(os.path.join(folder, f"{split}.csv"), encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.folder, r["path"])).convert("RGB")
        return self.tf(img), r["text"] or "", int(r["label"])


def _collate(batch):
    imgs, txts, ys = zip(*batch)
    enc = tokenizer(list(txts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return torch.stack(imgs), enc["input_ids"], enc["attention_mask"], torch.tensor(ys)


def make_loader(data, kind, bs=16, train=False, workers=2):
    if not isinstance(data, Dataset):
        data = HFData(data, kind, train)
    return DataLoader(data, batch_size=bs, shuffle=train, num_workers=workers,
                      collate_fn=_collate, pin_memory=torch.cuda.is_available(),
                      persistent_workers=(train and workers > 0), drop_last=False)


@torch.no_grad()
def collect(model, loader, dev="cuda"):
    import numpy as np
    model.eval()
    ys, P = [], {"img": [], "txt": [], "fuse": []}
    for imgs, ids, mask, y in loader:
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            out = model(imgs.to(dev), ids.to(dev), mask.to(dev))
        for k in P:
            P[k].append(out[k].float().softmax(-1)[:, FAKE_IDX].cpu().numpy())
        ys.append((y == FAKE_IDX).long().numpy())
    return np.concatenate(ys), {k: np.concatenate(v) for k, v in P.items()}


def score(y, p):
    from sklearn.metrics import f1_score, roc_auc_score
    pred = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    return {"acc": float((pred == y).mean()), "f1": float(f1_score(y, pred, zero_division=0)), "auc": auc}
