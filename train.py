import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from models import MultiModalDetector
from utils import HFData, LocalData, collect, make_loader, score

DATA = "data/fakeddit"
MODES = ("img", "txt", "fuse")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_optim(model, kind):
    lr_img = 1e-4 if kind == "resnet" else 2e-5
    groups = {"img": (model.img, lr_img), "txt": (model.txt, 2e-5),
              "head": (nn.ModuleList([model.head_img, model.head_txt, model.head_fuse]), 1e-3)}
    pg, tags = [], []
    for name, (mod, lr) in groups.items():
        decay, no_decay = [], []
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
        if decay:
            pg.append({"params": decay, "lr": lr, "weight_decay": 0.01}); tags.append(name)
        if no_decay:
            pg.append({"params": no_decay, "lr": lr, "weight_decay": 0.0}); tags.append(name)
    return torch.optim.AdamW(pg), tags


def lr_lambdas(tags, kind, total, warm):
    def make(cosine):
        def f(step):
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * prog)) if cosine else max(0.0, 1 - prog)
        return f
    return [make(cosine=(kind == "resnet" and t == "img")) for t in tags]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["vit", "resnet"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--tag", default="mirage_fakeddit_e10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    seed_all(a.seed)
    dev, kind = "cuda", a.backbone
    os.makedirs("ckpt", exist_ok=True)
    best_path = f"ckpt/{kind}_bert_{a.tag}.pt"
    last_path = f"ckpt/last_{kind}_{a.tag}.pt"
    hist_path = f"ckpt/history_{kind}_{a.tag}.json"

    sub = json.load(open("data/mirage_subset.json"))
    mir = load_dataset("anson-huang/mirage-news")
    train_ds = ConcatDataset([HFData(mir["train"].select(sub["train"]), kind, train=True),
                              LocalData(DATA, "train_5k", kind, train=True)])
    tr = make_loader(train_ds, kind, a.bs, train=True)
    va_m = make_loader(HFData(mir["validation"].select(sub["validation"]), kind), kind, a.bs * 2)
    va_d = make_loader(LocalData(DATA, "validation_1250", kind), kind, a.bs * 2)
    print(f"Data latih: {len(train_ds)} (5000 MiRAGeNews + 5000 Fakeddit)")

    model = MultiModalDetector(kind).to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_tr / 1e6:.1f}M  (harus 58.1M untuk vit, 52.4M untuk resnet)")

    opt, tags = build_optim(model, kind)
    total = math.ceil(len(tr) / a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambdas(tags, kind, total, int(0.1 * total)))
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()

    start, best, hist = 0, -1.0, []
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, hist = ck["epoch"], ck["best"], ck["hist"]
        print(f"Resume dari epoch {start}, best {best:.4f}")

    for ep in range(start, a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(tr, desc=f"epoch {ep + 1}/{a.epochs}")
        for i, (imgs, ids, mask, y) in enumerate(bar):
            imgs, ids, mask, y = imgs.to(dev), ids.to(dev), mask.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(imgs, ids, mask)
                loss = (ce(out["img"], y) + ce(out["txt"], y) + ce(out["fuse"], y)) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0 or (i + 1) == len(tr):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run = (0.98 * run + 0.02 * loss.item() * a.accum) if i else loss.item() * a.accum
            bar.set_postfix(loss=f"{run:.3f}")

        ym, Pm = collect(model, va_m, dev)
        yd, Pd = collect(model, va_d, dev)
        y_all = np.concatenate([ym, yd])
        P_all = {m: np.concatenate([Pm[m], Pd[m]]) for m in MODES}
        res = {"epoch": ep + 1, "loss": run, "minutes": round((time.time() - t0) / 60, 1)}
        res.update({m: score(y_all, P_all[m]) for m in MODES})
        res["val_mirage"] = {m: score(ym, Pm[m]) for m in MODES}
        res["val_fakeddit"] = {m: score(yd, Pd[m]) for m in MODES}
        hist.append(res)
        print(f"[ep {ep + 1}] val fuse acc: gabungan {res['fuse']['acc']:.4f} | "
              f"MiRAGeNews {res['val_mirage']['fuse']['acc']:.4f} | Fakeddit {res['val_fakeddit']['fuse']['acc']:.4f}"
              f"  ({res['minutes']} menit)")
        if res["fuse"]["acc"] > best:
            best = res["fuse"]["acc"]
            torch.save(model.state_dict(), best_path)
            print("  -> checkpoint terbaik disimpan")
        json.dump(hist, open(hist_path, "w", encoding="utf-8"), indent=1)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best, "hist": hist}, last_path)

    print(f"Selesai. Best fuse val acc (gabungan) = {best:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
