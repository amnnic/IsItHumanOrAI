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
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from models import MultiModalDetector
from train import MODES, build_optim, lr_lambdas, seed_all
from utils import HFData, LocalData, collect, make_loader, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["vit", "resnet"], required=True)
    ap.add_argument("--v6", default="../skripsi-nicole-v6", help="folder hasil MiRAGeNews + Fakeddit")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-scale", type=float, default=0.3)
    ap.add_argument("--repeat", type=int, default=3, help="data buatan sendiri diulang berapa kali per epoch")
    ap.add_argument("--replay", type=int, default=1000, help="jumlah data lama per dataset yang ikut dilatih")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--tag", default="custom_e3")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seed_all(a.seed)
    rng = random.Random(a.seed)
    dev, kind = "cuda", a.backbone
    os.makedirs("ckpt", exist_ok=True)
    base = os.path.join(a.v6, "ckpt", f"{kind}_bert_mirage_fakeddit_e10.pt")
    best_path = f"ckpt/{kind}_bert_{a.tag}.pt"
    hist_path = f"ckpt/history_{kind}_{a.tag}.json"
    fdir = os.path.join(a.v6, "data", "fakeddit")

    sub = json.load(open(os.path.join(a.v6, "data", "mirage_subset.json")))
    mir = load_dataset("anson-huang/mirage-news")
    custom = LocalData("data/custom", "train", kind, train=True)
    m_ids = rng.sample(sub["train"], min(a.replay, len(sub["train"])))
    fak = LocalData(fdir, "train_5k", kind, train=True)
    f_ids = rng.sample(range(len(fak)), min(a.replay, len(fak)))
    train_ds = ConcatDataset([custom] * a.repeat + [HFData(mir["train"].select(m_ids), kind, train=True), Subset(fak, f_ids)])
    tr = make_loader(train_ds, kind, a.bs, train=True)
    va_m = make_loader(HFData(mir["validation"].select(sub["validation"]), kind), kind, a.bs * 2)
    va_f = make_loader(LocalData(fdir, "validation_1250", kind), kind, a.bs * 2)
    print(f"Data latih: {len(train_ds)} = {len(custom)} buatan sendiri x{a.repeat} + {len(m_ids)} MiRAGeNews + {len(f_ids)} Fakeddit")

    model = MultiModalDetector(kind).to(dev)
    model.load_state_dict(torch.load(base, map_location=dev, weights_only=True))
    print("Mulai dari:", base)
    opt, tags = build_optim(model, kind)
    for g in opt.param_groups:
        g["lr"] *= a.lr_scale
    total = math.ceil(len(tr) / a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambdas(tags, kind, total, int(0.1 * total)))
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()

    hist = []
    for ep in range(a.epochs):
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
        yf, Pf = collect(model, va_f, dev)
        res = {"epoch": ep + 1, "loss": run, "minutes": round((time.time() - t0) / 60, 1),
               "val_mirage": {m: score(ym, Pm[m]) for m in MODES}, "val_fakeddit": {m: score(yf, Pf[m]) for m in MODES}}
        hist.append(res)
        print(f"[ep {ep + 1}] val fuse acc (pantau lupa): MiRAGeNews {res['val_mirage']['fuse']['acc']:.4f} | "
              f"Fakeddit {res['val_fakeddit']['fuse']['acc']:.4f}  ({res['minutes']} menit)")
        json.dump(hist, open(hist_path, "w", encoding="utf-8"), indent=1)
    torch.save(model.state_dict(), best_path)
    print("Selesai. Checkpoint epoch terakhir ->", best_path)


if __name__ == "__main__":
    main()
