import argparse
import csv
import json
import os

import numpy as np
import torch
from datasets import load_dataset

from models import MultiModalDetector
from utils import LocalData, collect, make_loader, score

KINDS = ["vit", "resnet"]
KNAME = {"vit": "ViT-BERT", "resnet": "ResNet50-BERT"}
MODES = ["img", "txt", "fuse"]
MNAME = {"img": "Gambar", "txt": "Teks", "fuse": "Gambar+Teks"}
MIR_OOD = ["test2_bbc_dalle", "test3_cnn_dalle", "test4_bbc_sdxl", "test5_cnn_sdxl"]


def models(root):
    r = lambda p: os.path.join(root, p)
    return {
        "mirage": ("MiRAGeNews saja", r("skripsi-nicole-v1/ckpt/{k}_bert_e10.pt"), False),
        "defactify": ("Defactify saja", r("skripsi-nicole-v2/ckpt/{k}_bert_defactify_e10.pt"), False),
        "mir_def": ("MiRAGeNews + Defactify", r("skripsi-nicole-v3/ckpt/{k}_bert_combined_e10.pt"), False),
        "mir_fake": ("MiRAGeNews + Fakeddit (awal)", r("skripsi-nicole-v6/ckpt/{k}_bert_mirage_fakeddit_e10.pt"), True),
        "v7": ("MiRAGeNews + Fakeddit + buatan sendiri", "ckpt/{k}_bert_custom_e3.pt", True),
    }


def run(root, bs, workers):
    R = {}
    arrays = {}
    mir = load_dataset("anson-huang/mirage-news")
    fdir = os.path.join(root, "skripsi-nicole-v6", "data", "fakeddit")
    for kind in KINDS:
        sets = {"custom/test": make_loader(LocalData("data/custom", "test", kind), kind, bs, workers=workers),
                "custom/semua": make_loader(LocalData("data/custom", "all", kind), kind, bs, workers=workers)}
        extra = {f"mirage/{s}": make_loader(mir[s], kind, bs, workers=workers) for s in MIR_OOD}
        extra["fakeddit/test"] = make_loader(LocalData(fdir, "test", kind), kind, bs, workers=workers)
        for sc, (name, tmpl, with_extra) in models(root).items():
            path = tmpl.format(k=kind)
            key = f"{sc}__{kind}"
            if not os.path.exists(path):
                print("LEWATI (checkpoint tidak ada):", path)
                continue
            model = MultiModalDetector(kind).cuda()
            model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            R[key] = {}
            todo = dict(sets)
            if sc == "v7":
                todo.pop("custom/semua")
            if with_extra:
                todo.update(extra)
            for sname, loader in todo.items():
                y, P = collect(model, loader)
                R[key][sname] = {m: score(y, P[m]) for m in MODES}
                tag = sname.replace("/", "_")
                arrays[f"{key}__{tag}__y"] = y.astype(np.int8)
                for m in MODES:
                    arrays[f"{key}__{tag}__{m}"] = P[m].astype(np.float32)
                print(f"{key:18s} {sname:24s} " + "  ".join(f"{m}={R[key][sname][m]['acc']:.4f}" for m in MODES), flush=True)
            del model
            torch.cuda.empty_cache()
    json.dump(R, open("results_v7.json", "w", encoding="utf-8"), indent=1)
    np.savez_compressed("preds_v7.npz", **arrays)
    return R


def report(R, root):
    os.makedirs("report_v7", exist_ok=True)
    names = {k: v[0] for k, v in models(root).items()}
    rows = []
    for key, r in R.items():
        sc, kind = key.split("__")
        for m in MODES:
            row = {"skenario": names[sc], "model": KNAME[kind], "mode": MNAME[m]}
            for sname in ("custom/test", "custom/semua"):
                for metric in ("acc", "f1"):
                    row[f"{metric} {sname}"] = r[sname][m][metric] if sname in r else ""
            if "fakeddit/test" in r:
                row["acc MiRAGeNews OOD"] = float(np.mean([r[f"mirage/{s}"][m]["acc"] for s in MIR_OOD]))
                row["acc Fakeddit test"] = r["fakeddit/test"][m]["acc"]
            rows.append(row)
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open("report_v7/summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.4f}" if isinstance(v, float) else v) for c, v in r.items()})
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "  -  "
    print("\nHASIL (Gambar+Teks). custom/test = bagian uji dataset buatan sendiri (tidak dipakai latih)")
    print(f"{'Skenario':40s} {'Model':14s} {'custom/test':>11} {'custom/semua':>12} {'MirOOD':>7} {'Fakedd':>7}")
    for r in rows:
        if r["mode"] == "Gambar+Teks":
            print(f"{r['skenario']:40s} {r['model']:14s} {fmt(r['acc custom/test']):>11} {fmt(r['acc custom/semua']):>12} "
                  f"{fmt(r.get('acc MiRAGeNews OOD')):>7} {fmt(r.get('acc Fakeddit test')):>7}")
    print("\nTersimpan: results_v7.json, preds_v7.npz, report_v7/summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.getcwd(), "..")))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    R = json.load(open("results_v7.json", encoding="utf-8")) if a.report_only else run(a.root, a.bs, a.workers)
    report(R, a.root)


if __name__ == "__main__":
    main()
