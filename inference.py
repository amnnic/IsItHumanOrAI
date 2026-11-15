import gc
import io
import json
import os

import streamlit as st
import torch
from PIL import Image

from models import MultiModalDetector
from utils import FAKE_IDX, MAX_LEN, get_transform, tokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KIND = {"ViT-BERT": "vit", "ResNet50-BERT": "resnet"}
LABEL_AI, LABEL_HUMAN = "AI-Generated", "Human-Generated"
HERE = os.path.dirname(os.path.abspath(__file__))

ENABLED = ("custom",)
ALL_DATASETS = {"mirage": "MiRAGeNews saja", "custom": "MiRAGeNews + Fakeddit + dataset sendiri"}
DATASETS = {k: v for k, v in ALL_DATASETS.items() if k in ENABLED}

CKPT_FILE = {
    ("mirage", "vit"): "vit_bert_e10.pt",
    ("mirage", "resnet"): "resnet_bert_e10.pt",
    ("custom", "vit"): "vit_bert_custom_e3.pt",
    ("custom", "resnet"): "resnet_bert_custom_e3.pt",
}
# Cari di ckpt/ folder ini dulu; kalau tidak ada, cari di folder v1 / v7 yang bersebelahan.
SIBLING = {"mirage": "skripsi-nicole-v1", "custom": "skripsi-nicole-v7"}


def ckpt_path(ds, kind):
    name = CKPT_FILE[(ds, kind)]
    for p in (os.path.join(HERE, "ckpt", name),
              os.path.join(HERE, "..", SIBLING[ds], "ckpt", name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Checkpoint {name} tidak ditemukan (taruh di folder ckpt/).")


@st.cache_resource(show_spinner=False)
def _store() -> dict:
    return {}


def load_model(kind: str, ds: str):
    store = _store()
    for key in [k for k in store if k[0] != ds]:  # GPU 4 GB: simpan model satu dataset saja
        del store[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if (ds, kind) not in store:
        model = MultiModalDetector(kind)
        state = torch.load(ckpt_path(ds, kind), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        store[(ds, kind)] = model.to(DEV).eval()
    return store[(ds, kind)]


def _pred(logits) -> dict:
    p_ai = logits.float().softmax(-1)[0, FAKE_IDX].item()
    is_ai = p_ai >= 0.5
    return {"label": LABEL_AI if is_ai else LABEL_HUMAN,
            "confidence": p_ai if is_ai else 1.0 - p_ai, "p_ai": p_ai}


@torch.no_grad()
def _run(kind, ds, image_bytes=None, text=None) -> dict:
    model = load_model(kind, ds)
    px = ids = mask = None
    if image_bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        px = get_transform(kind)(img).unsqueeze(0).to(DEV)
    if text:
        enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids, mask = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
        return model(px, ids, mask)


def predict_text(text, model_name, ds):
    return _pred(_run(KIND[model_name], ds, text=text)["txt"])


def predict_image(image_bytes, model_name, ds):
    return _pred(_run(KIND[model_name], ds, image_bytes=image_bytes)["img"])


def predict_all(text, image_bytes, model_name, ds):
    out = _run(KIND[model_name], ds, image_bytes=image_bytes, text=text)
    return {"img": _pred(out["img"]), "txt": _pred(out["txt"]), "fuse": _pred(out["fuse"])}


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _agg(sets, mode):
    """sets: dict nama-set -> hasil. Kembalikan (acc, f1) rata-rata mode tsb."""
    return (sum(s[mode]["acc"] for s in sets) / len(sets), sum(s[mode]["f1"] for s in sets) / len(sets))


def load_metrics():
    """Metrik per dataset latih -> model -> mode:
       m_* = MiRAGeNews OOD (rata-rata test2-test5), f_* = Fakeddit test, c_* = test dataset sendiri."""
    cmp_, v7 = _load("compare_results.json"), _load("results_v7.json")
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for name, kind in KIND.items():
            if ds == "mirage":
                a = (cmp_ or {}).get(f"mirage__{kind}")
                b = (v7 or {}).get(f"mirage__{kind}")
                if a is None or b is None:
                    return None
                ood = [a[s] for s in a if s.startswith("mirage/test")]
                fk, cu = a["fakeddit/test"], b["custom/test"]
            else:
                b = (v7 or {}).get(f"v7__{kind}")
                if b is None:
                    return None
                ood = [b[s] for s in b if s.startswith("mirage/test")]
                fk, cu = b["fakeddit/test"], b["custom/test"]
            out[ds][name] = {}
            for mode in ("img", "txt", "fuse"):
                m_acc, m_f1 = _agg(ood, mode)
                out[ds][name][mode] = {
                    "m_acc": m_acc, "m_f1": m_f1, "f_acc": fk[mode]["acc"], "f_f1": fk[mode]["f1"],
                    "c_acc": cu[mode]["acc"], "c_f1": cu[mode]["f1"],
                    "avg_f1": (m_f1 + fk[mode]["f1"] + cu[mode]["f1"]) / 3,
                }
    return out
