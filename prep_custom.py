import argparse
import csv
import html
import io
import os
import random
import re
import zipfile

from PIL import Image

OUT = "data/custom"


def parse(zpath):
    zf = zipfile.ZipFile(zpath)
    htm = next(n for n in zf.namelist() if n.lower().endswith(".html"))
    t = zf.read(htm).decode("utf-8", "ignore")
    items = []
    for row in re.findall(r"<tr.*?</tr>", t, flags=re.S):
        cells = re.findall(r"<td.*?</td>", row, flags=re.S)
        if len(cells) != 2:
            continue
        src = re.findall(r'src="(images/[^"]+)"', cells[0])
        txt = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cells[1]))).strip()
        if src and txt:
            items.append((src[0], txt))
    return zf, items


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def save_img(zf, name, dst):
    img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
    img.thumbnail((320, 320))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img.save(dst, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-zip", default="dataset_generatesendiri_AI.zip")
    ap.add_argument("--ai-zip", default="dataset_generatesendiri_Asli.zip")
    ap.add_argument("--test-frac", type=float, default=0.25)
    a = ap.parse_args()

    zr, real = parse(a.real_zip)
    za, ai = parse(a.ai_zip)
    print(f"Asli: {len(real)} pasang gambar+teks | AI: {len(ai)} pasang gambar+teks")

    rows = []
    for i, (src, txt) in enumerate(real):
        p = f"images/real_{i:03d}.jpg"
        save_img(zr, src, os.path.join(OUT, p))
        rows.append({"path": p, "text": txt, "label": 0, "gen": 0, "group": i})
    rw = [words(t) for _, t in real]
    nxt, paired = len(real), 0
    for j, (src, txt) in enumerate(ai):
        p = f"images/ai_{j:03d}.jpg"
        save_img(za, src, os.path.join(OUT, p))
        w = words(txt)
        sims = [jaccard(w, x) for x in rw]
        best = max(range(len(sims)), key=sims.__getitem__)
        if sims[best] >= 0.25:
            g, paired = best, paired + 1
        else:
            g, nxt = nxt, nxt + 1
        rows.append({"path": p, "text": txt, "label": 1, "gen": 1, "group": g})
    print(f"AI yang berhasil dipasangkan dengan berita asli: {paired} dari {len(ai)} (sisanya grup sendiri)")

    groups = sorted({r["group"] for r in rows})
    rng = random.Random(42)
    rng.shuffle(groups)
    n_test = round(len(groups) * a.test_frac)
    test_g = set(groups[:n_test])
    os.makedirs(OUT, exist_ok=True)
    for name, sel in (("train", [r for r in rows if r["group"] not in test_g]),
                      ("test", [r for r in rows if r["group"] in test_g]),
                      ("all", rows)):
        rng.shuffle(sel)
        with open(os.path.join(OUT, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "text", "label", "gen", "group"])
            w.writeheader()
            w.writerows(sel)
        print(f"{name}: {len(sel)} baris ({sum(r['label'] for r in sel)} AI / {sum(1 - r['label'] for r in sel)} asli)")
    print("Selesai. Folder:", OUT)


if __name__ == "__main__":
    main()
