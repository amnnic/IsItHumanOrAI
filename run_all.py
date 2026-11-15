"""Satu kali jalan: siapkan data -> fine-tune ResNet50 -> fine-tune ViT -> evaluasi.

Jalankan dari folder skripsi-nicole-v7:  python run_all.py
Langkah yang hasilnya sudah ada otomatis dilewati, jadi kalau terputus cukup jalankan lagi.
Paksa ulang semua langkah: python run_all.py --force
"""
import os
import subprocess
import sys
import time

PY = sys.executable
FORCE = "--force" in sys.argv

STEPS = [
    ("1/4 Siapkan data buatan sendiri", [PY, "prep_custom.py"], ["data/custom/train.csv", "data/custom/test.csv", "data/custom/all.csv"]),
    ("2/4 Fine-tune ResNet50-BERT", [PY, "finetune.py", "--backbone", "resnet"], ["ckpt/resnet_bert_custom_e3.pt"]),
    ("3/4 Fine-tune ViT-BERT", [PY, "finetune.py", "--backbone", "vit"], ["ckpt/vit_bert_custom_e3.pt"]),
    ("4/4 Evaluasi", [PY, "evaluate_v7.py"], ["results_v7.json"]),
]

t_all = time.time()
for name, cmd, outs in STEPS:
    if not FORCE and all(os.path.exists(o) for o in outs):
        print(f"\n=== {name}: DILEWATI (hasil sudah ada) ===", flush=True)
        continue
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"\nLangkah '{name}' gagal (kode {r.returncode}). Perbaiki, lalu jalankan python run_all.py lagi; langkah yang sudah selesai dilewati.")
    print(f"--- selesai dalam {(time.time() - t0) / 60:.1f} menit", flush=True)

print(f"\nSEMUA SELESAI dalam {(time.time() - t_all) / 60:.1f} menit. Hasil: report_v7/summary.csv dan results_v7.json")
