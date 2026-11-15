Menambah dataset buatan sendiri (200 foto berita asli + 199 gambar buatan AI) ke model MiRAGeNews + Fakeddit

Prasyarat: folder skripsi-nicole-v1, -v2, -v3, -v6 sejajar dengan folder ini; v6 sudah berisi ckpt/ dan data/.
Catatan: nama zip tertukar dengan isinya. dataset_generatesendiri_AI.zip = foto berita ASLI,
dataset_generatesendiri_Asli.zip = gambar BUATAN AI. prep_custom.py sudah memetakan ini.

Cara cepat: python run_all.py (menjalankan semua langkah di bawah berurutan, langkah yang sudah ada hasilnya dilewati)

Urutan manual:
1. python prep_custom.py                       -> data/custom/{train,test,all}.csv (pembagian per berita, tanpa kebocoran)
2. python finetune.py --backbone resnet        -> ckpt/resnet_bert_custom_e3.pt   (sekitar 10-20 menit)
   python finetune.py --backbone vit           -> ckpt/vit_bert_custom_e3.pt      (sekitar 30-45 menit)
3. python evaluate_v7.py                       -> results_v7.json, report_v7/summary.csv
