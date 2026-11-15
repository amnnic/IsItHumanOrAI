import base64
import html
import inspect
import io

import streamlit as st
from PIL import Image

import inference as inf

st.set_page_config(page_title="Is It Human or AI?", page_icon="🤖", layout="wide")

STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

MODELS = ["ViT-BERT", "ResNet50-BERT"]
ARCH = {"ViT-BERT": "ViT", "ResNet50-BERT": "ResNet50"}
COMBO = {"ViT-BERT": "ViT + BERT", "ResNet50-BERT": "ResNet50 + BERT"}
STATS = inf.load_metrics()
DS_LABEL = inf.DATASETS
DS_KEYS = list(DS_LABEL)

NOTE_BASE = "Confidence adalah probabilitas softmax model, bukan jaminan kebenaran. "
NOTE_BY_DS = {
    "mirage": ("MiRAGeNews saja: dilatih hanya pada berita berbahasa Inggris MiRAGeNews; pada gambar/teks dari sumber lain "
               "(Fakeddit, gambar buatan sendiri) kemampuannya turun."),
    "custom": ("MiRAGeNews + Fakeddit + dataset sendiri: dilatih pada gabungan MiRAGeNews dan Fakeddit lalu di-fine-tune dengan "
               "foto berita asli dan gambar AI buatan sendiri. Data sendiri kecil (±100 gambar uji), jadi angkanya hanya perkiraan kasar."),
}
NOTE_METRIC = (
    "MiRAGeNews OOD = rata-rata test2–test5. Fakeddit = test Fakeddit. Data sendiri = test dataset sendiri (±100 gambar, "
    "pembagian per kelompok). Hasil dari satu kali pelatihan."
)


def stat(ds, model, mode, key):
    try:
        return STATS[ds][model][mode][key]
    except (TypeError, KeyError):
        return None

def fmt_pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"

def fmt_f1(x):
    return "-" if x is None else f"{x:.3f}"

def f1_avg_pct(ds, model, mode):
    v = stat(ds, model, mode, "avg_f1")
    return None if v is None else v * 100

def metric_cells(ds, model, mode="fuse"):
    g = lambda k: stat(ds, model, mode, k)
    return {
        "Akurasi (MiRAGeNews OOD)": fmt_pct(g("m_acc")),
        "Akurasi (Fakeddit test)": fmt_pct(g("f_acc")),
        "Akurasi (data sendiri)": fmt_pct(g("c_acc")),
        "F1 (MiRAGeNews OOD)": fmt_f1(g("m_f1")),
        "F1 (Fakeddit test)": fmt_f1(g("f_f1")),
        "F1 (data sendiri)": fmt_f1(g("c_f1")),
    }

def text_model(ds):
    return max(MODELS, key=lambda m: stat(ds, m, "txt", "avg_f1") or 0.0)

def lc_first(s):
    return s[:1].lower() + s[1:]

LOC = "tiga data uji (rata-rata MiRAGeNews OOD, Fakeddit test, dan data sendiri)"

def compare_sentence(noun, a_name, fa, b_name, fb, kedua):
    if fa is None or fb is None:
        return "Metrik evaluasi belum tersedia (file compare_results.json / results_v7.json tidak ditemukan)."
    text = (f"{noun} {a_name} memiliki F1-Score rata-rata {fa:.2f}% dan {lc_first(noun)} {b_name} "
            f"memiliki F1-Score rata-rata {fb:.2f}% pada {LOC}. ")
    d = abs(fa - fb)
    if d < 1.0:
        return text + f"Selisihnya hanya {d:.2f} poin, sehingga {kedua} dianggap sebanding."
    win = a_name if fa > fb else b_name
    return text + f"Dapat disimpulkan bahwa {lc_first(noun)} {win} lebih unggul dengan selisih {d:.2f} poin."

def evaluasi(ds, ms, mode, noun, disp, kedua):
    f = {m: f1_avg_pct(ds, m, mode) for m in ms}
    if len(ms) == 2:
        a, b = ms
        return compare_sentence(noun, disp[a], f[a], disp[b], f[b], kedua)
    m = ms[0]
    if f[m] is None:
        return "Metrik evaluasi belum tersedia."
    return f"{noun} {disp[m]} memiliki F1-Score rata-rata {f[m]:.2f}% pada {LOC}."

def agreement(preds: dict) -> str:
    if len(preds) < 2:
        return ""
    labels = {p["label"] for p in preds.values()}
    if len(labels) == 1:
        return f" Kedua model sepakat: input ini diprediksi {labels.pop()}."
    parts = ", ".join(f"{n}: {p['label']}" for n, p in preds.items())
    return f" Kedua model tidak sepakat pada input ini ({parts})."

def agree_line(prefix: str, preds: dict) -> str:
    if len(preds) < 2:
        return ""
    labels = {p["label"] for p in preds.values()}
    if len(labels) == 1:
        return f" {prefix}: kedua model sepakat, diprediksi {labels.pop()}."
    parts = ", ".join(f"{n}: {p['label']}" for n, p in preds.items())
    return f" {prefix}: kedua model tidak sepakat ({parts})."


DEFAULTS = {"dss": list(DS_KEYS), "page": "home", "image_bytes": None, "text": "", "result": None}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def go(page: str):
    st.session_state.page = page

def run_analysis():
    img = st.session_state.get("uploader")
    txt = (st.session_state.get("text_input") or "").strip()
    img_bytes = img.getvalue() if img is not None else None
    if img_bytes is None and not txt:
        st.session_state.error = "Upload gambar atau isi teks terlebih dahulu."
        return
    dss = [k for k in DS_KEYS if st.session_state.get(f"use_{k}", True)]
    if not dss:
        st.session_state.error = "Pilih minimal satu dataset."
        return
    st.session_state.pop("error", None)
    st.session_state.dss = dss
    st.session_state.image_bytes = img_bytes
    st.session_state.text = txt
    st.session_state.result = None
    go("result")

def models_for(ds, img_bytes, txt):
    return [text_model(ds)] if (txt and not img_bytes) else list(MODELS)

def analyze(img_bytes, txt, dss):
    res = {"mode": "both" if (img_bytes and txt) else ("image" if img_bytes else "text"), "ds": {}}
    for ds in dss:  # satu dataset selesai dulu, supaya memori GPU hanya dipakai satu dataset
        ms = models_for(ds, img_bytes, txt)
        if res["mode"] == "both":
            r = {m: inf.predict_all(txt, img_bytes, m, ds) for m in ms}
        elif res["mode"] == "image":
            r = {m: inf.predict_image(img_bytes, m, ds) for m in ms}
        else:
            r = {m: inf.predict_text(txt, m, ds) for m in ms}
        res["ds"][ds] = {"models": ms, "pred": r}
    return res


def title(hero=False):
    cls = "title hero" if hero else "title"
    st.markdown(f'<div class="{cls}">Is It Human or AI?</div>', unsafe_allow_html=True)

def navbar(show_analyze=True):
    left, right = st.columns([1, 1])
    with left:
        st.button("←Home", key="nav_home", on_click=go, args=("home",))
    with right:
        if show_analyze:
            st.button("Analyze→", key="nav_analyze", on_click=go, args=("analyze",))

def card(label: str, body: str, left=False):
    cls = "card-body left" if left else "card-body"
    st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="{cls}">{body}</div></div>',
                unsafe_allow_html=True)

def note(text: str, left=False):
    style = ' style="text-align:left"' if left else ""
    st.markdown(f'<div class="note"{style}>{html.escape(text)}</div>', unsafe_allow_html=True)

def img_tag(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((900, 600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" alt="Gambar yang dianalisis">'

def pred_card(label: str, pred: dict):
    pct = pred["confidence"] * 100
    st.markdown(
        f'<div class="card pred-card"><div class="card-label" style="text-align:left">{label}</div>'
        f'<div class="pred-label">{pred["label"]}</div>'
        f'<div class="conf-text">Confidence: {pct:.1f}%</div>'
        f'<div class="conf-bar"><span style="width:{pct:.1f}%"></span></div></div>',
        unsafe_allow_html=True)

def html_table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="cmp-table"><tr>{head}</tr>{body}</table>'

def ds_header(ds):
    st.markdown(f'<div class="ds-head"><span>Model dilatih pada</span><b>{DS_LABEL[ds]}</b></div>',
                unsafe_allow_html=True)


def page_home():
    title(hero=True)
    st.markdown(
        '<div class="hero-desc">Our multimodal AI technology analyzes both images and text to identify '
        "whether digital content is Human-Generated or AI-Generated. Pilih satu atau dua model dataset, "
        "lalu bandingkan hasil Vision Transformer-BERT dan ResNet50-BERT.</div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.button("Get started", key="pill_start", on_click=go, args=("analyze",), **STRETCH)
    st.markdown(
        '<div class="how"><div class="how-title">How It Works</div>'
        "<p><b>STEP 1 : Upload</b><br>Upload an image, write a text (English), or provide both.</p>"
        "<p><b>STEP 2 : Pilih dataset</b><br>Pilih model MiRAGeNews saja, model gabungan, atau keduanya sekaligus.</p>"
        "<p><b>STEP 3 : Get Results</b><br>Lihat prediksi dan confidence di setiap dataset.</p></div>",
        unsafe_allow_html=True)

def page_analyze():
    navbar(show_analyze=False)
    title()
    _, mid, _ = st.columns([1, 2.6, 1])
    with mid:
        with st.container(key="upload_card"):
            up = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "webp"], key="uploader")
            if up is not None:
                st.image(up, **STRETCH)
        st.write("")
        with st.container(key="text_card"):
            st.text_area("Input teks (bahasa Inggris)", key="text_input", height=110,
                         placeholder="Tulis atau tempel teks berbahasa Inggris di sini…")
        if len(DS_KEYS) > 1:
            st.markdown('<div class="mode-hint" style="margin-bottom:.2rem">Model dilatih pada dataset (boleh pilih keduanya)</div>',
                        unsafe_allow_html=True)
            cols = st.columns(len(DS_KEYS))
            for c, k in zip(cols, DS_KEYS):
                with c:
                    st.checkbox(DS_LABEL[k], value=True, key=f"use_{k}")
        has_img = st.session_state.get("uploader") is not None
        has_txt = bool((st.session_state.get("text_input") or "").strip())
        if has_img and has_txt:
            hint = "Mode: gambar + teks (membandingkan ResNet50-BERT dan ViT-BERT)"
        elif has_img:
            hint = "Mode: gambar saja (membandingkan ResNet50 dan ViT)"
        elif has_txt:
            hint = "Mode: teks saja (BERT)"
        else:
            hint = "Isi gambar, teks, atau keduanya."
        st.markdown(f'<div class="mode-hint">{hint}</div>', unsafe_allow_html=True)
        if "error" in st.session_state:
            st.warning(st.session_state.error)
        _, b, _ = st.columns([1, 1, 1])
        with b:
            st.button("Analyze", key="pill_analyze", on_click=run_analysis, **STRETCH)

def render_ds(ds, mode, block, text, img_bytes):
    ms, pred = block["models"], block["pred"]
    if len(st.session_state.dss) > 1 or len(DS_KEYS) > 1:
        ds_header(ds)
    if mode == "text":
        left, right = st.columns([1.8, 1])
        with left:
            card("Teks", text)
        with right:
            for m in ms:
                pred_card("Prediksi Teks (BERT)", pred[m])
        body = (evaluasi(ds, ms, "txt", "Model", {m: "BERT" for m in MODELS}, "kedua cabang teks")
                + agreement(pred))
    elif mode == "image":
        left, right = st.columns([1.8, 1])
        with left:
            card("Gambar", img_tag(img_bytes))
        with right:
            for m in ms:
                pred_card(f"Prediksi {ARCH[m]}", pred[m])
        body = (evaluasi(ds, ms, "img", "Arsitektur", ARCH, "kedua arsitektur")
                + agreement(pred))
    else:  # 6 cards: (gambar, teks, gabungan) x (ResNet50-BERT, ViT-BERT)
        order = sorted(ms, key=lambda m: m != "ResNet50-BERT")
        left, right = st.columns([1, 2.2])
        with left:
            card("Gambar & Teks", img_tag(img_bytes) + text)
        with right:
            for m in order:
                c1, c2, c3 = st.columns(3)
                with c1:
                    pred_card(f"Prediksi Gambar ({ARCH[m]})", pred[m]["img"])
                with c2:
                    pred_card(f"Prediksi Teks ({m})", pred[m]["txt"])
                with c3:
                    pred_card(f"Prediksi {m} (gabungan)", pred[m]["fuse"])
        body = (evaluasi(ds, ms, "img", "Arsitektur", ARCH, "kedua arsitektur")
                + agree_line("Gambar", {ARCH[m]: pred[m]["img"] for m in order}) + "<br><br>"
                + evaluasi(ds, ms, "txt", "Cabang teks", {m: m for m in MODELS}, "kedua cabang teks")
                + agree_line("Teks", {m: pred[m]["txt"] for m in order}) + "<br><br>"
                + evaluasi(ds, ms, "fuse", "Kombinasi", COMBO, "kedua kombinasi")
                + agree_line("Gabungan", {m: pred[m]["fuse"] for m in order}))
    _, mid, _ = st.columns([0.3, 3, 0.3])
    with mid:
        card("Evaluasi", body)
        note(NOTE_BASE + NOTE_BY_DS[ds])

def page_result():
    navbar()
    title()
    dss = st.session_state.dss
    if st.session_state.result is None:
        if not (st.session_state.image_bytes or st.session_state.text):
            go("analyze")
            st.rerun()
        with st.spinner("Menganalisis… memuat model pertama kali bisa memakan waktu."):
            try:
                st.session_state.result = analyze(st.session_state.image_bytes, st.session_state.text, dss)
            except Exception as e:
                st.error(f"Analisis gagal: {e}")
                st.stop()
    res = st.session_state.result
    mode = res["mode"]
    text = html.escape(st.session_state.text).replace("\n", "<br>")
    for ds in dss:
        render_ds(ds, mode, res["ds"][ds], text, st.session_state.image_bytes)
    if mode == "both":
        _, b, _ = st.columns([1, 1, 1])
        with b:
            st.button("Lihat Perbandingan", key="pill_compare", on_click=go, args=("compare",), **STRETCH)

def page_compare():
    res = st.session_state.result
    if not res or res["mode"] != "both":
        go("analyze")
        st.rerun()
    navbar()
    title()
    for ds in st.session_state.dss:
        ms = res["ds"][ds]["models"]
        boxes = ""
        for n in ms:
            c = res["ds"][ds]["pred"][n]
            grid = "".join(f"<div><small>{k}</small><b>{v}</b></div>" for k, v in metric_cells(ds, n).items())
            boxes += (
                f'<div class="model-box"><div class="model-name">{n}</div><div class="metric-grid">{grid}</div>'
                f'<div class="mini-pred">Prediksi gabungan: <b>{c["fuse"]["label"]}</b> '
                f'({c["fuse"]["confidence"] * 100:.1f}%)<br>'
                f'Gambar saja: {c["img"]["label"]} ({c["img"]["confidence"] * 100:.1f}%) · '
                f'Teks saja: {c["txt"]["label"]} ({c["txt"]["confidence"] * 100:.1f}%)</div></div>')
        cells = {n: metric_cells(ds, n) for n in ms}
        table = html_table([f"Metrik (gabungan)"] + ms, [[k] + [cells[n][k] for n in ms] for k in cells[ms[0]]])
        st.markdown(
            f'<div class="card"><div class="card-label">Perbandingan — {DS_LABEL[ds]}</div>'
            f'<div class="model-grid">{boxes}</div>{table}</div>', unsafe_allow_html=True)
    dss = st.session_state.dss
    if len(dss) == 2:  # ringkasan lintas dataset
        rows = []
        for m in MODELS:
            row = [m]
            for ds in dss:
                p = res["ds"][ds]["pred"][m]["fuse"]
                row.append(f'{p["label"]} ({p["confidence"] * 100:.1f}%)')
            rows.append(row)
        st.markdown(
            '<div class="card"><div class="card-label">Prediksi gabungan: antar dataset</div>'
            + html_table(["Model"] + [DS_LABEL[d] for d in dss], rows) + "</div>", unsafe_allow_html=True)
    note(NOTE_METRIC)
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        with st.container(key="back"):
            st.button("← Kembali ke hasil", key="back_btn", on_click=go, args=("result",), **STRETCH)

PAGES = {"home": page_home, "analyze": page_analyze, "result": page_result, "compare": page_compare}
PAGES[st.session_state.page]()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Lora:wght@400;500;600&family=Space+Grotesk:wght@400;500&family=Noto+Sans:wght@400;600;700&display=swap');
 
:root {
  --bg: #EEEFF5;
  --card: #FFFFFF;
  --ink: #111111;
  --muted: #8A8A8A;
  --pill: #FFDDDD;
}
 
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: var(--bg); }
.block-container { max-width: 980px; padding-top: 1.2rem; padding-bottom: 3rem; }
 
/* Judul gradasi */
.title {
  font-family: 'Abril Fatface', serif;
  font-size: clamp(2.2rem, 6vw, 3.4rem);
  text-align: center;
  margin: 0.6rem 0 1.8rem 0;
  line-height: 1.15;
  background: linear-gradient(90deg, #FF3DF0 0%, #F2C2D2 38%, #9B5CF0 70%, #5A1FFF 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.title.hero { font-size: clamp(2.6rem, 7vw, 4rem); margin-top: 4rem; }
 
/* Card putih */
.card {
  background: var(--card);
  border-radius: 14px;
  padding: 1rem 1.4rem 1.4rem 1.4rem;
  box-shadow: 0 2px 10px rgba(30, 30, 60, 0.05);
  margin-bottom: 1.1rem;
}
.card-label {
  font-family: 'Noto Sans', sans-serif;
  font-size: 0.68rem;
  color: var(--muted);
  margin-bottom: 0.8rem;
}
.card-body {
  font-family: 'Lora', serif;
  font-size: 0.95rem;
  line-height: 1.7;
  color: var(--ink);
  text-align: center;
}
.card-body.left { text-align: left; }
.card img { display: block; margin: 0 auto 0.8rem auto; max-width: 100%; max-height: 260px; border-radius: 6px; }
 
/* Kartu prediksi */
.pred-card { text-align: center; padding-bottom: 1.2rem; }
.pred-label { font-family: 'Noto Sans', sans-serif; font-weight: 700; font-size: 1.3rem; color: var(--ink); margin: 0.4rem 0 0.6rem 0; }
.conf-text { font-family: 'Noto Sans', sans-serif; font-size: 0.78rem; color: #555; margin-bottom: 0.4rem; }
.conf-bar { height: 6px; background: #ECECF3; border-radius: 99px; overflow: hidden; }
.conf-bar > span { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #FF3DF0, #5A1FFF); }
 
/* Nav atas */
[class*="st-key-nav_"] button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  font-family: 'Abril Fatface', serif !important; font-size: 1.05rem !important;
  color: var(--ink) !important; padding: 0 !important;
}
[class*="st-key-nav_"] button:hover { opacity: .6; }
.st-key-nav_analyze { align-self: flex-end !important; margin-left: auto !important; width: fit-content !important; }
[data-testid="stColumn"]:has(.st-key-nav_analyze) { display: flex !important; flex-direction: column !important; align-items: flex-end !important; }
[data-testid="stColumn"]:has(.st-key-nav_analyze) > [data-testid="stVerticalBlock"] { align-items: flex-end !important; width: 100%; }
 
/* Tombol */
.stButton > button, [class*="st-key-pill_"] button {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; color: var(--ink);
  background: var(--pill); border: 1px solid var(--ink); border-radius: 999px;
  padding: 0.55rem 2.4rem; transition: transform .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); border-color: var(--ink); color: var(--ink); background: #FFD0D0; }
.stButton > button:focus-visible { outline: 2px solid #5A1FFF; outline-offset: 2px; }
.st-key-back { margin: 1.6rem 0 2.5rem 0; padding-top: 1.4rem; border-top: 1px solid #DCDCE6; }
.st-key-back [data-testid="stButton"], .st-key-back .stButton { width: 100%; }
.st-key-back button {
  width: 100% !important; min-height: 3.4rem; padding: 0.7rem 2rem !important;
  background: var(--ink) !important; color: #FFFFFF !important;
  border: 1px solid var(--ink) !important; border-radius: 999px !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; letter-spacing: .01em;
  box-shadow: 0 4px 14px rgba(17, 17, 17, 0.18);
  transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.st-key-back button p { color: #FFFFFF !important; font-size: 1.1rem !important; margin: 0; }
.st-key-back button:hover {
  background: #5A1FFF !important; border-color: #5A1FFF !important; color: #FFFFFF !important;
  transform: translateY(-2px); box-shadow: 0 6px 18px rgba(90, 31, 255, 0.28);
}
.st-key-back button:active { transform: translateY(0); }
.st-key-back button:focus-visible { outline: 2px solid #5A1FFF; outline-offset: 3px; }
 
/* Form analyze */
.st-key-upload_card, .st-key-text_card {
  background: var(--card); border-radius: 14px; padding: 1rem 1.4rem 1.2rem 1.4rem;
  box-shadow: 0 2px 10px rgba(30, 30, 60, 0.05); border: none;
}
[data-testid="stFileUploaderDropzone"] { background: #FAFAFC; border: 1.5px dashed #CFCFD8; border-radius: 10px; min-height: 140px; }

.stTextArea textarea { font-family: 'Lora', serif; background: #FAFAFC; border-radius: 10px; }
.stTextArea label, [data-testid="stFileUploader"] label { font-family: 'Noto Sans', sans-serif; font-size: .68rem !important; color: var(--muted) !important; }
.mode-hint { text-align: center; font-family: 'Noto Sans', sans-serif; font-size: .8rem; color: #666; margin: .6rem 0 1rem 0; }
 
/* Tabel perbandingan */
.model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem; }
.model-box { background: #FBFAF8; border-radius: 10px; padding: .9rem 1rem; }
.model-name { font-family: 'Noto Sans', sans-serif; font-weight: 600; font-size: .85rem; margin-bottom: .6rem; color: var(--ink); }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem 1rem; }
.metric-grid small { display: block; font-family: 'Noto Sans', sans-serif; color: var(--muted); font-size: .68rem; }
.metric-grid b { font-family: 'Noto Sans', sans-serif; font-weight: 500; font-size: 1.2rem; color: var(--ink); }
.mini-pred { margin-top: .8rem; font-family: 'Noto Sans', sans-serif; font-size: .75rem; color: #555; }
.mini-pred b { color: var(--ink); }
.cmp-table { width: 100%; border-collapse: collapse; font-family: 'Noto Sans', sans-serif; font-size: .82rem; }
.cmp-table th { text-align: right; font-weight: 400; color: #555; padding: .5rem .3rem; border-bottom: 1px solid #333; }
.cmp-table td { text-align: right; padding: .55rem .3rem; border-bottom: 1px solid #E6E6EC; color: var(--ink); background: transparent; }
.cmp-table th:first-child, .cmp-table td:first-child { text-align: left; font-weight: 600; }
.cmp-table tr:last-child td { border-bottom: none; }
 
/* Panduan di home */
.how { max-width: 440px; margin: 2.2rem auto 0 auto; background: #fff; border-radius: 12px; padding: 1rem 1.3rem; box-shadow: 0 2px 10px rgba(30,30,60,.05); }
.how-title { font-family: 'Noto Sans', sans-serif; font-size: .7rem; color: #555; margin-bottom: .6rem; }
.how p { font-family: 'Lora', serif; font-size: .82rem; margin: .1rem 0 .55rem 0; line-height: 1.5; color: var(--ink); }
.how b { color: var(--ink); }
.hero-desc { color: var(--ink); font-family: 'Lora', serif; font-size: .95rem; line-height: 1.7; text-align: center; max-width: 560px; margin: 0 auto 1.6rem auto; }
 
 
/* font hitam */
.card .model-name, .card .metric-grid b, .card .mini-pred b,
.card .cmp-table td, .card .cmp-table th:first-child,
.how, .how p, .how b, .hero-desc { color: #111111 !important; }
.card .metric-grid small { color: #8A8A8A !important; }
.card .mini-pred, .card .cmp-table th { color: #555555 !important; }
 
 
/* Input teks */
.stTextArea textarea, .stTextArea [data-baseweb="textarea"] textarea, .stTextArea [data-baseweb="base-input"] {
  color: #111111 !important; -webkit-text-fill-color: #111111 !important; caret-color: #111111 !important;
  background: #FAFAFC !important;
}
.stTextArea textarea::placeholder { color: #9A9AA5 !important; -webkit-text-fill-color: #9A9AA5 !important; opacity: 1; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] { background: #FAFAFC !important; border-color: #CFCFD8 !important; }
.stTextArea [data-baseweb="textarea"]:focus-within { border-color: #5A1FFF !important; box-shadow: 0 0 0 1px #5A1FFF; }
/* Nama file yang di-upload */
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color: #111111 !important; }
 
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

/* Catatan kecil */
.note { font-family: 'Noto Sans', sans-serif; font-size: .72rem; color: #777; text-align: center; margin: .2rem 0 1rem 0; line-height: 1.5; }

/* Uploader: teks & tombol rapi (ikon Material tidak ikut diganti font-nya) */
[data-testid="stFileUploaderDropzoneInstructions"] span:not([data-testid="stIconMaterial"]),
[data-testid="stFileUploaderDropzoneInstructions"] small { font-family: 'Noto Sans', sans-serif; color: #777; }
[data-testid="stFileUploaderDropzone"] button {
  font-family: 'Space Grotesk', sans-serif; background: var(--pill); color: var(--ink);
  border: 1px solid var(--ink); border-radius: 999px; padding: .35rem 1.2rem;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: #FFD0D0; border-color: var(--ink); color: var(--ink); }

/* Kartu tambahan: pilihan model, kontrol & grafik halaman evaluasi */
.st-key-model_card, .st-key-eval_controls, [class*="st-key-chart_"] {
  background: var(--card); border-radius: 14px; padding: 1rem 1.4rem 1.2rem 1.4rem;
  box-shadow: 0 2px 10px rgba(30, 30, 60, 0.05); border: none; margin-bottom: 1.1rem;
}
.ds-head { display:flex; gap:.6rem; align-items:baseline; justify-content:center; margin: 1.6rem 0 .8rem 0;
  font-family:'Noto Sans',sans-serif; font-size:.8rem; color:#666; }
.ds-head b { font-family:'Abril Fatface',serif; font-weight:400; font-size:1.25rem; color:#111; }
.stCheckbox label p { font-family:'Noto Sans',sans-serif; font-size:.85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
