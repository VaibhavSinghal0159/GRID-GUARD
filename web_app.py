"""
Localhost web demo for Handwritten Prescription Recognition.

Reuses the model and CTC decoder defined in the notebook and the trained
checkpoint in runs/public_sources_seed42/best_model.pt.

Usage:  python web_app.py          then open http://127.0.0.1:5000
"""
import ast, io, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from flask import Flask, jsonify, request, send_file, abort
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'runs' / 'public_sources_seed42' / 'best_model.pt'
METRICS = ROOT / 'runs' / 'public_sources_seed42' / 'test_metrics.json'
RX_ROOT = ROOT / 'data' / 'raw' / 'rxhandbd'

# Load model/decoder definitions straight from the notebook (same as demo.py)
nb = json.loads((ROOT / 'ML_Project (1).ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        selected = [n for n in ast.parse(''.join(cell['source'])).body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and n.name in {'TinyVGGCRNN', 'ctc_decode', 'edit_distance', 'recognition_metrics'}]
        if selected:
            exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook>', 'exec'), globals())

checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
chars = checkpoint['chars']
unk_idx = checkpoint['unk_idx']
blank_idx = checkpoint['blank_idx']
idx_to_char = {i: ch for i, ch in enumerate(chars)}
idx_to_char[unk_idx] = '�'
model = TinyVGGCRNN(len(chars) + 2, hidden_size=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.ToTensor()
metrics = json.loads(METRICS.read_text(encoding='utf-8'))

# RxHandBD test samples (filename -> label)
rx_df = pd.read_csv(RX_ROOT / 'Test_Labels.csv', keep_default_na=False)
rx_cols = {c.lower().strip(): c for c in rx_df.columns}
img_col = next((rx_cols[k] for k in ['images', 'image', 'image_id', 'filename'] if k in rx_cols), rx_df.columns[0])
lbl_col = next((rx_cols[k] for k in ['labels', 'label', 'text', 'transcription'] if k in rx_cols), rx_df.columns[1])
rx_files = {p.name: p for p in RX_ROOT.rglob('*.jpg')}
samples = []
for _, row in rx_df.iterrows():
    name = Path(str(row[img_col]).strip()).name
    name = name if name.lower().endswith('.jpg') else name + '.jpg'
    if name in rx_files:
        samples.append({'id': name, 'label': str(row[lbl_col])})


def predict(img: Image.Image) -> str:
    img = img.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        return ctc_decode(model(tensor))[0]


app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Prescription OCR</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222}
h1{margin-bottom:.2rem} .muted{color:#666;font-size:.9rem}
.card{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
.ok{color:#2E8B57;font-weight:600} .bad{color:#CC3333;font-weight:600}
img{max-width:100%;border:1px solid #eee;background:#fff}
button{padding:.5rem 1rem;margin-right:.5rem;cursor:pointer}
code{background:#f4f4f4;padding:2px 6px;border-radius:4px}
</style></head><body>
<h1>Handwritten Prescription Recognition</h1>
<p class="muted">Model: TinyVGG-CRNN, CTC decoding. Trained on BD Prescription + RxHandBD (public releases).</p>
<div class="card" id="metrics"></div>
<div class="card">
  <h3>Try a real test sample</h3>
  <button onclick="randomSample()">Random RxHandBD test image</button>
  <div id="sample"></div>
</div>
<div class="card">
  <h3>Upload your own word image</h3>
  <input type="file" id="file" accept="image/*">
  <button onclick="uploadImage()">Recognize</button>
  <div id="upload"></div>
</div>
<script>
fetch('/api/metrics').then(r=>r.json()).then(m=>{
  document.getElementById('metrics').innerHTML =
    '<b>Test set (' + m.samples + ' images)</b><br>' +
    'Exact match: <b>' + (m.exact_match_accuracy*100).toFixed(2) + '%</b> &nbsp; ' +
    'CER: <b>' + (m.cer*100).toFixed(2) + '%</b> &nbsp; WER: <b>' + (m.wer*100).toFixed(2) + '%</b>';
});
function show(el, d){
  const ok = d.label === null ? null : d.prediction === d.label;
  const cls = ok === null ? '' : (ok ? 'ok' : 'bad');
  el.innerHTML = '<img src="' + d.image_url + '"><p>Prediction: <code>' + d.prediction + '</code>' +
    (d.label !== null ? ' &nbsp; Reference: <code>' + d.label + '</code> &nbsp; <span class="' + cls + '">' + (ok ? 'exact match' : 'mismatch') + '</span>' : '') + '</p>';
}
function randomSample(){
  fetch('/api/random').then(r=>r.json()).then(d=>show(document.getElementById('sample'), d));
}
function uploadImage(){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('Choose an image first'); return; }
  const fd = new FormData(); fd.append('image', f);
  fetch('/api/predict', {method:'POST', body: fd}).then(r=>r.json()).then(d=>{
    d.image_url = URL.createObjectURL(f); d.label = null;
    show(document.getElementById('upload'), d);
  });
}
</script></body></html>"""


@app.route('/')
def index():
    return PAGE


@app.route('/api/metrics')
def api_metrics():
    return jsonify(metrics)


@app.route('/api/random')
def api_random():
    s = random.choice(samples)
    return jsonify({
        'image_url': f"/image/{s['id']}",
        'label': s['label'],
        'prediction': predict(Image.open(rx_files[s['id']])),
    })


@app.route('/image/<name>')
def image(name):
    if name not in rx_files:
        abort(404)
    return send_file(rx_files[name], mimetype='image/jpeg')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    f = request.files.get('image')
    if f is None:
        abort(400)
    return jsonify({'prediction': predict(Image.open(io.BytesIO(f.read())))})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
