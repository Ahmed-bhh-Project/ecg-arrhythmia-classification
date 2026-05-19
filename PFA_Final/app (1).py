from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter

from model import ECGCNN

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/beats", exist_ok=True)

# === Chargement du modèle ===
device = torch.device("cpu")
model = ECGCNN(num_classes=8)
checkpoint = torch.load("ECGCNN_model.pt", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# === Classes d'arythmie ===
classes = ['APC', 'LBBB', 'NOR', 'PAB', 'PVC', 'RBBB', 'VEB', 'VFE']

# === Transformations d'image ===
img_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === Fonction d'analyse de signal CSV ===
def segmentation_from_csv(csv_path, output_dir='static/beats/'):
    os.makedirs(output_dir, exist_ok=True)
    signal = np.loadtxt(csv_path)
    signal = preprocessing.scale(np.nan_to_num(signal))

    peaks, _ = find_peaks(signal, distance=100, prominence=0.6)
    window = 96
    abnormal_beats = []
    anomaly_counts = Counter()

    bpm = int((len(peaks) / 1800) * 60)

    for idx, r in enumerate(peaks):
        if r - window < 0 or r + window >= len(signal):
            continue

        segment = signal[r - window:r + window]
        if np.ptp(segment) < 0.3 or np.std(segment) > 1.5 or len(segment) != 192:
            continue

        fig, ax = plt.subplots(figsize=(2, 2), dpi=64)
        ax.plot(segment, linewidth=0.5)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert('L')
        img = img.resize((128, 128))
        tensor = img_transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()
            pred_class = classes[pred]

        if pred_class != "NOR":
            filename = f"{uuid.uuid4().hex}.png"
            img.save(os.path.join(output_dir, filename))
            abnormal_beats.append({
                "filename": filename,
                "class": pred_class,
                "original_index": idx + 1,
                "signal_position": r
            })
            anomaly_counts[pred_class] += 1

    return abnormal_beats, bpm, dict(anomaly_counts)

# === ROUTES FLASK ===

@app.route("/")
def home():
    return redirect(url_for("about"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/info")
def info():
    return render_template("info.html", code=None)



@app.route("/upload")
def upload():
    return render_template("index.html")

@app.route("/upload_csv", methods=["GET", "POST"])
def upload_csv():
    if request.method == "GET":
        return render_template("upload_csv.html")

    f = request.files["file"]
    filepath = os.path.join("uploads", f.filename)
    f.save(filepath)

    results, bpm, summary = segmentation_from_csv(filepath)
    if not results:
        return "✅ Ce patient est tout à fait normal. Aucun battement anormal détecté."

    np.save("static/beats/results.npy", results)
    return render_template("viewer.html", index=0, total=len(results), beat=results[0], bpm=bpm, summary=summary)

@app.route("/next/<int:index>")
def next_beat(index):
    results = np.load("static/beats/results.npy", allow_pickle=True).tolist()
    if index >= len(results):
        return "Fin de la séquence."

    bpm = int((results[-1]['original_index'] / 1800) * 60)
    summary = {}
    for beat in results:
        summary[beat["class"]] = summary.get(beat["class"], 0) + 1

    return render_template("viewer.html", index=index, total=len(results), beat=results[index], bpm=bpm, summary=summary)

@app.route("/predict", methods=["POST"])
def predict_image():
    f = request.files["file"]
    filepath = os.path.join("uploads", f.filename)
    f.save(filepath)

    img = Image.open(filepath).convert('RGB')
    tensor = img_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        pred_class = classes[pred]
        confidence = float(prob[0][pred].item())

    full_names = {
        'APC': "Contraction Auriculaire Prématurée",
        'LBBB': "Bloc de Branche Gauche",
        'NOR': "Rythme Normal",
        'PAB': "Bloc Auriculaire Prématuré",
        'PVC': "Contraction Ventriculaire Prématurée",
        'RBBB': "Bloc de Branche Droit",
        'VEB': "Battement Ectopique Ventriculaire",
        'VFE': "Fibrillation Ventriculaire"
    }

    return jsonify({
        "abbreviation": pred_class,
        "full_name": full_names[pred_class],
        "probability": f"{confidence*100:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
