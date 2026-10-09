"""Image + crop metadata CNN, with TensorFlow and PyTorch training paths."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from .common import directory, metadata

SIZE = 64

def image(raw):
    data = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if data is None or data.shape[0] * data.shape[1] > 20_000_000:
        raise ValueError("Provide a valid image below 20 megapixels.")
    return cv2.resize(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), (SIZE, SIZE)).astype("float32") / 255

def crop_features(crop, humidity, temperature, crops):
    if crop not in crops or not 0 <= humidity <= 100 or not -10 <= temperature <= 60:
        raise ValueError("Unsupported crop or metadata range.")
    return np.asarray([float(crop == c) for c in crops] + [humidity/100, temperature/60], dtype="float32")

def torch_model(meta_size, classes):
    import torch
    from torch import nn
    class CropCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(24+meta_size, 24), nn.ReLU(), nn.Linear(24, classes))
        def forward(self, pixels, meta):
            return self.head(torch.cat([self.vision(pixels), meta], dim=1))
    return CropCNN()

def train(manifest, epochs=8, framework="tensorflow", synthetic=False):
    frame = pd.read_csv(manifest)
    if len(frame) < 20 or frame["label"].value_counts().min() < 4:
        raise ValueError("Provide at least 20 images and four examples per disease class.")
    crops, labels = sorted(frame.crop.unique().tolist()), sorted(frame.label.unique().tolist())
    if len(labels) < 2:
        raise ValueError("At least two classes are required.")
    pixels = np.stack([image((Path(manifest).parent / p).read_bytes()) for p in frame.path])
    meta = np.stack([crop_features(r.crop, float(r.humidity), float(r.temperature), crops) for r in frame.itertuples()])
    y = np.asarray([labels.index(v) for v in frame.label])
    tr, te = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    path = directory("agri")
    if framework == "tensorflow":
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        vision = tf.keras.Input((SIZE, SIZE, 3), name="image")
        crop = tf.keras.Input((len(crops)+2,), name="metadata")
        z = tf.keras.layers.Conv2D(12, 3, activation="relu")(vision)
        z = tf.keras.layers.MaxPooling2D()(z)
        z = tf.keras.layers.Conv2D(24, 3, activation="relu")(z)
        z = tf.keras.layers.GlobalAveragePooling2D()(z)
        z = tf.keras.layers.Concatenate()([z, crop])
        z = tf.keras.layers.Dense(24, activation="relu")(z)
        model = tf.keras.Model([vision, crop], tf.keras.layers.Dense(len(labels), activation="softmax")(z))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit([pixels[tr], meta[tr]], y[tr], epochs=epochs, verbose=0)
        pred = model([pixels[te], meta[te]], training=False).numpy().argmax(1)
        model.save(path / "model.keras")
    elif framework == "pytorch":
        import torch
        torch.manual_seed(42)
        model = torch_model(len(crops)+2, len(labels))
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        x = torch.tensor(pixels.transpose(0, 3, 1, 2))
        m, targets = torch.tensor(meta), torch.tensor(y, dtype=torch.long)
        for _ in range(epochs):
            for indices in np.array_split(tr, max(1, len(tr)//32)):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x[indices], m[indices]), targets[indices])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x[te], m[te]).argmax(1).numpy()
        torch.save(model.state_dict(), path / "model.pt")
    else:
        raise ValueError("framework must be tensorflow or pytorch")
    report = {"crops": crops, "labels": labels, "framework": framework, "synthetic": synthetic, "accuracy": float(accuracy_score(y[te], pred)), "macro_f1": float(f1_score(y[te], pred, average="macro", zero_division=0)), "held_out_rows": len(te)}
    metadata(path / "report.json", report)
    return report

def predict(raw, crop, humidity, temperature):
    path = directory("agri")
    report = metadata(path / "report.json")
    x = image(raw)[None]
    m = crop_features(crop, humidity, temperature, report["crops"])[None]
    if report["framework"] == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(path / "model.keras", compile=False)
        probs = model([x, m], training=False).numpy()[0]
    else:
        import torch
        model = torch_model(m.shape[1], len(report["labels"]))
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(x.transpose(0,3,1,2)), torch.tensor(m)).softmax(1).numpy()[0]
    label = report["labels"][int(probs.argmax())]
    return {"label": label, "confidence": float(probs.max()), "probabilities": dict(zip(report["labels"], map(float, probs))), "review_required": bool(probs.max() < .7 or report["synthetic"]), "workflow": ["Isolate affected plants and photograph symptoms over time.", "Check irrigation, leaf moisture and ventilation.", "Ask a local agronomist to confirm the suspected disease before treatment."], "evaluation": report}
