"""PulmoTriage vision module: chest X-ray images (+ the clinical note) -> findings, heatmap, priority.

Same idea as pulmo_text.py:
    vm = VisionModel("vision_model")                        # load once
    image_result = vm.predict_image(images)                  # image only
    fusion_result = vm.predict_fusion(images, note, tm)      # image + note (tm = pulmo_text.TextModel)
    priority = vm.priority(fusion_result)                    # 'high' | 'medium' | 'low'
    overlay, region = vm.heatmap(images[0])                  # Grad-CAM picture + region text

`images` = one patient's X-rays (1-5 views): file paths, bytes, PIL images or Streamlit uploaded files.
Images are read exactly like in the preprocessing notebook (grayscale, padded to 224x224, divided by 255)
and encoded with the same frozen CheXNet, so the app's features match the training features.
"""
import io
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 224
MAX_IMAGES = 5


def load_image(source):
    """One X-ray -> float tensor (224, 224, 1) in [0, 1] (same steps as the preprocessing notebook)."""
    if isinstance(source, (str, os.PathLike)):
        data = tf.io.read_file(str(source))
    elif isinstance(source, (bytes, bytearray)):
        data = tf.constant(bytes(source))
    elif hasattr(source, "getvalue"):                     # Streamlit UploadedFile / BytesIO
        data = tf.constant(source.getvalue())
    else:                                                 # PIL image
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        data = tf.constant(buf.getvalue())
    image = tf.io.decode_image(data, channels=1, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    return image / 255.0


class VisionModel:
    def __init__(self, model_dir="vision_model"):
        self.dir = model_dir
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.classes = cfg["classes"]
        self.thr_image = np.array(cfg["thresholds_image"])
        self.thr_fusion = np.array(cfg["thresholds_fusion"])
        self.fusion_method = cfg["fusion_method"]          # 'early' or 'late' (chosen on the validation set)
        self.late_weight = cfg["late_fusion_image_weight"]
        self.urgent = [c for c in cfg["urgent_findings"] if c in self.classes]
        sc = np.load(os.path.join(model_dir, "scalers.npz"))
        self.img_mean, self.img_scale = sc["img_mean"], sc["img_scale"]
        self.txt_mean, self.txt_scale = sc["txt_mean"], sc["txt_scale"]
        load_kw = {"compile": False}
        self.image_model = keras.models.load_model(os.path.join(model_dir, "image_model.keras"), **load_kw)
        self.fusion_model = keras.models.load_model(os.path.join(model_dir, "fusion_model.keras"), **load_kw)
        self.text_only_model = keras.models.load_model(os.path.join(model_dir, "text_only_model.keras"), **load_kw)

        # frozen CheXNet (DenseNet-121), loaded exactly like in the preprocessing notebook
        self.chexnet = keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        self.chexnet.load_weights(os.path.join(model_dir, cfg["chexnet_weights"]), by_name=True, skip_mismatch=True)
        self.chexnet.trainable = False

    # ------------------------------------------------------------------ features
    def _conv(self, images):
        x = tf.stack([load_image(s) for s in images[:MAX_IMAGES]])
        return self.chexnet(tf.concat([x, x, x], axis=-1), training=False)          # (n, 7, 7, 1024)

    def embed(self, images):
        """One patient's X-rays -> 1024 numbers (average over the images), same as image_embeddings.csv."""
        return tf.reduce_mean(self._conv(images), axis=[1, 2]).numpy().mean(axis=0)

    def _scale_img(self, emb):
        return ((emb - self.img_mean) / self.img_scale).astype("float32")[None, :]

    def _scale_txt(self, emb, has_note):
        return (((emb - self.txt_mean) / self.txt_scale) * has_note).astype("float32")[None, :]

    # ------------------------------------------------------------------ predictions
    def _result(self, p, thresholds, top_k=5):
        order = np.argsort(-p)
        item = lambda i: {"label": self.classes[i], "probability": round(float(p[i]), 4)}
        res = {"findings": [item(i) for i in order if p[i] > thresholds[i] and self.classes[i] != "normal"],
               "top": [item(i) for i in order if self.classes[i] != "normal"][:top_k],
               "probabilities": {c: float(v) for c, v in zip(self.classes, p)}}
        if "normal" in self.classes:
            res["p_normal"] = round(float(p[self.classes.index("normal")]), 4)
        return res

    def predict_image(self, images, with_region=True):
        """Image only -> {'findings', 'top', 'p_normal', 'probabilities', 'region'}."""
        p = self.image_model.predict(self._scale_img(self.embed(images)), verbose=0)[0]
        res = self._result(p, self.thr_image)
        if with_region:
            res["region"] = self.heatmap(images[0])[1]
        return res

    def predict_fusion(self, images, note=None, text_model=None):
        """Image + clinical note -> same format as predict_image (falls back to image only without a note)."""
        has_note = 0.0
        txt = np.zeros_like(self.txt_mean)
        if note and str(note).strip() and text_model is not None:
            txt = text_model.embed([note])[0]
            has_note = float(np.abs(txt).sum() > 0)
        x_img = self._scale_img(self.embed(images))
        x_txt = self._scale_txt(txt, has_note)
        note_col = np.array([[has_note]], dtype="float32")
        if self.fusion_method == "early":
            p = self.fusion_model.predict(np.hstack([x_img, x_txt, note_col]), verbose=0)[0]
        else:
            p_img = self.image_model.predict(x_img, verbose=0)[0]
            p_txt = self.text_only_model.predict(np.hstack([x_txt, note_col]), verbose=0)[0]
            p = self.late_weight * p_img + (1 - self.late_weight) * p_txt if has_note else p_img
        res = self._result(p, self.thr_fusion)
        res["used_note"] = bool(has_note)
        return res

    def priority(self, result):
        """Triage rule: an urgent finding flagged -> high; any other finding flagged -> medium; nothing -> low."""
        flagged = [d["label"] for d in result["findings"]]
        if any(f in self.urgent for f in flagged):
            return "high"
        if flagged:
            return "medium"
        return "low"

    # ------------------------------------------------------------------ explanation
    def heatmap(self, image, label=None):
        """Grad-CAM of the image model on one X-ray -> (RGB overlay uint8, region text).
        Region assumes a frontal (PA/AP) view: the patient's right is on the left side of the image."""
        x = load_image(image)
        if label is None:
            p = self.image_model.predict(self._scale_img(self.embed([image])), verbose=0)[0]
            p = np.where(np.array(self.classes) == "normal", -1, p)
            k = int(np.argmax(p))
        else:
            k = self.classes.index(label)
        x3 = tf.concat([x, x, x], axis=-1)[None]
        mean, scale = tf.constant(self.img_mean, tf.float32), tf.constant(self.img_scale, tf.float32)
        with tf.GradientTape() as tape:
            conv = self.chexnet(x3, training=False)
            tape.watch(conv)
            pooled = (tf.reduce_mean(conv, axis=[1, 2]) - mean) / scale
            score = self.image_model(pooled, training=False)[0, k]
        grads = tape.gradient(score, conv)[0]
        weights = tf.reduce_mean(grads, axis=[0, 1])
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]

        # region of the strongest activation
        r, c = np.unravel_index(np.argmax(cam), cam.shape)
        zone = ["upper", "middle", "lower"][min(2, int(3 * r / IMG_SIZE))]
        side = "right" if c < IMG_SIZE / 2 else "left"
        region = f"{side} {zone} zone"

        # red overlay on the X-ray
        gray = np.repeat(x.numpy(), 3, axis=-1)
        heat = np.stack([cam, np.zeros_like(cam), np.zeros_like(cam)], axis=-1)
        overlay = np.clip(0.65 * gray + 0.5 * heat, 0, 1)
        return (overlay * 255).astype("uint8"), region
