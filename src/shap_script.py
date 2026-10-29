import numpy as np
import shap
import tensorflow as tf

from pulmo_vision import IMG_SIZE, MAX_IMAGES, load_image

def _fusion_probs(vm, img_emb, txt_emb, has_note):
    x_img = ((img_emb - vm.img_mean) / vm.img_scale).astype("float32")
    x_txt = (((txt_emb - vm.txt_mean) / vm.txt_scale) * has_note[:, None]).astype("float32")
    if vm.fusion_method == "early":
        return vm.fusion_model.predict(np.hstack([x_img, x_txt, has_note[:, None]]), verbose=0)
    p_img = vm.image_model.predict(x_img, verbose=0)
    p_txt = vm.text_only_model.predict(np.hstack([x_txt, has_note[:, None]]), verbose=0)
    w = vm.late_weight
    return np.where(has_note[:, None] == 1, w * p_img + (1 - w) * p_txt, p_img)

def _image_features(vm, batch, chunk=8):
    out = []
    for i in range(0, len(batch), chunk):
        x = tf.convert_to_tensor(batch[i:i + chunk], tf.float32)
        out.append(tf.reduce_mean(vm.chexnet(tf.concat([x, x, x], axis=-1), training=False), axis=[1, 2]).numpy())
    return np.vstack(out)

def _note_embedding(tm, notes):
    if tm is None:
        return np.zeros((len(notes), 768), "float32"), np.zeros(len(notes), "float32")
    emb = tm.embed(list(notes))
    return emb, (np.abs(emb).sum(axis=1) > 0).astype("float32")

def explain_case(vm, tm, images, note=None, target_label=None, max_evals=300, explain_image=True):
    images = list(images)[:MAX_IMAGES]
    has_text = bool(note and str(note).strip()) and tm is not None
    pixels = np.stack([load_image(s).numpy() for s in images])
    feats = _image_features(vm, pixels)
    txt_emb, has_note = _note_embedding(tm, [note if has_text else ""])

    p = _fusion_probs(vm, feats.mean(axis=0, keepdims=True), txt_emb, has_note)[0]
    if target_label is None:
        p_rank = np.where(np.array(vm.classes) == "normal", -1, p)
        target_label = vm.classes[int(np.argmax(p_rank))]
    k = vm.classes.index(target_label)
    out = {"target": target_label, "probability": float(p[k]), "used_note": has_text,
           "word_contributions": [], "top_words": [], "key_words": [],
           "image_values": None, "overlay": None, "region": None}

    if has_text:
        img_fixed = feats.mean(axis=0, keepdims=True)

        def f_text(texts):
            emb, has = _note_embedding(tm, [str(t) for t in texts])
            return _fusion_probs(vm, np.repeat(img_fixed, len(texts), axis=0), emb, has)[:, k]

        sv = shap.Explainer(f_text, shap.maskers.Text(r"\W+"))([note], max_evals=max_evals)[0]
        words = [(str(w).strip(" ,.;:!?()"), float(v)) for w, v in zip(sv.data, sv.values) if str(w).strip(" ,.;:!?()")]
        out["word_contributions"] = words
        out["top_words"] = sorted([w for w in words if w[1] > 0], key=lambda x: -x[1])[:5]
        out["key_words"] = [w for w, _ in out["top_words"]]

    if explain_image:
        others = feats[1:].sum(axis=0)
        n_img = len(images)

        def f_image(masked):
            batch_feats = _image_features(vm, masked)
            patient = (batch_feats + others) / n_img
            n = len(masked)
            return _fusion_probs(vm, patient, np.repeat(txt_emb, n, axis=0), np.repeat(has_note, n))[:, k]

        masker = shap.maskers.Image("blur(16,16)", pixels[0].shape)
        sv = shap.Explainer(f_image, masker)(pixels[:1], max_evals=max_evals, batch_size=16)
        values = np.asarray(sv.values[0])[..., 0]
        out["image_values"] = values
        pos = np.clip(values, 0, None)
        if pos.max() > 0:
            r, c = np.unravel_index(np.argmax(pos), pos.shape)
            zone = ["upper", "middle", "lower"][min(2, int(3 * r / IMG_SIZE))]
            side = "right" if c < IMG_SIZE / 2 else "left"
            out["region"] = f"{side} {zone} zone"
            heat = pos / pos.max()
            gray = np.repeat(pixels[0], 3, axis=-1)
            overlay = np.clip(0.65 * gray + 0.5 * np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], -1), 0, 1)
            out["overlay"] = (overlay * 255).astype("uint8")
    return out

def evidence_text(exp):
    parts = [f"Model estimate for {exp['target']}: {exp['probability']:.0%}."]
    if exp["used_note"]:
        parts.append("Note words supporting it: " + (", ".join(exp["key_words"]) if exp["key_words"] else "none") + ".")
    if exp["region"]:
        parts.append(f"Image area supporting it: {exp['region']}.")
    return " ".join(parts)

def llm_prompt(exp, note=None):
    return ("Rewrite the following AI evidence as two short sentences for a doctor. Use only these facts; do not add a "
            "diagnosis, symptoms or numbers that are not listed.\n"
            f"Facts: {evidence_text(exp)}" + (f"\nClinical note: {note}" if note else ""))
