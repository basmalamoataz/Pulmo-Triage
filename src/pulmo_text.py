import re, json, os, glob, string
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

_MISSING = {'none', 'nan', 'none.', 'none available', 'n/a', 'na', 'null', '', 'xxxx', 'xxxx, xxxx.', 'xxxx, xxxx', 'xxxx.'}
_PUNCT = str.maketrans('', '', string.punctuation)
_NON_INFO = re.compile(r"\b(hiddenyearold|hiddenyearsold|yearold|yearsold|male|female|patient)\b")

def clean_indication(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    t = t.translate(_PUNCT)
    if t in _MISSING or t == "not provided":
        return ""
    t = re.sub(r"(?i)xxxx", "hidden", t)
    rest = re.sub(r"\s+", " ", _NON_INFO.sub("", t)).strip().replace("hidden", "").strip()
    if len(rest) == 0 or rest in ["with", "old", "year", "and", "or"]:
        return ""
    return t

class TextModel:

    def __init__(self, model_dir, device=None):
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.dir = model_dir
        self.classes = list(self.meta["classes"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.models = [AutoModelForSequenceClassification.from_pretrained(d).float().to(self.device).eval()
                       for d in sorted(glob.glob(os.path.join(model_dir, "fold_*")))]
        self.model = self.models[0]
        self.tfidf = self._load("tfidf_ovr.joblib")
        self.emb_lr = self._load("emb_lr.joblib")
        self.w = self.meta["blend_weights"]
        self.max_len = int(self.meta["max_len"])
        self.thresholds = np.asarray(self.meta["thresholds"], dtype=np.float32)
        self._encoder = None

    def _load(self, name):
        path = os.path.join(self.dir, name)
        if os.path.exists(path):
            import joblib
            return joblib.load(path)
        return None

    def _enc(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    @torch.no_grad()
    def _finetuned(self, clean, batch_size=64):
        out = np.zeros((len(clean), len(self.classes)), dtype=np.float32)
        for s in range(0, len(clean), batch_size):
            enc = self._enc(clean[s:s + batch_size])
            out[s:s + batch_size] = torch.stack([torch.sigmoid(m(**enc).logits.float()) for m in self.models]).mean(0).cpu().numpy()
        return out

    def predict_proba(self, texts, batch_size=64):
        clean = [clean_indication(t) for t in texts]
        idx = [i for i, c in enumerate(clean) if c]
        p = np.full((len(clean), len(self.classes)), np.nan, dtype=np.float32)
        if idx:
            sub = [clean[i] for i in idx]
            mix = self.w["finetuned"] * self._finetuned(sub, batch_size)
            if self.w["tfidf"] > 0:
                mix += self.w["tfidf"] * self.tfidf.predict_proba(sub)
            if self.w["embeddings"] > 0:
                mix += self.w["embeddings"] * self.emb_lr.predict_proba(self.embed(sub))
            p[idx] = mix
        return p

    def predict_text(self, text, top_k=5):
        clean = clean_indication(text)
        if not clean:
            return {"has_note": False, "findings": [], "top": [], "warning": "No usable clinical note"}
        p = self.predict_proba([text])[0]
        order = np.argsort(-p)
        item = lambda i: {"label": self.classes[i], "probability": round(float(p[i]), 4)}
        out = {"has_note": True, "clean_text": clean,
               "findings": [item(i) for i in order if p[i] > self.thresholds[i]],
               "top": [item(i) for i in order[:top_k]]}
        if "normal" in self.classes:
            out["p_normal"] = round(float(p[self.classes.index("normal")]), 4)
        return out

    @torch.no_grad()
    def embed(self, texts, batch_size=64):
        if self._encoder is None:
            self._encoder = AutoModel.from_pretrained(os.path.join(self.dir, "frozen_encoder")).float().to(self.device).eval()
        clean = [clean_indication(t) for t in texts]
        dim = getattr(self._encoder.config, "hidden_size", None) or self._encoder.config.dim
        out = np.zeros((len(clean), dim), dtype=np.float32)
        idx = [i for i, c in enumerate(clean) if c]
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            enc = self._enc([clean[i] for i in b])
            h = self._encoder(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[b] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        return out

    def explain(self, text, label=None, top_k=8, n_steps=50):
        from captum.attr import LayerIntegratedGradients
        clean = clean_indication(text)
        if not clean:
            return {"label": None, "tokens": [], "top": []}
        if label is None:
            p = self.predict_proba([text])[0]
            label = next((self.classes[i] for i in np.argsort(-p) if self.classes[i] != "normal"), self.classes[int(np.argmax(p))])
        k = self.classes.index(label)
        enc = self._enc([clean])
        ids, mask = enc["input_ids"], enc["attention_mask"]
        special = {i for i in (self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id,
                               self.tok.bos_token_id, self.tok.eos_token_id) if i is not None}
        base = ids.clone()
        base[0, [j for j in range(ids.shape[1]) if int(ids[0, j]) not in special]] = self.tok.pad_token_id
        fwd = lambda input_ids, attention_mask: torch.sigmoid(self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float())[:, k]
        attr = LayerIntegratedGradients(fwd, self.model.get_input_embeddings()).attribute(
            ids, baselines=base, additional_forward_args=(mask,), n_steps=n_steps, internal_batch_size=25)
        scores = attr.sum(-1).squeeze(0).detach().cpu().numpy()
        words, wscores = [], []
        for tkn, sc, tid in zip(self.tok.convert_ids_to_tokens(ids[0].tolist()), scores, ids[0].tolist()):
            if tid in special:
                continue
            if tkn.startswith("##") and words:
                words[-1] += tkn[2:]; wscores[-1] += sc
            else:
                words.append(tkn); wscores.append(sc)
        norm = float(np.abs(wscores).max()) or 1.0 if wscores else 1.0
        items = [{"word": w, "score": round(float(s / norm), 4)} for w, s in zip(words, wscores)]
        return {"label": label, "tokens": items, "top": sorted(items, key=lambda d: -abs(d["score"]))[:top_k]}
