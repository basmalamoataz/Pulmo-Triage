"""PulmoTriage REST API (FastAPI).

Run:   uvicorn main:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs

Folder layout next to this file:
    main.py  app.py  test_api.py
    src/pulmo_text.py  src/pulmo_vision.py  src/pulmo_summary.py
    models_weights/text_model/    (meta.json, fold_*, tfidf_ovr.joblib, emb_lr.joblib, frozen_encoder/ ...)
    models_weights/vision_model/  (config.json, scalers.npz, image_model.keras, fusion_model.keras, ...)

Model folders can be changed with environment variables:
    TEXT_MODEL_DIR   (default: models_weights/text_model)
    VISION_MODEL_DIR (default: models_weights/vision_model)
    PULMO_LOAD_NER   (default: 1)  -> 0 = skip the biomedical NER model, use the lexicon matcher only (no internet needed)
"""
import base64
import io
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# pulmo_text / pulmo_vision / pulmo_summary live in the "src" folder next to this file
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import pulmo_summary
from pulmo_text import TextModel
from pulmo_vision import MAX_IMAGES, VisionModel, load_image

log = logging.getLogger("pulmotriage")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TEXT_MODEL_DIR = os.getenv("TEXT_MODEL_DIR", str(BASE_DIR / "models_weights" / "text_model"))
VISION_MODEL_DIR = os.getenv("VISION_MODEL_DIR", str(BASE_DIR / "models_weights" / "vision_model"))
LOAD_NER = os.getenv("PULMO_LOAD_NER", "1") == "1"
MAX_IMAGE_BYTES = 15 * 1024 * 1024          # 15 MB per X-ray
MAX_NOTE_CHARS = 5000

# TensorFlow and PyTorch live in the same process: serialise inference so requests never overlap.
_LOCK = threading.Lock()
_STATE: Dict[str, object] = {"text": None, "vision": None, "ner": False}


# ------------------------------------------------------------------------------------------------ schemas
class Finding(BaseModel):
    label: str
    probability: float


class ModelResult(BaseModel):
    """Same shape for the image model, the fusion model and the text model."""
    findings: List[Finding] = []
    top: List[Finding] = []
    p_normal: Optional[float] = None
    region: Optional[str] = Field(None, description="Strongest Grad-CAM region (image models only)")
    used_note: Optional[bool] = Field(None, description="Fusion only: was the note used?")
    has_note: Optional[bool] = Field(None, description="Text model only")
    clean_text: Optional[str] = Field(None, description="Text model only: the cleaned note")
    warning: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = Field(None, description="All findings (image / fusion models)")


class WordScore(BaseModel):
    word: str
    score: float = Field(..., description="-1 .. +1: pushes the finding down (-) or up (+)")


class Explanation(BaseModel):
    label: Optional[str] = None
    tokens: List[WordScore] = []
    top: List[WordScore] = []


class Heatmap(BaseModel):
    index: int
    region: str
    png_base64: str


class Summary(BaseModel):
    sections: Dict[str, str]
    text: str
    problems: List[str] = Field([], description="Traceability problems found by check_summary (empty = OK)")


class PredictResponse(BaseModel):
    priority: str = Field(..., description="high | medium | low")
    urgent_findings: List[str]
    n_images: int
    used_note: bool
    fusion: ModelResult
    image: ModelResult
    text: ModelResult
    explanation: Optional[Explanation] = None
    heatmaps: List[Heatmap] = []
    summary: Optional[Summary] = None
    timings: Dict[str, float] = {}
    warnings: List[str] = []


class HealthResponse(BaseModel):
    status: str
    text_model: bool
    vision_model: bool
    ner_model: bool
    classes: List[str] = []


# ------------------------------------------------------------------------------------------------ startup
def _limit_gpu_memory():
    """Stop TensorFlow from grabbing the whole GPU (PyTorch needs some too)."""
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:                                                   # noqa: BLE001
        log.warning("GPU memory growth not set: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _limit_gpu_memory()
    t0 = time.time()
    log.info("Loading text model from '%s' ...", TEXT_MODEL_DIR)
    _STATE["text"] = TextModel(TEXT_MODEL_DIR)
    log.info("Loading vision / fusion models from '%s' ...", VISION_MODEL_DIR)
    _STATE["vision"] = VisionModel(VISION_MODEL_DIR)
    if LOAD_NER:
        log.info("Loading the biomedical NER model (needs internet the first time) ...")
        _STATE["ner"] = pulmo_summary.load_ner()
    log.info("Ready in %.1fs", time.time() - t0)
    yield
    _STATE.update(text=None, vision=None)


app = FastAPI(title="PulmoTriage API", version="1.0",
              description="Chest X-ray + clinical note -> findings, priority, heatmap and word evidence. "
                          "Educational prototype: the final decision is always the doctor's.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------------------------------------ helpers
def _png_b64(array: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _crop_overlay(overlay: np.ndarray, raw: bytes, fallback_region: str):
    """The vision module pads every X-ray to a 224x224 square. Cut the black bars off, scale the picture back up
    and find the strongest red area on the real X-ray only. Falls back to the module's own output on any problem."""
    try:
        w, h = Image.open(io.BytesIO(raw)).size
        size = overlay.shape[0]
        scale = min(size / w, size / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        x0, y0 = (size - nw) // 2, (size - nh) // 2
        box = overlay[y0:y0 + nh, x0:x0 + nw]
        # the module draws heat as extra red on a grey picture: red - green ~ heat
        heat = box[..., 0].astype(np.int16) - box[..., 1].astype(np.int16)
        r, c = np.unravel_index(int(np.argmax(heat)), heat.shape)
        zone = ["upper", "middle", "lower"][min(2, int(3 * r / nh))]
        side = "right" if c < nw / 2 else "left"          # frontal view: patient's right = left of the picture
        k = 512 / max(nw, nh)
        big = Image.fromarray(box).resize((max(1, int(nw * k)), max(1, int(nh * k))), Image.BILINEAR)
        return np.asarray(big), f"{side} {zone} zone"
    except Exception as e:                                   # noqa: BLE001
        log.warning("heatmap crop failed: %s", e)
        return overlay, fallback_region


def _model_result(d: Optional[dict]) -> ModelResult:
    if not d:
        return ModelResult()
    return ModelResult(**{k: v for k, v in d.items() if k in ModelResult.model_fields})


# ------------------------------------------------------------------------------------------------ endpoints
def _read_images(images: List[UploadFile]):
    """Validate the uploaded X-rays. Returns (list of bytes, warnings)."""
    if not images:
        raise HTTPException(422, "Upload at least one X-ray image.")
    warnings: List[str] = []
    if len(images) > MAX_IMAGES:
        warnings.append(f"Only the first {MAX_IMAGES} images are used.")
        images = images[:MAX_IMAGES]
    raw: List[bytes] = []
    for f in images:
        data = f.file.read()
        if not data:
            raise HTTPException(422, f"'{f.filename}' is empty.")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"'{f.filename}' is larger than {MAX_IMAGE_BYTES // 2**20} MB.")
        try:
            load_image(data)                                  # same decoding as the models -> fails early if unreadable
        except Exception:                                     # noqa: BLE001
            raise HTTPException(422, f"'{f.filename}' is not a readable image (use PNG or JPG).")
        raw.append(data)
    return raw, warnings


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    vm, tm = _STATE["vision"], _STATE["text"]
    return HealthResponse(status="ok" if (vm and tm) else "loading", text_model=tm is not None,
                          vision_model=vm is not None, ner_model=bool(_STATE["ner"]),
                          classes=list(vm.classes) if vm else [])


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(
    images: List[UploadFile] = File(..., description="1-5 X-ray views of ONE patient (png / jpg)"),
    note: Optional[str] = Form(None, description="Clinical note (indication). Can be empty."),
    explain: bool = Form(True, description="Word-level explanation of the note (a bit slower)"),
    heatmap: bool = Form(True, description="Grad-CAM heatmap for every image"),
):
    tm: TextModel = _STATE["text"]
    vm: VisionModel = _STATE["vision"]
    if tm is None or vm is None:
        raise HTTPException(503, "Models are still loading, try again in a moment.")

    # ---- validate input
    raw, warnings = _read_images(images)
    note = (note or "").strip()
    if len(note) > MAX_NOTE_CHARS:
        raise HTTPException(422, f"The note is longer than {MAX_NOTE_CHARS} characters.")

    timings: Dict[str, float] = {}
    with _LOCK:
        t = time.time()
        # ---- text model
        text_result = tm.predict_text(note)
        timings["text"] = round(time.time() - t, 2)

        # ---- image model (heatmaps are computed separately below)
        t = time.time()
        image_result = vm.predict_image(raw, with_region=False)
        timings["image"] = round(time.time() - t, 2)

        # ---- fusion model + priority
        t = time.time()
        fusion_result = vm.predict_fusion(raw, note, tm)
        priority = vm.priority(fusion_result)
        timings["fusion"] = round(time.time() - t, 2)

        # ---- Grad-CAM on every image, for the finding the image model finds most likely
        heatmaps: List[Heatmap] = []
        if heatmap:
            t = time.time()
            label = image_result["top"][0]["label"] if image_result["top"] else None
            for i, data in enumerate(raw):
                overlay, region = vm.heatmap(data, label=label)
                overlay, region = _crop_overlay(overlay, data, region)
                heatmaps.append(Heatmap(index=i, region=region, png_base64=_png_b64(overlay)))
            image_result["region"] = heatmaps[0].region
            timings["heatmap"] = round(time.time() - t, 2)

        # ---- word-level evidence from the note
        explanation = None
        if explain and text_result.get("has_note"):
            t = time.time()
            try:
                flagged = [d["label"] for d in text_result.get("findings", []) if d["label"] != "normal"]
                explanation = Explanation(**tm.explain(note, label=flagged[0] if flagged else None))
            except Exception as e:                            # noqa: BLE001  (e.g. captum not installed)
                log.warning("explain failed: %s", e)
                warnings.append("Word highlighting is not available.")
            timings["explain"] = round(time.time() - t, 2)

    # ---- doctor's summary (never blocks the answer)
    summary = None
    try:
        t = time.time()
        key_words = [d.word for d in explanation.top] if explanation else None
        s = pulmo_summary.make_summary(note, text_result=text_result, image_result=image_result,
                                       fusion_result=fusion_result, priority=priority, key_words=key_words)
        problems = pulmo_summary.check_summary(s, note, text_result, image_result, fusion_result)
        summary = Summary(sections=s["sections"], text=s["text"], problems=problems)
        timings["summary"] = round(time.time() - t, 2)
    except Exception as e:                                    # noqa: BLE001
        log.warning("summary failed: %s", e)
        warnings.append("The written summary is not available.")

    if not text_result.get("has_note"):
        warnings.append("No usable clinical note: the result is based on the image only.")

    return PredictResponse(
        priority=priority, urgent_findings=list(vm.urgent), n_images=len(raw),
        used_note=bool(fusion_result.get("used_note")),
        fusion=_model_result(fusion_result), image=_model_result(image_result), text=_model_result(text_result),
        explanation=explanation, heatmaps=heatmaps, summary=summary, timings=timings, warnings=warnings)


# ------------------------------------------------------------------------------------------------ deep explanation
class WordContribution(BaseModel):
    word: str
    value: float = Field(..., description="SHAP value: > 0 pushes the finding up, < 0 pushes it down")


class ExplainResponse(BaseModel):
    target: str
    probability: float
    used_note: bool
    key_words: List[str] = []
    word_contributions: List[WordContribution] = []
    region: Optional[str] = None
    overlay_png_base64: Optional[str] = None
    evidence: str
    llm_text: Optional[str] = None
    llm_error: Optional[str] = None
    timings: Dict[str, float] = {}
    warnings: List[str] = []


def _evidence(exp: dict, region: Optional[str]) -> str:
    """Facts only, no probability (the app does not show confidence numbers)."""
    parts = [f"Finding under review: {exp['target'].replace('_', ' ')}."]
    if exp["used_note"]:
        words = ", ".join(exp["key_words"]) if exp["key_words"] else "none"
        parts.append(f"Note words supporting it: {words}.")
    if region:
        parts.append(f"Image area supporting it: {region}.")
    return " ".join(parts)


@app.post("/explain", response_model=ExplainResponse, tags=["prediction"])
def explain(
    images: List[UploadFile] = File(...),
    note: Optional[str] = Form(None),
    target_label: Optional[str] = Form(None, description="Finding to explain. Empty = the most likely one"),
    explain_image: bool = Form(True, description="SHAP on the image too (slow on CPU)"),
    max_evals: int = Form(300, description="SHAP evaluations: more = steadier but slower"),
    use_llm: bool = Form(False, description="Rewrite the evidence in two sentences with an LLM (OpenRouter)"),
    openrouter_key: Optional[str] = Form(None, description="Optional. Else env var OPENROUTER_API_KEY is used"),
    llm_model: Optional[str] = Form(None, description="OpenRouter model id, e.g. openai/gpt-4o-mini"),
):
    """On-demand, slower explanation with SHAP (src/shap_script.py) and an optional LLM sentence (src/LLM_reponse.py)."""
    tm: TextModel = _STATE["text"]
    vm: VisionModel = _STATE["vision"]
    if tm is None or vm is None:
        raise HTTPException(503, "Models are still loading, try again in a moment.")
    try:
        import shap_script
    except ImportError as e:
        raise HTTPException(501, f"SHAP is not installed ({e}). Run: pip install shap openai")

    raw, warnings = _read_images(images)
    note = (note or "").strip()
    if len(note) > MAX_NOTE_CHARS:
        raise HTTPException(422, f"The note is longer than {MAX_NOTE_CHARS} characters.")
    target_label = (target_label or "").strip() or None
    if target_label and target_label not in vm.classes:
        raise HTTPException(422, f"Unknown finding '{target_label}'.")
    max_evals = max(50, min(int(max_evals), 1000))

    timings: Dict[str, float] = {}
    with _LOCK:
        t = time.time()
        exp = shap_script.explain_case(vm, tm, raw, note or None, target_label=target_label,
                                       max_evals=max_evals, explain_image=explain_image)
        timings["shap"] = round(time.time() - t, 2)

    region, overlay_b64 = exp.get("region"), None
    if exp.get("overlay") is not None:
        overlay, region = _crop_overlay(exp["overlay"], raw[0], region or "")
        overlay_b64 = _png_b64(overlay)
    evidence = _evidence(exp, region)

    llm_text = llm_error = None
    if use_llm:
        key = (openrouter_key or "").strip() or os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            llm_error = "No OpenRouter key. Paste it in the app sidebar or set OPENROUTER_API_KEY before starting the API."
        else:
            try:
                from LLM_reponse import ask_openai
                prompt = ("Rewrite the following AI evidence as two short sentences for a doctor. Use only these facts; "
                          "do not add a diagnosis, symptoms or numbers that are not listed.\n"
                          f"Facts: {evidence}" + (f"\nClinical note: {note}" if note else ""))
                t = time.time()
                answer = ask_openai(prompt, key, llm_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
                timings["llm"] = round(time.time() - t, 2)
                if answer.startswith("Error"):
                    llm_error = answer[:300]
                else:
                    llm_text = answer
            except ImportError as e:
                llm_error = f"LLM module not available ({e}). Run: pip install openai"

    return ExplainResponse(
        target=exp["target"], probability=exp["probability"], used_note=exp["used_note"],
        key_words=exp["key_words"],
        word_contributions=[WordContribution(word=w, value=v) for w, v in exp["word_contributions"]],
        region=region, overlay_png_base64=overlay_b64, evidence=evidence,
        llm_text=llm_text, llm_error=llm_error, timings=timings, warnings=warnings)