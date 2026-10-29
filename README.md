# PulmoTriage

Multimodal triage of chest X-rays. The system reads the **X-ray** and the doctor's **clinical note**, predicts **110 findings**, gives each patient a **priority** (high / medium / low), and writes a short, traceable **summary** for the doctor.

Graduation project · Samsung Innovation Campus (SIC 8) · AI track · 2026

<!-- ![PulmoTriage app](docs/images/app.png) -->

## How it works

```
X-ray images ──► CheXNet (frozen) ──► image features ──┐
                                                        ├─► fusion model ──► 110 findings ──► priority
Clinical note ──► text encoder (frozen) ──► text features ┘                        │
      │                                                                           ▼
      └──► NER + ConText (symptoms, history, negations) ───────────────► doctor's summary
```

- **Text model:** TF-IDF + fine-tuned DistilBERT ensemble + embedding model, blended.
- **Vision model:** CheXNet (DenseNet-121 trained on chest X-rays), frozen, with a small classifier on top.
- **Fusion:** early fusion (one network on image + text features) and late fusion (weighted average); the better one on the validation set is used.
- **Triage:** an urgent finding (pneumothorax, pleural effusion, consolidation, …) → high; any other finding → medium; nothing → low.
- **Summary:** clinical NLP pipeline (biomedical NER + medspaCy ConText) and a fixed template, so every sentence can be traced to the note or a model output.
- **Explainability:** Grad-CAM heatmaps, SHAP for words and image areas, Integrated Gradients for the text model.
- **Safety:** the radiologist's report is never an input (it is only used to build the labels), and every summary is fact-checked.

## Data

[Indiana University Chest X-rays](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university): 3,851 patients, 7,470 images, and reports with MeSH findings.

Patient-level split (no patient in two sets): train 2,464 · validation 616 · test 771 (`data/splits.csv`).

## Results (test set)

| Model | Macro ROC-AUC (95% CI) |
|---|---|
| Text only | 0.635 (0.610–0.663) |
| Image only | _add from `results/fusion_results.csv`_ |
| Early fusion | _add from `results/fusion_results.csv`_ |
| Late fusion | _add from `results/fusion_results.csv`_ |

| Summary extractor (gold notes) | Precision | Recall | F1 |
|---|---|---|---|
| Rules | 100% | 85.2% | 92.0% |
| NER | 100% | 81.8% | 90.0% |
| **Hybrid (used)** | **100%** | **98.9%** | **99.4%** |

The hybrid extractor finds at least one item in 89.1% of the real test notes (rules alone: 67.1%).

## Repository

```
notebooks/
  1_data_exploring_cleaning_eda.ipynb   cleaning, EDA, split and labels
  2_text_model.ipynb                    text model (NLP)
  3_summary_generator.ipynb             summary for the doctor
  4_vision_model.ipynb                  vision model (CheXNet + classifier, Grad-CAM)
  5_fusion_model.ipynb                  image + text fusion, triage, app package
src/
  pulmo_text.py      text model
  pulmo_vision.py    vision + fusion model
  pulmo_summary.py   summary generator
  shap_script.py     SHAP explanations
  LLM_reponse.py     optional LLM sentence (OpenRouter)
app/
  main.py            FastAPI backend (models, /predict, /explain)
  app.py             Streamlit frontend for the doctor
data/                splits.csv, mlb_classes.json, extractor_gold.csv
results/             metrics tables and figures
docs/                proposal, work plan, presentation
```

## Run the notebooks

The notebooks run on Kaggle, in this order. The inputs each one needs are listed in its first cells.

1. `1_data_exploring_cleaning_eda` → `nlp_dataset.csv`, `mlb_classes.json`, `splits.csv`
2. `2_text_model` → `text_model/`, `pulmo_text.py`, `text_embeddings.parquet`, `text_predictions.csv` (GPU)
3. `3_summary_generator` → `pulmo_summary.py`
4. `4_vision_model` → `image_embeddings.csv`, `image_model.keras`, Grad-CAM figures
5. `5_fusion_model` → `vision_model/`, `pulmo_vision.py`, `fusion_results.csv` (the app package)

Notebooks 4 and 5 need the CheXNet weights: [sinamhd9/chexnet-weights](https://www.kaggle.com/datasets/sinamhd9/chexnet-weights).

## Run the app

The app has two parts: a **FastAPI** backend (`app/main.py`) that runs the models, and a **Streamlit** frontend (`app/app.py`) for the doctor.

The trained models are too large for GitHub. Download them from **[Kaggle dataset link]** and put them here:

```
app/models_weights/text_model/
app/models_weights/vision_model/
```

```bash
pip install -r requirements.txt
cd app
uvicorn main:app --port 8001      # terminal 1: the API
streamlit run app.py              # terminal 2: the doctor app
```

- API documentation: http://localhost:8001/docs
- The first start downloads the biomedical NER model, so it needs internet (or start the API with `PULMO_LOAD_NER=0` to use the lexicon only).
- The optional LLM sentence in the "Deep explanation" tab needs an OpenRouter key: paste it in the app sidebar or set `OPENROUTER_API_KEY` before starting the API.

## Limitations

- The clinical notes are very short (4–9 words) and partly masked in the dataset ("XXXX"), which limits the text model.
- One public dataset (Indiana University); the models are not validated on other hospitals.
- The summary extractor's gold set was written by the team; an independent annotation is future work.
- The image region in the heatmap assumes a frontal (PA/AP) view.

## Team

| Role | Member |
|---|---|
| Data exploration and cleaning | |
| Vision model | |
| Text model and summary (NLP) | Basmala Moataz |
| Fusion model | |
| App and deployment | |

## Disclaimer

Educational prototype, not a medical device. The final decision is always the doctor's.

The dataset belongs to its original authors (Indiana University / Open-i); see the dataset page for its terms of use.
