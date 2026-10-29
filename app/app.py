"""PulmoTriage doctor app (Streamlit). Talks to the FastAPI service in main.py.

Run:   streamlit run app.py
API:   set PULMO_API_URL if the API is not on http://localhost:8000
"""
import base64
import html
import os
import re

import pandas as pd
import requests
import streamlit as st

API_URL_DEFAULT = os.getenv("PULMO_API_URL", "http://localhost:8001")
SHOW_PROBABILITIES = False      # True = show confidence numbers (percentages) again
HIDDEN_LABELS = {"normal", "no indexing"}          # not shown as findings (same idea as pulmo_summary.SKIP_LABELS)

PRIORITY_STYLE = {
    "high":   {"bg": "#FDECEA", "fg": "#8C1D18", "bar": "#C62828", "title": "High priority",
               "step": "Review first: radiologist read as soon as possible."},
    "medium": {"bg": "#FFF4E0", "fg": "#7A4A00", "bar": "#E08A00", "title": "Medium priority",
               "step": "Review today in the normal queue."},
    "low":    {"bg": "#E8F5EC", "fg": "#14532D", "bar": "#2E8B57", "title": "Low priority",
               "step": "Routine review."},
}
EXAMPLES = {
    "Fever and cough": "58 year old male with fever and cough for 5 days. History of COPD, smoker.",
    "Chest pain": "Chest pain and shortness of breath since yesterday. No fever.",
    "Pre-operative": "Pre-operative chest x-ray before surgery. No cough, no fever.",
}

st.set_page_config(page_title="PulmoTriage", page_icon="🫁", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  .prio {border-left: 8px solid; border-radius: 6px; padding: 14px 18px; margin: 6px 0 14px 0;}
  .prio h2 {margin: 0; font-size: 1.5rem;}
  .prio p {margin: 4px 0 0 0;}
  .note-words {font-size: 1.05rem; line-height: 2.1; padding: 12px 14px; border: 1px solid #8884; border-radius: 6px;}
  .note-words span {padding: 2px 3px; border-radius: 3px;}
  .small {font-size: 0.85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


def api_health(url):
    try:
        r = requests.get(f"{url}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:                                                        # noqa: BLE001
        return None


def call_predict(url, files, note, explain, heatmap):
    payload = [("images", (f["name"], f["bytes"], f["type"] or "image/png")) for f in files]
    data = {"note": note or "", "explain": str(explain).lower(), "heatmap": str(heatmap).lower()}
    r = requests.post(f"{url}/predict", files=payload, data=data, timeout=600)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:                                                    # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def pct(p):
    return None if p is None else p * 100


def pretty(label):
    return label.replace("_", " ")


def strip_probs(text):
    """Remove confidence numbers from model-written text, e.g. 'lung opacity (11%)' -> 'lung opacity'."""
    if SHOW_PROBABILITIES or not text:
        return text
    text = re.sub(r"\s*Probability of a normal study[^.]*?\d+%\.", "", text)
    text = re.sub(r"\s*\(\d+(\.\d+)?%\)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def call_explain(url, files, note, target, explain_image, max_evals, use_llm, key, model):
    payload = [("images", (f["name"], f["bytes"], f["type"] or "image/png")) for f in files]
    data = {"note": note or "", "target_label": target or "", "explain_image": str(explain_image).lower(),
            "max_evals": str(max_evals), "use_llm": str(use_llm).lower(), "openrouter_key": key or "",
            "llm_model": model or ""}
    r = requests.post(f"{url}/explain", files=payload, data=data, timeout=1800)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:                                                    # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def custom_priority(fusion, urgent, threshold):
    """Re-apply the triage rule (urgent -> high, other finding -> medium, none -> low) with one global threshold."""
    probs = fusion.get("probabilities") or {}
    flagged = {l: p for l, p in probs.items() if l not in HIDDEN_LABELS and p > threshold}
    if any(l in urgent for l in flagged):
        return "high", flagged
    return ("medium" if flagged else "low"), flagged


def word_html(tokens):
    parts = []
    for t in tokens:
        w, s = html.escape(t["word"]), t["score"]
        if abs(s) < 0.05:
            parts.append(w)
            continue
        color = "198,40,40" if s > 0 else "30,100,200"
        parts.append(f'<span style="background: rgba({color},{min(abs(s), 1) * 0.55:.2f})">{w}</span>')
    return '<div class="note-words">' + " ".join(parts) + "</div>"


def set_example(text):
    st.session_state["note"] = text


def clear_all():
    st.session_state["note"] = ""
    st.session_state.pop("result", None)
    st.session_state.pop("deep", None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


st.session_state.setdefault("note", "")
st.session_state.setdefault("uploader_key", 0)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API address", API_URL_DEFAULT).rstrip("/")
    health = api_health(api_url)
    if health is None:
        st.error("API not reachable. Start it with: uvicorn main:app --port 8000")
    elif health["status"] != "ok":
        st.warning("API is still loading the models.")
    else:
        st.success("API ready")
        if not health.get("ner_model"):
            st.caption("Note reader: lexicon mode (NER model not loaded).")

    show_words = st.checkbox("Highlight important words in the note", value=True)
    show_heat = st.checkbox("Show heatmap on the X-ray", value=True)

    st.subheader("Sensitivity")
    mode = st.radio("How are findings flagged?", ["Tuned by the model (recommended)", "One threshold for all findings"],
                    label_visibility="collapsed")
    custom_thr = 0.5
    if mode.startswith("One"):
        custom_thr = st.slider("A finding is flagged above", 0.05, 0.95, 0.30, 0.05,
                               help="Lower = more cases flagged (fewer missed, more extra reviews). "
                                    "Higher = fewer flagged (more missed, fewer extra reviews).")
    st.subheader("AI explanation (optional)")
    use_llm = st.checkbox("Write a short explanation with an LLM", value=False,
                          help="Used by the 'Deep explanation' tab. Needs an OpenRouter key.")
    llm_key = st.text_input("OpenRouter key", value="", type="password",
                            help="Leave empty if the API was started with OPENROUTER_API_KEY set.")
    llm_model = st.text_input("Model", value="openai/gpt-4o-mini")
    st.divider()
    st.caption("Educational prototype. It supports prioritisation, it does not diagnose. "
               "The final decision is always the doctor's.")

st.title("PulmoTriage")
st.caption("Upload a patient's chest X-ray and add the clinical note. You get a review priority and the evidence behind it.")

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Chest X-ray")
    uploads = st.file_uploader("X-ray images of one patient (1 to 5 views)", type=["png", "jpg", "jpeg"],
                               accept_multiple_files=True, key=f"up_{st.session_state['uploader_key']}")
    if uploads and len(uploads) > 5:
        st.warning("Only the first 5 images will be used.")
        uploads = uploads[:5]
    if uploads:
        cols = st.columns(min(len(uploads), 3))
        for i, f in enumerate(uploads):
            cols[i % len(cols)].image(f, caption=f.name, use_container_width=True)

with right:
    st.subheader("Clinical note")
    st.text_area("Reason for the exam / patient history", key="note", height=170,
                 placeholder="For example: fever and cough for 5 days, history of COPD")
    st.caption("The note is optional. Without one, the result uses the image only.")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

b1, b2, _ = st.columns([1, 1, 3])
analyze = b1.button("Analyze", type="primary", use_container_width=True, disabled=not uploads or health is None)
b2.button("Clear", on_click=clear_all, use_container_width=True)
if not uploads:
    st.info("Add at least one X-ray to start.")

if analyze:
    files = [{"name": f.name, "bytes": f.getvalue(), "type": f.type} for f in uploads]
    with st.spinner("Analyzing the X-ray and the note..."):
        try:
            result = call_predict(api_url, files, st.session_state["note"], show_words, show_heat)
            result["_files"] = files
            result["_note"] = st.session_state["note"]
            st.session_state["result"] = result
            st.session_state.pop("deep", None)
        except requests.exceptions.ConnectionError:
            st.error("Could not reach the API. Check that it is running and the address in the sidebar is right.")
        except Exception as e:                                               # noqa: BLE001
            st.error(f"The analysis failed. {e}")

res = st.session_state.get("result")
if res:
    st.divider()
    fusion, urgent = res["fusion"], res["urgent_findings"]
    if mode.startswith("One"):
        priority, flagged = custom_priority(fusion, urgent, custom_thr)
        flagged_list = sorted(flagged.items(), key=lambda kv: -kv[1])
    else:
        priority = res["priority"]
        flagged_list = [(d["label"], d["probability"]) for d in fusion["findings"] if d["label"] not in HIDDEN_LABELS]
    style = PRIORITY_STYLE[priority]

    extra = f"<p class='small'>Custom threshold {custom_thr:.2f} is on (sidebar).</p>" if mode.startswith("One") else ""
    banner = (f"<div class='prio' style='background:{style['bg']}; color:{style['fg']}; border-color:{style['bar']}'>"
              f"<h2>{style['title']}</h2><p>{style['step']}</p>{extra}</div>")
    st.markdown(banner, unsafe_allow_html=True)
    for w in res.get("warnings", []):
        st.warning(w)

    m1, m2, m3 = st.columns(3)
    m1.metric("Findings flagged", len(flagged_list))
    m2.metric("X-ray views analyzed", res["n_images"])
    m3.metric("Note used", "Yes" if res["used_note"] else "No")

    tab_sum, tab_find, tab_heat, tab_words, tab_deep, tab_tech = st.tabs(
        ["Summary", "Findings", "Heatmap", "Note words", "Deep explanation", "Details"])

    # ---- summary
    with tab_sum:
        summ = res.get("summary")
        if not summ:
            st.info("The written summary is not available for this case.")
        else:
            sec = summ["sections"]
            labels = {"presentation": "Patient", "reason": "Reason", "history": "History", "denied": "Denied",
                      "other": "Other", "assessment": "Combined assessment", "image": "Image", "note_model": "Note",
                      "key_words": "Key words", "priority": "Priority"}
            custom = mode.startswith("One")
            if custom:
                sec = dict(sec)
                sec["priority"] = (f"Priority: {priority.upper()}. {style['step']} "
                                   f"(custom threshold {custom_thr:.2f}; the model's own rule gave {res['priority'].upper()}.)")
            for key in ["presentation", "reason", "history", "denied", "other", "assessment", "image", "note_model",
                        "key_words", "priority"]:
                if key in sec:
                    st.markdown(f"**{labels[key]}.** {strip_probs(sec[key])}")
            if "disclaimer" in sec:
                st.caption(sec["disclaimer"])
            if summ.get("problems"):
                st.warning("Check these statements against the note: " + "; ".join(summ["problems"]))
            report = (f"PulmoTriage report\nPriority: {priority.upper()}"
                      + (f" (custom threshold {custom_thr:.2f})" if custom else "") + f"\n\n{strip_probs(summ['text'])}\n")
            st.download_button("Download summary (.txt)", report, file_name="pulmotriage_summary.txt")

    with tab_find:
        if flagged_list:
            st.markdown("**Flagged findings (combined image + note)**")
            for label, p in flagged_list:
                st.markdown(f"- {pretty(label)}" + (f" ({p:.0%})" if SHOW_PROBABILITIES else ""))
        else:
            st.success("No specific finding is flagged.")

        st.markdown("**Most likely findings, from each model**")
        def top_names(r, n=5):
            return [pretty(d["label"]) for d in r.get("top", []) if d["label"] not in HIDDEN_LABELS][:n]
        c1, c2, c3 = st.columns(3)
        for col, title, r in [(c1, "Combined", fusion), (c2, "Image only", res["image"]), (c3, "Note only", res["text"])]:
            col.markdown(f"*{title}*")
            names = top_names(r)
            col.markdown("\n".join(f"{i}. {n}" for i, n in enumerate(names, 1)) if names else "-")
        st.caption("Ranked from most to least likely. The ranking says what the models lean toward; it is not a diagnosis.")

    with tab_heat:
        if not res["heatmaps"]:
            st.info("Heatmap is turned off. Enable it in the sidebar and analyze again.")
        else:
            for hm in res["heatmaps"]:
                c1, c2 = st.columns(2)
                original = res["_files"][hm["index"]]
                c1.image(original["bytes"], caption=f"X-ray {hm['index'] + 1}: {original['name']}",
                         use_container_width=True)
                c2.image(base64.b64decode(hm["png_base64"]), caption=f"Strongest activation: {hm['region']}",
                         use_container_width=True)
            st.caption("Red areas are where the image model found the most evidence. The region name assumes a front "
                       "view (the patient's right side is on the left of the picture). Use it as a pointer, not a diagnosis.")

    with tab_words:
        expl = res.get("explanation")
        if not res["text"].get("has_note"):
            st.info("No usable clinical note was provided.")
        elif not expl or not expl["tokens"]:
            st.info("Word highlighting is not available for this case.")
        else:
            st.markdown(f"Words that pushed the note model toward **{pretty(expl['label'])}**:")
            st.markdown(word_html(expl["tokens"]), unsafe_allow_html=True)
            st.caption("Red raises the probability of this finding, blue lowers it. Stronger colour = stronger effect.")
            if expl["top"]:
                st.markdown("**Strongest words:** " + ", ".join(t["word"] for t in expl["top"]))
            st.caption(f"The note was cleaned before analysis: \"{res['text'].get('clean_text', '')}\"")

    # ---- deep explanation (SHAP + optional LLM sentence)
    with tab_deep:
        st.write("A slower, more careful explanation with SHAP: which words of the note and which part of the X-ray "
                 "support the finding. It can take a few minutes on a CPU.")
        d1, d2, d3 = st.columns(3)
        top_labels = [d["label"] for d in fusion.get("top", []) if d["label"] not in HIDDEN_LABELS]
        target_choice = d1.selectbox("Finding to explain", ["Most likely one"] + [pretty(l) for l in top_labels])
        deep_image = d2.checkbox("Explain the X-ray too (slower)", value=True)
        deep_evals = d3.slider("Detail level", 100, 600, 300, 100,
                               help="More = steadier result but slower.")
        if st.button("Run deep explanation", type="primary"):
            target = None if target_choice == "Most likely one" else top_labels[[pretty(l) for l in top_labels].index(target_choice)]
            with st.spinner("Running SHAP... this can take a few minutes."):
                try:
                    st.session_state["deep"] = call_explain(api_url, res["_files"], res["_note"], target, deep_image,
                                                            deep_evals, use_llm, llm_key, llm_model)
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the API.")
                except Exception as e:                                       # noqa: BLE001
                    st.error(f"The deep explanation failed. {e}")
        deep = st.session_state.get("deep")
        if deep:
            st.markdown(f"**Finding explained:** {pretty(deep['target'])}")
            if deep.get("llm_text"):
                st.info(deep["llm_text"])
            elif deep.get("llm_error"):
                st.warning(f"AI sentence not available: {deep['llm_error']}")
            st.markdown(strip_probs(deep["evidence"]))
            for w in deep.get("warnings", []):
                st.warning(w)
            wc = deep.get("word_contributions") or []
            if wc:
                peak = max(abs(x["value"]) for x in wc) or 1.0
                st.markdown("**Note words** (red supports the finding, blue speaks against it)")
                st.markdown(word_html([{"word": x["word"], "score": x["value"] / peak} for x in wc]),
                            unsafe_allow_html=True)
            if deep.get("overlay_png_base64"):
                e1, e2 = st.columns(2)
                e1.image(res["_files"][0]["bytes"], caption="X-ray 1", use_container_width=True)
                e2.image(base64.b64decode(deep["overlay_png_base64"]),
                         caption=f"Image area supporting it: {deep.get('region') or '-'}", use_container_width=True)
            st.caption(f"Time (s): {deep.get('timings', {})}")

    with tab_tech:
        st.write(f"X-ray images: {res['n_images']}")
        st.write("Time per step (seconds):", res.get("timings", {}))
        st.write("Urgent findings used by the priority rule:", ", ".join(pretty(u) for u in urgent) or "none")
        with st.expander("Raw API response"):
            st.json({k: v for k, v in res.items() if not k.startswith("_") and k != "heatmaps"})