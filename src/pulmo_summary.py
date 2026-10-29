import re

SYMPTOMS = {
    "chest pain":           [r"chest pain", r"\bcp\b", r"chest discomfort", r"chest tightness", r"chest pressure"],
    "shortness of breath":  [r"shortness of breath", r"short of breath", r"\bsob\b", r"dyspn(?:o)?ea", r"difficulty breathing",
                             r"breathless\w*", r"\bdoe\b"],
    "cough":                [r"\bcough(?:ing|s)?\b"],
    "sputum":               [r"sputum", r"productive"],
    "fever":                [r"fevers?\b", r"febrile", r"pyrexia"],
    "chills":               [r"chills"],
    "coughing blood":       [r"ha?emoptysis", r"coughing up blood"],
    "wheezing":             [r"wheez\w*"],
    "dizziness / fainting": [r"syncop\w*", r"presyncop\w*", r"dizz\w*", r"lightheaded\w*", r"passed out"],
    "fatigue / weakness":   [r"fatigue", r"weakness", r"malaise", r"lethargy"],
    "weight loss":          [r"weight loss"],
    "palpitations":         [r"palpitations?", r"tachycardi\w*"],
    "low oxygen":           [r"hypoxi\w*", r"desaturation", r"low oxygen"],
    "swelling":             [r"o?edema", r"swelling"],
    "abdominal pain":       [r"abdominal pain", r"\babd pain\b", r"epigastric pain"],
    "back / rib pain":      [r"back pain", r"rib pain", r"flank pain", r"shoulder pain"],
    "nausea / vomiting":    [r"nausea", r"vomit\w*"],
}
HISTORY = {
    "smoker":                 [r"smok\w*", r"tobacco", r"pack years?"],
    "COPD / emphysema":       [r"\bcopd\b", r"emphysema"],
    "asthma":                 [r"asthma\w*"],
    "cancer":                 [r"cancer", r"carcinoma", r"malignan\w*", r"tumou?r", r"lymphoma", r"metasta\w*"],
    "tuberculosis / TB test": [r"\btb\b", r"tubercul\w*", r"\bppd\b", r"quantiferon"],
    "heart failure":          [r"\bchf\b", r"heart failure", r"cardiomyopathy"],
    "coronary disease":       [r"\bcad\b", r"coronary", r"myocardial infarction", r"\bcabg\b", r"\bstemi\b"],
    "high blood pressure":    [r"hypertension", r"\bhtn\b"],
    "diabetes":               [r"diabet\w*", r"\bdm\b"],
    "pneumonia":              [r"pneumonia", r"\bpna\b"],
    "blood clot (PE / DVT)":  [r"pulmonary embol\w*", r"\bdvt\b", r"\bpe\b"],
    "HIV":                    [r"\bhiv\b"],
    "trauma / fall":          [r"trauma\w*", r"\bfall\b", r"\bfell\b", r"\bmva\b", r"\bmvc\b", r"accident", r"injur\w*", r"gunshot"],
}
PURPOSE = {
    "pre-operative check":  [r"pre[- ]?op\w*", r"before surgery", r"surgical clearance"],
    "routine / screening":  [r"routine", r"screening", r"\bphysical\b", r"annual", r"employment", r"immigration", r"check ?up"],
    "follow-up":            [r"follow[- ]?up", r"\bf/?u\b", r"\bre-?eval\w*"],
    "line / tube position": [r"line placement", r"tube placement", r"\bpicc\b", r"\bport placement\b", r"\bng tube\b", r"\bet tube\b"],
}
LEXICON = {"SYMPTOM": SYMPTOMS, "CONDITION": HISTORY, "PURPOSE": PURPOSE}

NER_MODEL = "d4data/biomedical-ner-all"
NER_TYPES = {"Sign_symptom": "SYMPTOM", "Disease_disorder": "CONDITION", "History": "CONDITION", "Family_history": "CONDITION",
             "Diagnostic_procedure": "OTHER", "Therapeutic_procedure": "OTHER", "Medication": "OTHER",
             "Duration": "DURATION", "Age": "AGE", "Sex": "SEX"}
NER_MODIFIERS = {"Biological_structure", "Severity", "Detailed_description", "Qualitative_concept"}
HEADS = {"Sign_symptom", "Disease_disorder"}
NER_MIN_SCORE = 0.5

CONTEXT_RULES = [
    *[(c, "POSSIBLE_EXISTENCE", "FORWARD") for c in ["evaluate for", "eval for", "evaluation for", "assess for", "assessment for",
                                                     "concern for", "question of", "suspected", "suspect", "possible", "probable"]],
    *[(c, "HISTORICAL", "FORWARD") for c in ["history of", "hx", "hx of", "h/o", "previous", "prior", "past", "known",
                                             "status post", "s/p"]],
    *[(c, "FAMILY", "FORWARD") for c in ["family history of", "fhx", "mother", "father", "sister", "brother"]],
    *[(c, "TERMINATE", "TERMINATE") for c in [",", ";", "but", "however", "now", "with", "presents with", "presenting with",
                                              "complains of", "c/o", "currently", "positive"]],
]
CONTEXT_REMOVE = {"evaluate for", "checked for", "positive", "presentation", "comes in with"}
CONTEXT_MAX_SCOPE = 6
_CATS = ["NEGATED_EXISTENCE", "POSSIBLE_EXISTENCE", "HISTORICAL", "HYPOTHETICAL", "FAMILY"]
CONTEXT_TERMINATING = {c: [o for o in _CATS if o != c] for c in _CATS}

_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|several|few|a|1-2|2-3)"
_UNIT = r"(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)"
DURATION = re.compile(rf"\b(?:for|x|since|past|over|last)\s*(?:the\s+)?({_NUM}\s*{_UNIT})\b|\b({_NUM}\s*{_UNIT})\s+ago\b")
NUM_UNIT = re.compile(rf"\b({_NUM}\s*{_UNIT})\b")
AGE = re.compile(r"\b(\d{1,3})[- ]?(?:years?|yrs?|yo|y/o)[- ]?(?:old)?\b")
SEX = {"male": [r"\bmale\b", r"\bman\b", r"\bgentleman\b", r"\bboy\b", r"\bm\b"],
       "female": [r"\bfemale\b", r"\bwoman\b", r"\blady\b", r"\bgirl\b", r"\bf\b"]}
NOISE = re.compile(r"hidden|xxxx|^\W*$")
SEX_WORDS = {"male", "female", "man", "woman", "gentleman", "lady", "boy", "girl"}
GENERIC = {"evaluation", "eval", "evaluate", "status", "pt", "patient", "history", "hx", "exam", "examination", "workup",
           "check", "symptoms", "complaints", "problem", "problems", "disease", "condition", "loss", "sore", "onset", "shielded",
           "asymptomatic"}

PRETTY = {"lung": "lung abnormality", "opacity": "lung opacity", "markings": "prominent lung markings",
          "density": "increased density", "spine": "spine changes", "thoracic vertebrae": "thoracic spine changes",
          "technical quality of image unsatisfactory": "poor image quality", "catheters, indwelling": "indwelling catheter",
          "fractures, bone": "bone fracture", "aorta": "aortic changes", "diaphragm": "diaphragm changes",
          "cicatrix": "scarring", "calcinosis": "calcification", "pulmonary atelectasis": "atelectasis"}
SKIP_LABELS = {"no indexing"}

def pretty(label):
    return PRETTY.get(label, label)

_STATE = {"nlp": None, "matcher": None, "context": None, "ner": None, "method": "rules", "cache": {}}

def _pipeline():
    if _STATE["nlp"] is None:
        import spacy
        from medspacy.target_matcher import TargetMatcher, TargetRule
        from medspacy.context import ConText, ConTextRule
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        matcher = TargetMatcher(nlp)
        matcher.add([TargetRule(concept, etype, pattern="|".join(pats))
                     for etype, group in LEXICON.items() for concept, pats in group.items()])
        default = [r for r in ConText(nlp, rules="default").rules if r.literal not in CONTEXT_REMOVE]
        context = ConText(nlp, rules=None, max_scope=CONTEXT_MAX_SCOPE, terminating_types=CONTEXT_TERMINATING)
        context.add(default + [ConTextRule(lit, cat, direction=d) for lit, cat, d in CONTEXT_RULES])
        _STATE.update(nlp=nlp, matcher=matcher, context=context)
    return _STATE["nlp"], _STATE["matcher"], _STATE["context"]

def load_ner(model=NER_MODEL, device=None):
    try:
        if callable(model):
            _STATE["ner"] = model
        else:
            import torch
            from transformers import pipeline
            dev = device if device is not None else (0 if torch.cuda.is_available() else -1)
            _STATE["ner"] = pipeline("token-classification", model=model, aggregation_strategy="simple", device=dev)
        _STATE["method"], _STATE["cache"] = "hybrid", {}
        return True
    except Exception as e:
        print(f"NER model not loaded ({type(e).__name__}: {e}); using the lexicon matcher.")
        return False

def set_method(method):
    assert method in ("rules", "ner", "hybrid")
    if method != "rules" and _STATE["ner"] is None:
        raise ValueError("load the NER model first: load_ner()")
    _STATE["method"] = method

def get_method():
    return _STATE["method"]

def _ner_entities(text):
    if text not in _STATE["cache"]:
        raw = [e for e in sorted(_STATE["ner"](text), key=lambda e: e["start"]) if float(e["score"]) >= NER_MIN_SCORE]
        touching = lambda a, b: not text[a["end"]:b["start"]].strip()
        out, pending = [], None
        for e in raw:
            label = e["entity_group"]
            if label in NER_MODIFIERS:
                pending = (pending[0] if pending and touching({"end": pending[1]}, e) else e["start"], e["end"])
                continue
            start = e["start"]
            if pending and label in HEADS and touching({"end": pending[1]}, e):
                start = pending[0]
            pending = None
            same_kind = out and NER_TYPES.get(out[-1][2]) == NER_TYPES.get(label) in ("SYMPTOM", "CONDITION")
            if same_kind and touching({"end": out[-1][1]}, e):
                out[-1] = (out[-1][0], e["end"], out[-1][2])
            elif label in NER_TYPES:
                out.append((start, e["end"], label))
        _STATE["cache"][text] = out
    return _STATE["cache"][text]

def _normalise(span_text, etype):
    low = re.sub(r"\s+", " ", span_text.lower()).strip(" -/")
    for t in [etype] + [x for x in LEXICON if x != etype]:
        hits = [c for c, pats in LEXICON.get(t, {}).items() if any(re.search(p, low) for p in pats)]
        if hits:
            return hits, t
    if low in GENERIC or len(low) < 3:
        return [], etype
    return [low], etype

def _trim_noise(span):
    bad = lambda t: NOISE.search(t.text.lower()) or (span.label_ != "SEX" and t.text.lower() in SEX_WORDS)
    keep, best = [], []
    for tok in span:
        if bad(tok):
            keep = []
        else:
            keep.append(tok)
            best = list(keep)
    while best and (best[0].is_stop or best[0].is_punct):
        best = best[1:]
    while best and (best[-1].is_stop or best[-1].is_punct):
        best = best[:-1]
    if not best:
        return None
    return span.doc.char_span(best[0].idx, best[-1].idx + len(best[-1].text), label=span.label_)

def analyze(note, method=None):
    from spacy.util import filter_spans
    nlp, matcher, context = _pipeline()
    method = method or _STATE["method"]
    doc = nlp(str(note))
    spans = list(matcher(nlp(str(note))).ents) if method in ("rules", "hybrid") else []
    spans = [doc.char_span(s.start_char, s.end_char, label=s.label_) for s in spans]
    if method in ("ner", "hybrid") and _STATE["ner"] is not None:
        for start, end, label in _ner_entities(str(note)):
            s = doc.char_span(start, end, label=NER_TYPES[label], alignment_mode="expand")
            if s is not None:
                s = _trim_noise(s)
            if s is not None:
                spans.append(s)
    doc.ents = filter_spans([s for s in spans if s is not None])
    return context(doc)

def extract_clinical_info(note, method=None):
    info = {"symptoms": [], "denied": [], "history": [], "purpose": [], "other": [], "duration": None, "age": None, "sex": None,
            "entities": []}
    if not note or not str(note).strip():
        return info
    add = lambda key, val: info[key].append(val) if val not in info[key] else None
    doc = analyze(note, method)
    for ent in doc.ents:
        etype = ent.label_
        if etype in ("DURATION", "AGE", "SEX"):
            low = ent.text.lower()
            if etype == "DURATION" and info["duration"] is None and NUM_UNIT.search(low):
                info["duration"] = NUM_UNIT.search(low).group(1)
            elif etype == "AGE" and info["age"] is None and re.search(r"\d+", low):
                info["age"] = int(re.search(r"\d+", low).group())
            elif etype == "SEX":
                info["sex"] = next((s for s, pats in SEX.items() if any(re.search(p, low) for p in pats)), info["sex"])
            continue
        concepts, etype = _normalise(ent.text, etype)
        if ent._.is_negated:
            status = "denied"
        elif ent._.is_family:
            status = "family"
        elif ent._.is_uncertain or ent._.is_hypothetical:
            status = "possible"
        elif ent._.is_historical:
            status = "historical"
        else:
            status = "present"
        for c in concepts:
            info["entities"].append({"text": ent.text, "concept": c, "type": etype, "status": status})
            if etype == "PURPOSE":
                add("purpose", c)
            elif status == "denied":
                add("denied", c)
            elif status == "family":
                add("history", f"family history of {c}")
            elif status == "possible":
                add("purpose", f"possible {c}")
            elif etype == "OTHER":
                add("other", c)
            elif etype == "SYMPTOM" and status == "historical":
                add("history", f"previous {c}")
            elif etype == "SYMPTOM":
                add("symptoms", c)
            else:
                add("history", c)

    low = " " + str(note).lower() + " "
    if info["duration"] is None and DURATION.search(low):
        m = DURATION.search(low)
        info["duration"] = (m.group(1) or m.group(2)).strip()
    if info["age"] is None and AGE.search(low):
        info["age"] = int(AGE.search(low).group(1))
    if info["sex"] is None:
        info["sex"] = next((s for s, pats in SEX.items() if any(re.search(p, low) for p in pats[:-1])), None)
    if "sputum" in info["symptoms"] and "cough" in info["symptoms"]:
        info["symptoms"] = [s if s != "cough" else "cough with sputum" for s in info["symptoms"] if s != "sputum"]
    return info

def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]

def _findings_list(result, k=3):
    if not result:
        return []
    items = result.get("findings") or result.get("top") or []
    items = [d for d in items if d["label"] not in SKIP_LABELS and d["label"] != "normal"]
    return items[:k]

NEXT_STEP = {
    "high":   "Review first: radiologist read as soon as possible.",
    "medium": "Review today in the normal queue.",
    "low":    "Routine review.",
}
ORDER = ["presentation", "reason", "history", "denied", "other", "assessment", "image", "note_model", "key_words", "priority",
         "disclaimer"]

def make_summary(note, text_result=None, image_result=None, fusion_result=None, priority=None, key_words=None):
    info = extract_clinical_info(note)
    sections, facts = {}, {"note_terms": {k: v for k, v in info.items() if k != "entities"}, "findings": []}

    who = {"male": "male patient", "female": "female patient"}.get(info["sex"], "patient")
    who = f"{info['age']}-year-old {who}" if info["age"] else who.capitalize()
    if info["symptoms"]:
        dur = f" for {info['duration']}" if info["duration"] else ""
        sections["presentation"] = f"{who} presenting with {_join(info['symptoms'])}{dur}."
    else:
        sections["presentation"] = f"{who} with " + ("no symptoms stated in the note." if note and str(note).strip()
                                                      else "no clinical note.")
    if info["purpose"]:
        sections["reason"] = f"Reason for exam: {_join(info['purpose'])}."
    if info["history"]:
        sections["history"] = f"History: {_join(info['history'])}."
    if info["denied"]:
        sections["denied"] = f"Denies / no: {_join(info['denied'])}."
    if info["other"]:
        sections["other"] = f"Also mentioned: {_join(info['other'])}."

    def describe(name, result, extra=""):
        items = _findings_list(result)
        facts["findings"] += [d["label"] for d in items]
        if not items:
            return f"{name}: no specific finding stands out{extra}."
        listed = _join(f"{pretty(d['label'])} ({d['probability']:.0%})" for d in items)
        return f"{name}: most likely {listed}{extra}."

    if fusion_result:
        has_note = bool(note and str(note).strip())
        sections["assessment"] = describe("Combined image + note assessment" if has_note else "Combined assessment", fusion_result)
    if image_result:
        region = f", mainly in the {image_result['region']}" if image_result.get("region") else ""
        sections["image"] = describe("Image analysis", image_result, region)
    if text_result and text_result.get("has_note"):
        pn = text_result.get("p_normal")
        normal = f" Probability of a normal study from the note alone: {pn:.0%}." if pn is not None else ""
        sections["note_model"] = describe("From the note alone", text_result) + normal
        if key_words:
            sections["key_words"] = f"Words that influenced the note analysis: {', '.join(key_words[:5])}."
    elif text_result is not None:
        sections["note_model"] = "Note analysis skipped: no usable clinical note."

    if priority:
        sections["priority"] = f"Priority: {priority.upper()}. {NEXT_STEP.get(priority.lower(), '')}".strip()
    elif not (image_result or fusion_result):
        sections["priority"] = "Priority: pending the image analysis."

    sections["disclaimer"] = "AI-assisted summary (educational prototype). The final decision is the doctor's."
    text = " ".join(sections[k] for k in ORDER if k in sections)
    return {"sections": sections, "text": text, "facts": facts}

def check_summary(summary, note, text_result=None, image_result=None, fusion_result=None):
    problems = []
    info = extract_clinical_info(note)
    for key in ("symptoms", "history", "denied", "purpose", "other"):
        for term in summary["facts"]["note_terms"].get(key, []):
            if term not in info[key]:
                problems.append(f"'{term}' is not in the note")
    allowed = set()
    for r in (text_result, image_result, fusion_result):
        if r:
            allowed |= {d["label"] for d in (r.get("findings") or []) + (r.get("top") or [])}
    for label in summary["facts"]["findings"]:
        if label not in allowed:
            problems.append(f"finding '{label}' was not predicted by any model")
    for term in info["denied"]:
        if term in info["symptoms"]:
            problems.append(f"'{term}' listed as both present and denied")
    return problems
